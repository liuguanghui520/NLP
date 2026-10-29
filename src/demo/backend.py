"""Single-comment inference with frozen seed42 artifacts and one GPU resident.

No experiment index, formal predictions or metrics are rewritten by this module.
"""
import copy
import gc
import json
from pathlib import Path
import sys
import threading
import time

from src.adapters.prompts import render, text_sha
from src.common.io import digest, load, rows
from src.evaluation.metrics import parse


MODELS = {
    "nlp-tfidf-seed42": "tfidf",
    "nlp-macbert-seed42": "bert",
    "nlp-qwen-zero-shot": "qwen_zero_shot",
    "nlp-qwen-four-shot": "qwen_few_shot",
    "nlp-qwen-lora-t01-seed42": "qwen_lora",
}


def extract_comment(messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages必须是非空数组")
    users = [message for message in messages if message.get("role") == "user"]
    if not users:
        raise ValueError("请发送一条评论")
    content = users[-1].get("content")
    if isinstance(content, list):
        if any(part.get("type") != "text" for part in content):
            raise ValueError("演示仅支持评论文本，不支持图片或附件")
        content = "".join(part.get("text", "") for part in content)
    if not isinstance(content, str) or not content.strip():
        raise ValueError("评论不能为空")
    if len(content) > 100000:
        raise ValueError("评论过长，请输入100000字符以内的文本")
    normalized = content.replace("\r\n", "\n").replace("\r", "\n").strip()
    return content, normalized, normalized[:512], len(messages) - 1


def prompt_registry(root, data, tokenizer):
    registry = copy.deepcopy(load(root / "configs/methods/prompt_registry.json"))
    if text_sha(tokenizer.chat_template or "") != registry["chat_template_sha256"]:
        raise ValueError("Qwen聊天模板哈希不匹配")
    train = {row["sample_id"]: row for row in rows(data / "train.jsonl")}
    examples = load(root / "configs/methods/few_shot_examples.json")["examples"]
    zero = registry["modes"]["qwen_zero_shot"]["messages_template"][0]["content"]
    few = registry["modes"]["qwen_few_shot"]
    assert few["example_ids"] == [example["sample_id"] for example in examples]
    for index, example in enumerate(examples):
        row = train[example["sample_id"]]
        assert row["label"] == example["label"]
        assert text_sha(row["normalized_text"]) == example["original_text_sha256"]
        content = zero.replace("{{comment}}", row["normalized_text"])
        assert text_sha(content) == few["messages_template"][index * 2]["content"].split("original_content_sha256=")[1]
        few["messages_template"][index * 2]["content"] = content
    for name, mode in registry["modes"].items():
        value = mode["messages_template"] if name == "qwen_zero_shot" else dict(messages=mode["messages_template"], example_ids=mode["example_ids"])
        assert text_sha(json.dumps(value, ensure_ascii=False, sort_keys=True)) == mode["template_sha256"]
    return registry


class Backend:
    def __init__(self, root, paths):
        self.root, self.paths = Path(root), paths
        self.lock = threading.RLock()
        self.model = self.tokenizer = self.registry = None
        self.loaded = self.checkpoint = None
        self.verified_qwen = False

    def unload(self):
        if self.loaded == "qwen_lora" and self.model is not None:
            self.model._loop.call_soon_threadsafe(self.model._loop.stop)
            self.model._thread.join(timeout=5)
            if not self.model._thread.is_alive():
                self.model._loop.close()
        self.model = self.tokenizer = self.registry = None
        self.loaded = self.checkpoint = None
        gc.collect()
        if "torch" in sys.modules:
            sys.modules["torch"].cuda.empty_cache()

    def ensure(self, method):
        if self.loaded == method:
            return 0.0
        self.unload()
        started = time.perf_counter()
        if method in ("tfidf", "bert"):
            base = Path(self.paths["baseline_results" if method == "tfidf" else "bert_results"])
            winner = load(base / "frozen_winner.json")["config"]
            selection = load(base / f"{method}-{winner['config_id']}-seed42/selection.json")
            artifact = Path(selection["model"])
            target = artifact / "model.safetensors" if artifact.is_dir() else artifact
            assert digest(target) == selection["model_sha256"]
            self.checkpoint = selection["model_sha256"]
            if method == "tfidf":
                import joblib
                self.model = joblib.load(artifact)
            else:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                torch.set_num_threads(4)
                self.tokenizer = AutoTokenizer.from_pretrained(artifact, local_files_only=True)
                self.model = AutoModelForSequenceClassification.from_pretrained(artifact, local_files_only=True).cuda().eval()
                self.maxlen = winner["maxlen"]
        else:
            import torch
            torch.set_num_threads(4)
            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                raise RuntimeError("Qwen演示要求CUDA和BF16；不自动改精度或模型")
            lock = load(self.root / "artifacts/lora-lock.json")
            if not self.verified_qwen:
                for name, value in lock["base_model_files"].items():
                    assert digest(Path(self.paths["qwen_model"]) / name) == value, name
                self.verified_qwen = True
            if method == "qwen_lora":
                sys.path.insert(0, self.paths["factory_root"] + "/src")
                from llamafactory.chat import ChatModel
                adapter = Path(self.paths["lora_adapters"]["42"])
                expected = next(row for row in lock["adapters"] if row["seed"] == 42)
                assert digest(adapter / "adapter_model.safetensors") == expected["adapter_sha256"]
                assert digest(adapter / "adapter_config.json") == expected["adapter_config_sha256"]
                self.checkpoint = expected["adapter_sha256"]
                self.model = ChatModel(dict(model_name_or_path=self.paths["qwen_model"], adapter_name_or_path=str(adapter),
                    template="qwen3", finetuning_type="lora", infer_backend="huggingface", infer_dtype="bfloat16",
                    enable_thinking=False, trust_remote_code=False, do_sample=False, max_new_tokens=8, repetition_penalty=1.0))
                self.tokenizer = self.model.engine.tokenizer
            else:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.paths["qwen_model"], local_files_only=True)
                self.registry = prompt_registry(self.root, Path(self.paths["data"]), self.tokenizer)
                self.model = AutoModelForCausalLM.from_pretrained(self.paths["qwen_model"], local_files_only=True,
                    dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="sdpa").cuda().eval()
        self.loaded = method
        if method != "tfidf":
            import torch
            torch.cuda.synchronize()
        return time.perf_counter() - started

    def predict(self, model_id, messages):
        if model_id not in MODELS:
            raise ValueError("未知模型，请从/v1/models选择")
        received, normalized, body, ignored = extract_comment(messages)
        method = MODELS[model_id]
        with self.lock:
            load_seconds = self.ensure(method)
            if method != "tfidf":
                import torch
                torch.cuda.synchronize()
            started = time.perf_counter()
            input_tokens = output_tokens = None
            truncated = False
            if method == "tfidf":
                raw = str(int(self.model.predict([body])[0]))
                rendered = body
            elif method == "bert":
                encoded = self.tokenizer(body, truncation=True, max_length=self.maxlen, return_tensors="pt").to("cuda")
                input_tokens = encoded.input_ids.shape[1]
                # Full count is logged for demo inspection, outside the model tensor path.
                truncated = len(self.tokenizer(body)["input_ids"]) > self.maxlen
                with torch.inference_mode():
                    raw = str(self.model(**encoded).logits.argmax(-1).item())
                rendered = body
            elif method == "qwen_lora":
                instruction = load(self.root / "configs/methods/lora-frozen.json")["inference"]["instruction"]
                original = body
                while True:
                    rendered = instruction + "\n" + body
                    prompt_messages = [dict(role="user", content=rendered)]
                    ids, _ = self.model.engine.template.encode_oneturn(self.tokenizer, prompt_messages + [dict(role="assistant", content="")], None, None)
                    if len(ids) <= 2048:
                        break
                    if not body:
                        raise ValueError("任务说明超过输入预算")
                    body = body[:-1]
                truncated = body != original
                response, = self.model.chat(prompt_messages, do_sample=False, max_new_tokens=8, repetition_penalty=1.0)
                raw, input_tokens, output_tokens = response.response_text, response.prompt_length, response.response_length
            else:
                rendered, actual, input_tokens = render(self.tokenizer, self.registry["modes"][method], body)
                truncated, body = actual != body, actual
                encoded = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to("cuda")
                with torch.inference_mode():
                    generated = self.model.generate(**encoded, do_sample=False, num_beams=1, max_new_tokens=8,
                        use_cache=True, pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
                continuation = generated[0, encoded.input_ids.shape[1]:]
                raw, output_tokens = self.tokenizer.decode(continuation, skip_special_tokens=True), len(continuation)
            label = parse(raw)
            if method != "tfidf":
                torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - started) * 1000
            return dict(model=model_id, method=method, seed=42 if method in ("tfidf", "bert", "qwen_lora") else None,
                raw_output=raw, parsed_label=label, label_name={0:"负面", 1:"正面"}.get(label, "非法输出"),
                status="ok" if label is not None else "invalid_format", received_comment=received, normalized_comment=normalized,
                actual_comment=body, actual_input=rendered, input_sha256=text_sha(rendered), input_chars=len(body),
                char_truncated=len(normalized)>512, token_truncated=truncated, input_tokens=input_tokens, output_tokens=output_tokens,
                latency_ms=latency_ms, model_load_seconds=load_seconds, ignored_messages=ignored,
                checkpoint_sha256=self.checkpoint, scope="interactive_demo_only",
                timing_scope="single request; loaded model; tokenizer/transfer/inference/parse; GPU synchronized; extra demo metadata included; not formal benchmark")
