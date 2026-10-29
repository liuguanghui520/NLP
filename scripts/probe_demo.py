"""Real local demo requests, separate from formal metrics and model selection."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8017")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(path, body=None):
        raw = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(args.base_url + path, data=raw, headers={"Content-Type":"application/json", "Authorization":"Bearer nlp-local-demo"})
        with opener.open(req, timeout=180) as response:
            return json.loads(response.read())

    cases = request("/demo/cases")["cases"][:3]
    listed = request("/v1/models")
    available = [row["id"] for row in listed["data"]]
    order = ["nlp-qwen-lora-t01-seed42", "nlp-tfidf-seed42", "nlp-macbert-seed42", "nlp-qwen-zero-shot", "nlp-qwen-four-shot"]
    assert set(available) == set(order)
    frozen_names = {"nlp-tfidf-seed42":"tfidf-char_1_3-seed42", "nlp-macbert-seed42":"bert-len512-seed42", "nlp-qwen-zero-shot":"test-qwen_zero_shot", "nlp-qwen-four-shot":"test-qwen_few_shot", "nlp-qwen-lora-t01-seed42":"test-lora-seed42"}
    results = []
    try:
        for model in order:
            for case in cases:
                started = time.perf_counter()
                response = request("/v1/chat/completions", dict(model=model, messages=[dict(role="user", content=case["comment"])], stream=False, temperature=0, max_tokens=8))
                result = response["demo"]
                assert result["raw_output"] == response["choices"][0]["message"]["content"]
                row = dict(model=model, case_id=case["id"], reference_label=case["expected_label"], raw_output=result["raw_output"], parsed_label=result["parsed_label"],
                    label_name=result["label_name"], status=result["status"], request_id=response["id"], latency_ms=result["latency_ms"], model_load_seconds=result["model_load_seconds"],
                    http_roundtrip_seconds=time.perf_counter()-started, input_sha256=result["input_sha256"], scope="demo_only_not_benchmark")
                if case.get("sample_id"):
                    prediction = next(json.loads(line) for line in (root / "results/predictions" / (frozen_names[model] + ".jsonl")).read_text(encoding="utf-8").splitlines() if json.loads(line)["sample_id"] == case["sample_id"])
                    row.update(sample_id=case["sample_id"], historical_raw_output=prediction["raw_output"], exact_historical_match=result["raw_output"] == prediction["raw_output"])
                results.append(row)
                (args.output / "predictions.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                print(model, case["id"], result["status"], result["parsed_label"], flush=True)
        # Empty input must fail before loading a different model or mutating results.
        try:
            request("/v1/chat/completions", dict(model=order[0], messages=[dict(role="user", content="   ")]))
            raise AssertionError("Empty input accepted")
        except urllib.error.HTTPError as error:
            assert error.code == 400
        # Leave LoRA loaded for the actual screen recording.
        warmup = request("/v1/chat/completions", dict(model=order[0], messages=[dict(role="user", content=cases[0]["comment"])], stream=False))
        raw = json.dumps(dict(model=order[0], messages=[dict(role="user", content=cases[0]["comment"])], stream=True), ensure_ascii=False).encode("utf-8")
        with opener.open(urllib.request.Request(args.base_url + "/v1/chat/completions", data=raw, headers={"Content-Type":"application/json"}), timeout=180) as response:
            stream = response.read().decode("utf-8")
        assert "data: [DONE]" in stream
        chunks = [json.loads(line[6:]) for line in stream.splitlines() if line.startswith("data: {")]
        assert "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks) == warmup["choices"][0]["message"]["content"]
        status = dict(status="passed", tested_models=order, real_predictions=len(results), historical_test_examples_compared=sum("exact_historical_match" in r for r in results),
            historical_raw_matches=sum(r.get("exact_historical_match",False) for r in results), empty_input_http=400, sse_stream_checked=True, final_loaded_model=order[0],
            training=False, inference=True, new_machine_model_reproduction=False, cherry_ui_verified=False, scope="Same-machine OpenAI-compatible service probe; not Cherry UI or full benchmark")
    except BaseException as error:
        status = dict(status="failed", error_type=type(error).__name__, error=str(error), real_predictions=len(results), training=False, inference=True)
        raise
    finally:
        status["completed_at"] = datetime.now(timezone(timedelta(hours=8))).isoformat()
        (args.output / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
