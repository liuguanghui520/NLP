"""Build traceable offline comparisons and scientific figures from existing records."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
import hashlib
import html
import json
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.metrics import evaluate

ORDER = ["tfidf", "bert", "qwen_zero_shot", "qwen_few_shot", "qwen_lora"]
NAMES = dict(tfidf="TF-IDF", bert="MacBERT", qwen_zero_shot="Qwen零样本",
             qwen_few_shot="Qwen少样本", qwen_lora="LoRA T01", majority="多数类参考")
COLORS = dict(zip(ORDER, ["#526b8b", "#167685", "#b58437", "#9470a4", "#c35959"]))


def macro_f1(y, predictions):
    """Fixed labels 0/1; -1 is an invalid prediction and contributes an FN."""
    scores = []
    y, predictions = np.asarray(y), np.asarray(predictions)
    for c in [0, 1]:
        tp = np.sum((y == c) & (predictions == c), axis=-1)
        fp = np.sum((y != c) & (predictions == c), axis=-1)
        fn = np.sum((y == c) & (predictions != c), axis=-1)
        denom = 2 * tp + fp + fn
        scores.append(np.divide(2 * tp, denom, out=np.zeros_like(denom, dtype=float), where=denom != 0))
    return (scores[0] + scores[1]) / 2


def holm(pvalues):
    result = [0.] * len(pvalues)
    running = 0.
    for position, i in enumerate(sorted(range(len(pvalues)), key=lambda j: pvalues[j])):
        running = max(running, min(1., (len(pvalues) - position) * pvalues[i]))
        result[i] = running
    return result


def paired_bootstrap(y, models, *, seed, resamples, batch=250):
    """Same stratified sample indices for all methods and all fixed trained seeds."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    classes = [np.flatnonzero(y == c) for c in [0, 1]]
    if any(len(group) == 0 for group in classes):
        raise ValueError("Stratified bootstrap requires both true classes")
    result = {method: [] for method in models}
    for start in range(0, resamples, batch):
        count = min(batch, resamples - start)
        indices = np.concatenate([rng.choice(group, size=(count, len(group)), replace=True) for group in classes], axis=1)
        truth = y[indices]
        for method, predictions in models.items():
            result[method].extend(np.mean([macro_f1(truth, p[indices]) for p in predictions], axis=0).tolist())
    return {k: np.array(v) for k, v in result.items()}


class Report:
    def __init__(self, config_path, run_id):
        if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in run_id):
            raise ValueError("Use a simple unique run ID")
        self.sources = {}
        self.config = self.read(config_path)
        self.id = run_id
        self.output = ROOT / "results/analysis" / run_id
        self.figures = ROOT / "reports/figures" / run_id
        self.report = ROOT / "reports/analysis" / run_id
        for path in [self.output, self.figures, self.report]:
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite: {path}")
        for path in [self.output, self.figures, self.report]:
            path.mkdir(parents=True)
        self.started = time.perf_counter()
        self.figure_index = []
        self.log = []
        self.config_path = config_path

    def read(self, path, lines=False):
        path = Path(path)
        if not path.is_absolute():
            path = ROOT / path
        payload = path.read_bytes()
        self.sources[path.relative_to(ROOT).as_posix()] = hashlib.sha256(payload).hexdigest()
        text = payload.decode("utf-8")
        return [json.loads(line) for line in text.splitlines() if line.strip()] if lines else json.loads(text)

    def write(self, name, value):
        (self.output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    def note(self, text):
        self.log.append(text)
        print(text, flush=True)

    def load_predictions(self):
        self.data = self.read("data/processed/v1-20261023/test.jsonl", lines=True)
        self.y = np.array([r["label"] for r in self.data])
        assert len(self.data) == 777 and Counter(self.y) == {0: 245, 1: 532}
        self.index = [r for r in self.read("results/run_index.jsonl", lines=True) if r["split"] == "test"]
        assert len(self.index) == 11
        self.records, self.models, self.metrics = {}, defaultdict(list), {}
        for r in self.index:
            records = self.read(f'results/predictions/{r["run_id"]}.jsonl', lines=True)
            check = evaluate(records, self.data)
            assert check["complete"], check["errors"]
            for metric in ["macro_f1", "accuracy", "weighted_f1", "invalid_rate"]:
                assert abs(check["metrics"][metric] - r["metrics"][metric]) < 1e-12
            self.records[r["run_id"]] = records
            prediction = np.array([p["parsed_label"] if p["status"] == "ok" else -1 for p in records])
            assert abs(float(macro_f1(self.y, prediction)) - r["metrics"]["macro_f1"]) < 1e-12
            self.models[r["method"]].append(prediction)
            self.metrics[r["run_id"]] = check["metrics"]
        self.reps = {m: next(r for r in self.index if r["method"] == m and r["seed"] in (42, None)) for m in ORDER}
        self.representative = {m: np.array([p["parsed_label"] if p["status"] == "ok" else -1 for p in self.records[r["run_id"]]]) for m, r in self.reps.items()}
        if self.config["include_majority_reference"]:
            train = self.read("data/processed/v1-20261023/train.jsonl", lines=True)
            counts = Counter(r["label"] for r in train)
            label = sorted(counts, key=lambda c: (-counts[c], c))[0]
            prediction = np.full(len(self.y), label)
            self.models["majority"] = [prediction]
            self.representative["majority"] = prediction
            self.write("majority-reference.json", dict(source="training labels only", train_counts=dict(counts), label=int(label),
                       n=len(self.y), accuracy=float(np.mean(prediction == self.y)), macro_f1=float(macro_f1(self.y, prediction)), cost_ranking=False))
            template = next(iter(self.records.values()))
            synthetic = [dict(p, run_id="majority-reference", method="majority", config_id="train-majority", seed=None,
                              raw_output=str(label), parsed_label=int(label), status="ok") for p in template]
            self.majority_metrics = evaluate(synthetic, self.data)["metrics"]
            self.write("majority-reference.json", dict(source="training labels only", train_counts=dict(counts), label=int(label),
                **self.majority_metrics, cost_ranking=False, reference_is_synthetic=True))
        self.summary = []
        for method in [*ORDER, *(["majority"] if "majority" in self.models else [])]:
            runs = [r for r in self.index if r["method"] == method]
            metrics = [self.metrics[r["run_id"]] for r in runs]
            if method == "majority":
                pred = self.models[method][0]
                metrics = [self.majority_metrics]
            row = dict(method=method, n=777, runs=[r["run_id"] for r in runs], seeds=[r["seed"] for r in runs],
                       count=len(metrics), per_seed=[dict(seed=r["seed"], **self.metrics[r["run_id"]]) for r in runs])
            for key in ["macro_f1", "accuracy", "weighted_f1", "invalid_rate"]:
                values = [m[key] for m in metrics if m[key] is not None]
                row[key] = dict(mean=statistics.mean(values) if values else None, sample_std=statistics.stdev(values) if len(values) > 1 else None)
            row["per_class"] = {str(c): {metric: statistics.mean(m["per_class"][str(c)][metric] for m in metrics) for metric in ["precision", "recall", "f1", "support"]} for c in [0, 1]}
            if method == "majority":
                row["per_seed"] = [dict(seed=None, **self.majority_metrics)]
            self.summary.append(row)
        self.write("methods.json", self.summary)
        self.note("Audited 11 test runs; all main metrics independently reproduced within 1e-12.")

    def pair_analysis(self):
        config = self.config
        values = paired_bootstrap(self.y, self.models, seed=config["analysis_seed"], resamples=config["bootstrap_resamples"], batch=config["bootstrap_batch"])
        self.pairs = []
        for a, b in config["comparison_pairs"]:
            delta = values[a] - values[b]
            ac, bc = self.representative[a] == self.y, self.representative[b] == self.y
            gain, loss = int(np.sum(ac & ~bc)), int(np.sum(~ac & bc))
            actual = statistics.mean(float(macro_f1(self.y, p)) for p in self.models[a]) - statistics.mean(float(macro_f1(self.y, p)) for p in self.models[b])
            self.pairs.append(dict(a=a, b=b, direction="A minus B", macro_f1_mean_delta=actual,
                bootstrap_95_percentile=np.quantile(delta, [.025, .975], method="linear").tolist(),
                bootstrap_scope=config["bootstrap_scope"], both_correct=int(np.sum(ac & bc)), both_wrong=int(np.sum(~ac & ~bc)),
                A_correct_B_wrong=gain, A_wrong_B_correct=loss, correctness_seed=42,
                accuracy_delta_seed42=float(np.mean(ac) - np.mean(bc)),
                exact_mcnemar_p=float(binomtest(gain, gain + loss, .5).pvalue) if gain + loss else 1.,
                statistical_scope="post-hoc exploratory; correctness test is not an F1 test; CI is pointwise"))
        adjusted = holm([r["exact_mcnemar_p"] for r in self.pairs])
        for row, p in zip(self.pairs, adjusted):
            row["holm_adjusted_p_seven_pairs"] = p
        self.write("paired-comparisons.json", self.pairs)
        self.note(f"Computed {config['bootstrap_resamples']} paired stratified bootstrap samples; seven separate exploratory comparisons.")

    def group_analysis(self):
        self.groups = []
        bins = self.config["length_bins"]
        groups = [(f"length_{lo}_{hi or 'plus'}", [lo <= r["normalized_char_count"] and (hi is None or r["normalized_char_count"] <= hi) for r in self.data]) for lo, hi in bins]
        groups += [(f"char_truncated_{flag}", [bool(r["char_truncated"]) == flag for r in self.data]) for flag in [False, True]]
        for group, mask in groups:
            mask = np.array(mask)
            n = int(mask.sum())
            support = {str(c): int(np.sum(self.y[mask] == c)) for c in [0, 1]}
            for method in ORDER:
                error_counts = [int(np.sum(p[mask] != self.y[mask])) for p in self.models[method]]
                self.groups.append(dict(group=group, method=method, n=n, support=support, error_counts_by_run=error_counts,
                    mean_errors=statistics.mean(error_counts), mean_error_rate=statistics.mean(error_counts) / n if n else None,
                    macro_f1=statistics.mean(float(macro_f1(self.y[mask], p[mask])) for p in self.models[method]) if n else None,
                    warning="small or single-class subgroup: descriptive only" if n < self.config["small_group_warning_n"] or 0 in support.values() else None))
        self.write("length-and-truncation.json", self.groups)
        stability = []
        token_groups = []
        for method in ORDER:
            runs = [r for r in self.index if r["method"] == method]
            if len(self.models[method]) == 3:
                failures = np.sum([p != self.y for p in self.models[method]], axis=0)
                stability.append(dict(method=method, error_count_histogram={str(c): int(np.sum(failures == c)) for c in range(4)}, n=777))
            for run in runs:
                records = self.records[run["run_id"]]
                for flag in [True, False, None]:
                    chosen = [p for p in records if p.get("token_truncated") is flag]
                    support = {str(c): sum(p["true_label"] == c for p in chosen) for c in [0,1]}
                    token_groups.append(dict(method=method, run_id=run["run_id"], token_truncated=flag, n=len(chosen),
                        errors=sum(p["parsed_label"] != p["true_label"] or p["status"] != "ok" for p in chosen),
                        support=support, warning="small or single-class subgroup: descriptive only" if len(chosen)<self.config["small_group_warning_n"] or 0 in support.values() else None,
                        scope="route-specific subgroup; unknown metadata retained"))
        self.write("seed-stability.json", stability)
        self.write("token-truncation.json", token_groups)

    def costs_and_parameters(self):
        self.validation = self.read("results/summaries/validation_searches.json")
        self.training = self.read("results/summaries/training-costs.json")
        audit_path = "reports/qwen17-comparisons-20261026-012117-9e8e51/audit-20261026-142734/summary.json"
        self.lora_audit = self.read(audit_path)
        self.repeat_audit = self.read("reports/qwen17-comparisons-20261026-200544-750ecb/audit-20261026-221123/summary.json")
        self.write("parameters.json", dict(baselines=self.validation, lora=self.lora_audit["results"],
            limitations=["Single seed42 per searched config", "BERT len512 also changes batch", "BERT partial also changes learning rate", "LoRA rank follows alpha=2r"]))
        self.benchmarks = self.read("results/benchmarks/integration.json")
        self.latency = {}
        fixed_ids = self.read("data/manifests/latency_ids.json")
        warmup_ids = self.read("data/manifests/smoke_ids.json")
        self.latency_stats = []
        for benchmark in self.benchmarks:
            run = benchmark["run_id"]
            detail = self.read(f"results/benchmarks/details/{run}.jsonl", lines=True)
            assert [r["sample_id"] for r in detail] == fixed_ids
            assert benchmark["benchmark"]["warmup_ids"] == warmup_ids and len(warmup_ids) == 20
            assert len(detail) == self.config["latency_n"] == 200
            ms = np.array([r["latency_ms"] for r in detail])
            assert np.all(np.isfinite(ms)) and np.all(ms > 0)
            recalculated = dict(mean_ms=float(ms.mean()), p50_ms=float(np.quantile(ms, .5, method="linear")), p95_ms=float(np.quantile(ms, .95, method="linear")), throughput=float(1000 * len(ms) / ms.sum()))
            for k, value in recalculated.items():
                assert abs(value - benchmark["benchmark"][k]) < 1e-8
            row = dict(run_id=run, method=benchmark["method"], seed=benchmark["seed"], n=200, **recalculated,
                       device=benchmark["benchmark"]["device"], resources=benchmark["resources"],
                       scope=benchmark["benchmark"]["timing_scope"])
            self.latency_stats.append(row)
            self.latency[run] = ms
        self.write("latency.json", self.latency_stats)
        self.storage = self.read("results/summaries/deployment-storage.json")
        ledger = []
        for r in self.training:
            if r["method"] in ("bert", "tfidf"):
                ledger.append(dict(identity=r["run_id"], method=r["method"], phase=r["phase"], seed=r["seed"],
                    seconds=r["end_to_end_s"], scope="Historical baseline wrapper: loading/training/validation/saving boundary; details not all separated", device="CPU" if r["method"] == "tfidf" else "GPU"))
        for r in self.lora_audit["results"]:
            ledger.append(dict(identity="lora-" + r["id"] + "-seed42", method="qwen_lora", seed=42, phase="search", seconds=r["perf_counter_seconds"],
                scope="Original perf_counter wrapper; separate from trainer-only interval; B00 clock discrepancy retained", utc_span_seconds=r["utc_span_seconds"], device="GPU"))
        for r in self.repeat_audit["results"]:
            if r["seed"] != 42:
                ledger.append(dict(identity=f'lora-T01-seed{r["seed"]}', method="qwen_lora", seed=r["seed"], phase="seed_repeat", seconds=r["perf_counter_seconds"],
                    scope="Original perf_counter wrapper; excludes already counted seed42 search", device="GPU"))
        assert len({r["identity"] for r in ledger}) == len(ledger)
        self.ledger = ledger
        totals = [dict(method=m, phase=phase, runs=len(group), recorded_seconds=sum(r["seconds"] for r in group),
            task_hours=sum(r["seconds"] for r in group) / 3600, interpretation="recorded successful-task cost only; excludes unknown failed-run and preparation cost")
            for m in ["tfidf", "bert", "qwen_lora"] for phase in ["search", "seed_repeat"]
            if (group := [r for r in ledger if r["method"] == m and r["phase"] == phase])]
        self.write("training-cost-ledger.json", dict(unique_runs=ledger, totals=totals,
            missing=["Not a complete project cost: failure durations and some load/save components absent", "Precision differs across routes; no isolated training-method efficiency claim", "Prompt routes have inference/preparation cost even without task training"],
            historical_failures=self.read("reports/audits/failures.json")))
        self.note("Audited all 11 fixed-200 latency distributions, frozen IDs and 20 warmup IDs; cost ledger deduplicated.")

    def setup_plotting(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        available = {f.name for f in font_manager.fontManager.ttflist}
        font = next((x for x in ["Microsoft YaHei", "Noto Sans CJK SC", "SimHei"] if x in available), "DejaVu Sans")
        plt.rcParams.update({"font.family": font, "axes.unicode_minus": False, "font.size": 10,
            "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white", "savefig.facecolor": "white"})
        self.plt = plt
        self.plot_font = font

    def figure(self, fig, name, title, caption, source, role="正文候选"):
        fig.suptitle(title, fontsize=15, fontweight="bold")
        fig.text(.5, .015, caption, ha="center", va="bottom", fontsize=8, color="#46596b")
        fig.tight_layout(rect=(0, .075, 1, .92))
        for ext in ["png", "pdf"]:
            fig.savefig(self.figures / f"{name}.{ext}", dpi=170)
        self.plt.close(fig)
        self.figure_index.append(dict(id=name, title=title, caption=caption, data=source, role=role, png=f"{name}.png", pdf=f"{name}.pdf"))

    def draw(self):
        self.setup_plotting()
        plt = self.plt
        summary = {r["method"]: r for r in self.summary}
        fig, ax = plt.subplots(figsize=(10, 4.8))
        for i, m in enumerate(ORDER):
            v = summary[m]
            ax.errorbar(i, v["macro_f1"]["mean"] * 100, yerr=(v["macro_f1"]["sample_std"] or 0) * 100, fmt="o", color=COLORS[m], capsize=5, markersize=8)
            ax.annotate(f'{v["macro_f1"]["mean"]*100:.2f}', (i, v["macro_f1"]["mean"]*100), xytext=(0, 13), textcoords="offset points", ha="center")
        if "majority" in summary:
            ax.axhline(summary["majority"]["macro_f1"]["mean"] * 100, linestyle="--", color="#999999", label="多数类参考（不训练）")
            ax.legend(loc="lower right")
        ax.set(xticks=range(5), xticklabels=[NAMES[m] for m in ORDER], ylabel="测试 Macro-F1（%）", ylim=(35, 100))
        self.figure(fig, "01-main-effect", "五种方案：最终测试效果", "测试777条；训练路线n=3，误差条为种子样本标准差；确定性提示n=1，无误差条。", "methods.json")

        fig, ax = plt.subplots(figsize=(10, 4.8))
        markers = ["o", "s", "^"]
        for i, m in enumerate(ORDER):
            values = summary[m]["per_seed"]
            for j, r in enumerate(values):
                x = i + (j - 1) * .12 if len(values) == 3 else i
                ax.scatter(x, r["macro_f1"] * 100, marker=markers[j], color=COLORS[m], s=65)
                ax.annotate(str(r["seed"]) if r["seed"] is not None else "固定推理", (x, r["macro_f1"]*100), xytext=(0, 7 + j * 5), textcoords="offset points", ha="center", fontsize=8)
        ax.set(xticks=range(5), xticklabels=[NAMES[m] for m in ORDER], ylabel="测试 Macro-F1（%）", ylim=(58, 98))
        self.figure(fig, "02-seed-stability", "单次成绩与种子波动", "测试777条；种子42/43/44逐次展示；提示单次确定性结果不作为训练随机重复。", "methods.json")

        fig, axes = plt.subplots(2, 5, figsize=(15, 6.3))
        for i, method in enumerate(ORDER):
            matrix = np.array(self.metrics[self.reps[method]["run_id"]]["confusion_matrix"])
            normalized = matrix / matrix.sum(axis=1, keepdims=True)
            for row, values in enumerate([matrix, normalized]):
                ax = axes[row, i]
                ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
                for y in range(2):
                    for x in range(3):
                        ax.text(x, y, str(values[y, x]) if row == 0 else f"{values[y,x]*100:.1f}%", ha="center", va="center", color="white" if normalized[y, x] > .6 else "#243246", fontsize=9)
                ax.set(xticks=[0,1,2], xticklabels=["负面", "正面", "非法"], yticks=[0,1], yticklabels=["负面", "正面"])
                if row == 0:
                    ax.set_title(NAMES[method])
                if i == 0:
                    ax.set_ylabel("真实类别\n" + ("数量" if row == 0 else "按行比例"))
        self.figure(fig, "03-confusion", "同一测试集的错误方向", "测试777条；训练路线固定seed42示例，未按测试成绩选种子；三种子汇总另见主表。", "methods.json / original predictions")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        lookup = {r["config_id"]: r for r in self.validation if r["method"] == "bert"}
        configs = ["lr_1e5", "main", "lr_5e5"]
        times = {r["run_id"]: r["end_to_end_s"] for r in self.training if "end_to_end_s" in r}
        for ax, key, ylabel in [(axes[0], "macro_f1", "最佳验证 Macro-F1（%，局部刻度）"), (axes[1], "time", "完整配置运行时长（秒，局部刻度）")]:
            values = [lookup[c][key] * 100 if key != "time" else times[f"bert-{c}-seed42"] for c in configs]
            ax.plot(range(3), values, "o-", color=COLORS["bert"])
            ax.set(xticks=range(3), xticklabels=["1e-5", "2e-5", "5e-5"], xlabel="学习率", ylabel=ylabel)
            ax.margins(y=.25)
            for i, v in enumerate(values):
                ax.annotate(f"{v:.2f}", (i, v), xytext=(0, 9), textcoords="offset points", ha="center")
        self.figure(fig, "04-bert-learning-rate", "BERT参数对照：学习率", "验证776条、seed42；长度256、batch16、3轮、全参；每配置按验证F1选最佳轮次。", "parameters.json / training-cost-ledger.json")

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
        lookup = {r["id"]: r for r in self.lora_audit["results"]}
        selected = [lookup[c] for c in ["B00", "R01", "T01"]]
        for ax, key, ylabel, factor in [(axes[0], "macro_f1", "最佳验证 Macro-F1（%）",100), (axes[1], "trainable_parameters", "可训练参数（百万）",1e-6), (axes[2], "perf_counter_seconds", "原运行计时（分钟）",1/60)]:
            values = [r[key] * factor for r in selected]
            ax.bar(range(3), values, color=["#75869c", "#c29648", COLORS["qwen_lora"]])
            ax.set(xticks=range(3), xticklabels=["B00\nq/v r8", "R01\nq/v r16", "T01\nq/k/v/o r8"], ylabel=ylabel)
            if key == "macro_f1":
                ax.set_ylim(0, 100)
            else:
                ax.set_ylim(0, max(values)*1.22)
            ax.bar_label(ax.containers[0], fmt="%.2f", padding=4, fontsize=9)
        self.figure(fig, "05-lora-parameters", "LoRA参数对照：容量、效果与成本", "验证776条、seed42；alpha/r固定为2；B00时钟差异保留，运行计时不等于纯训练内核时间。", "parameters.json")

        fig, axes = plt.subplots(1, 5, figsize=(15, 4.8))
        for ax, method in zip(axes, ORDER):
            run = next(r for r in self.latency_stats if r["method"] == method and r["seed"] in (42, None))
            values = np.sort(self.latency[run["run_id"]])
            ax.step(values, np.arange(1, 201)/200, where="post", color=COLORS[method])
            ax.axvline(run["p50_ms"], color="#667687", linestyle=":", label="P50")
            ax.axvline(run["p95_ms"], color="#be6b40", linestyle="--", label="P95")
            ax.set(title=NAMES[method]+("\nCPU" if method == "tfidf" else "\nGPU"), xlabel="端到端延迟 ms", ylim=(0,1.02))
        axes[0].set_ylabel("累计样本比例")
        axes[-1].legend(loc="lower right", fontsize=8)
        self.figure(fig, "06-latency-ecdf", "固定200条评论：延迟分布与尾延迟", "各图横轴不同；20条预热、batch1、含分词/传输/推理/解析，不含加载/读盘；训练模型展示seed42。", "latency.json / benchmark detail files")

        fig, ax = plt.subplots(figsize=(10, 5))
        for method in ORDER:
            group = [r for r in self.latency_stats if r["method"] == method]
            x = statistics.mean(r["mean_ms"] for r in group)
            y = summary[method]["macro_f1"]["mean"] * 100
            ax.errorbar(x, y, yerr=(summary[method]["macro_f1"]["sample_std"] or 0)*100, fmt="s" if method == "tfidf" else "o", color=COLORS[method], capsize=4, markersize=8)
            ax.annotate(NAMES[method]+("（CPU）" if method == "tfidf" else "（GPU）"), (x, y), xytext=(6, 8), textcoords="offset points", fontsize=9)
        ax.set(xscale="log", xlabel="平均端到端延迟 ms（对数轴，越左越快）", ylabel="测试 Macro-F1（%）", ylim=(58,98))
        ax.margins(x=.4)
        self.figure(fig, "07-effect-latency", "效果与响应时间的取舍", "实际CPU/GPU部署路线；x为各模型均值再平均，y误差为训练种子SD；不代表相同计算资源排名。", "methods.json / latency.json")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        memory, disk = [], []
        for method in ORDER:
            group = [r for r in self.latency_stats if r["method"] == method]
            values = [r["resources"]["inference_allocated_peak"] / 2**20 for r in group if r["resources"]["inference_allocated_peak"] is not None]
            memory.append(max(values) if values else np.nan)
            disk.append(statistics.mean(r["deployment_bytes"] for r in self.storage if r["method"] == method) / 2**20)
        x = np.arange(5)
        axes[0].bar(x[1:], memory[1:], color=[COLORS[m] for m in ORDER[1:]])
        axes[0].text(0, 50, "CPU\n不适用", ha="center")
        axes[0].set_ylabel("CUDA allocated峰值 MiB")
        axes[1].bar(x, disk, color=[COLORS[m] for m in ORDER])
        axes[1].set_ylabel("单模型完整部署 MiB")
        for ax in axes:
            ax.set_xticks(x, [NAMES[m] for m in ORDER], rotation=20)
        self.figure(fig, "08-resources-storage", "显存记录与完整部署体积", "峰值为加载后测试/预热/延迟全阶段；RSS未当峰值。LoRA部署含基座＋适配器；环境另计。", "latency.json / deployment-storage.json")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        length_names = [f"length_{lo}_{hi or 'plus'}" for lo, hi in self.config["length_bins"]]
        labels = ["1–50", "51–128", "129–256", "257–512", ">512"]
        first = [next(r for r in self.groups if r["group"] == g) for g in length_names]
        axes[0].bar(range(5), [r["support"]["0"] for r in first], label="负面", color="#7e8ca4")
        axes[0].bar(range(5), [r["support"]["1"] for r in first], bottom=[r["support"]["0"] for r in first], label="正面", color="#9fc7c1")
        axes[0].legend(); axes[0].set_ylabel("测试评论数")
        for m in ORDER:
            group = [next(r for r in self.groups if r["method"] == m and r["group"] == g) for g in length_names]
            axes[1].plot(range(5), [r["mean_error_rate"]*100 if r["mean_error_rate"] is not None else np.nan for r in group], "o-", color=COLORS[m], label=NAMES[m])
        axes[1].set_ylabel("错误率（%）"); axes[1].legend(fontsize=8)
        for ax in axes:
            ax.set_xticks(range(5), labels, rotation=20); ax.set_xlabel("规范化原文字符数（截断前）")
        self.figure(fig, "09-length-groups", "长度分组：样本构成与错误率", "训练路线为3次错误率均值；含小样本分组，仅描述；>512组经历共同字符截断，不能单凭相关性确认因果。", "length-and-truncation.json")

        fig, ax = plt.subplots(figsize=(11, 5.5))
        for i, row in enumerate(self.pairs):
            low, high = np.array(row["bootstrap_95_percentile"]) * 100
            mid = row["macro_f1_mean_delta"] * 100
            ax.plot([low, high], [i, i], color="#3b687b", linewidth=2)
            ax.scatter(mid, i, color="#a44e45", zorder=3)
        ax.axvline(0, color="#888888", linestyle="--")
        ax.set(yticks=range(len(self.pairs)), yticklabels=[NAMES[r["a"]]+" − "+NAMES[r["b"]] for r in self.pairs], xlabel="Macro-F1差异（百分点）")
        ax.invert_yaxis()
        self.figure(fig, "10-paired-f1-difference", "配对差异：固定模型下的测试样本不确定性", "5000次按真实类别分层的配对bootstrap、95%逐项百分位区间；种子先分别算F1再平均；事后探索性分析。", "paired-comparisons.json")

        fig, ax = plt.subplots(figsize=(10, 5))
        for i, row in enumerate(self.pairs):
            ax.barh(i, row["A_correct_B_wrong"], color="#3b8376", label="A对B错" if i == 0 else None)
            ax.barh(i, -row["A_wrong_B_correct"], color="#bd7363", label="A错B对" if i == 0 else None)
        ax.axvline(0, color="#777777")
        ax.set(yticks=range(len(self.pairs)), yticklabels=[NAMES[r["a"]]+" / "+NAMES[r["b"]] for r in self.pairs], xlabel="评论数（左侧为A退化，右侧为A改善）")
        ax.legend(); ax.invert_yaxis()
        self.figure(fig, "11-corrections-regressions", "同样本对比：改善与退化同时展示", "测试777条；训练方法固定seed42，不是三种子投票或集成；多数类只作分析参考。", "paired-comparisons.json")

        fig, ax = plt.subplots(figsize=(10, 4.8))
        for method in ["tfidf", "bert", "qwen_lora"]:
            if method == "qwen_lora":
                timing = [r["original_timing"]["trainer_interval_seconds"] for r in self.training if r["method"] == method]
            else:
                timing = [r["train_s"] + (r["validation_s"] or 0) for r in self.training if r["run_id"] in summary[method]["runs"]]
            seconds = statistics.mean(timing)
            y = summary[method]["macro_f1"]["mean"] * 100
            ax.scatter(seconds/60, y, s=75, color=COLORS[method])
            ax.annotate(NAMES[method], (seconds/60, y), xytext=(5, 8), textcoords="offset points")
        ax.set(xscale="log", xlabel="已有单次训练区间均值（分钟，对数轴）", ylabel="测试 Macro-F1（%）", ylim=(83,96))
        ax.margins(x=.5)
        self.figure(fig, "12-effect-recorded-training-time", "效果与已记录训练投入（路线间计时边界不同）", "训练边界不同：BERT训练＋验证；TF-IDF拟合；LoRA Trainer区间含验证/保存。仅部署方案观测，不能作纯训练算法效率结论。", "training-cost-ledger.json / training-costs.json")

        fig, ax = plt.subplots(figsize=(9, 4.8))
        for method, seed, path in [("LoRA", 42, "results/qwen17-comparisons/qwen17-comparisons-20261026-012117-9e8e51/T01/validation_history.json"),
            ("LoRA", 43, "results/qwen17-comparisons/qwen17-comparisons-20261026-200544-750ecb/T01-S43/validation_history.json"),
            ("LoRA", 44, "results/qwen17-comparisons/qwen17-comparisons-20261026-200544-750ecb/T01-S44/validation_history.json")]:
            history = self.read(path)
            ax.plot([r["epoch"] for r in history], [r["macro_f1"]*100 for r in history], "o-", label=f"{method} seed{seed}")
        ax.set(xticks=[1,2,3], xlabel="训练轮次", ylabel="验证 Macro-F1（%）")
        ax.legend()
        self.figure(fig, "13-lora-validation", "LoRA T01：三种子的逐轮验证", "验证776条、seed42/43/44；本图为验证F1（局部刻度），不是训练loss。", "LoRA validation_history.json", role="附录候选")
        self.note(f"Created {len(self.figure_index)} figure pairs (PNG/PDF) from real records.")

    def documents(self):
        def table(headers, rows):
            return "\n".join(["|"+"|".join(headers)+"|", "|"+"|".join(["---"]*len(headers))+"|", *["|"+"|".join(str(c) for c in r)+"|" for r in rows]])
        def fmt(value, scale=1):
            return "不适用" if value is None else (f"{value*scale:.3e}" if 0 < abs(value*scale) < .0001 else f"{value*scale:.4f}")
        lines = ["# 当前比较数据表（待学生解释）", "", f"分析批次：{self.id}。只读取历史记录，未运行模型。", "", "## 测试主表", "",
            table(["方法","N","Macro-F1 %","种子样本SD 百分点","Accuracy %","负面F1 %","正面F1 %","非法率 %"],
                [[NAMES[r["method"]],r["n"],fmt(r["macro_f1"]["mean"],100),fmt(r["macro_f1"]["sample_std"],100),fmt(r["accuracy"]["mean"],100),
                  fmt(r["per_class"]["0"]["f1"] if r["per_class"] else None,100),fmt(r["per_class"]["1"]["f1"] if r["per_class"] else None,100),fmt(r["invalid_rate"]["mean"],100)] for r in self.summary]),
            "", "训练路线为各种子指标先算后平均；多数类为无训练参考，未加入成本排名。完整分类别P/R/F1与单种子值见methods.json。", "", "## 配对差异", "",
            table(["A−B","F1差 百分点","95%逐项配对区间","seed42 A对B错","seed42 A错B对","正误McNemar Holm p"],
                [[NAMES[r["a"]]+"−"+NAMES[r["b"]],fmt(r["macro_f1_mean_delta"],100),"["+", ".join(fmt(v,100) for v in r["bootstrap_95_percentile"])+"]",r["A_correct_B_wrong"],r["A_wrong_B_correct"],fmt(r["holm_adjusted_p_seven_pairs"])] for r in self.pairs]),
            "", "区间条件于固定训练模型、固定类别组成，非训练总体区间；McNemar检验正误而非F1。所有分析为事后探索性，不显著不等于等效。", "", "## 推理延迟（逐运行）", "",
            table(["方法","种子","均值ms","P50 ms","P95 ms","条/秒","设备"],[[NAMES[r["method"]],r["seed"],fmt(r["mean_ms"]),fmt(r["p50_ms"]),fmt(r["p95_ms"]),fmt(r["throughput"]),r["device"]] for r in self.latency_stats]),
            "", "每运行200条同ID同顺序、20条预热，batch1；计时含分词/传输/推理/解析，不含加载与读盘；CPU/GPU路线不同。", "", "## 参数对照索引", "",
            "BERT学习率、TF-IDF四配置与LoRA十配置的源数值均保留在parameters.json。均为seed42验证结果；不能将T01三个种子外推为全部配置的稳定性。", "", "## 仍需人工完成", "",
            "- 固定20条试标、编码规则确认、决定正式覆盖量。", "- ≥3类真实错误归因与≥3真实案例；未填归因不生成类型分布。", "- 学生解释、图表选择和结论证据台账。", "- 训练原始逐步loss/时间戳补齐后再重绘；现有归档曲线保留。", "- 新机真实模型复现、可获取外链与最终打包。", ""]
        (self.report / "数据对比与待办.md").write_text("\n".join(lines), encoding="utf-8")
        inventory = []
        for path in sorted((ROOT / "reports/figures").glob("*.png")):
            title = path.name
            if "curves" in title:
                status, reason = "保留归档，待原始日志后统一重绘", "训练原始逐步数据部分在外部目录；不从图片伪造逐点数据"
            elif "validation" in title:
                status, reason = "附录参考", "保留验证选优证据；正文参数图已按新协议重绘"
            else:
                status, reason = "需替换展示", "新版提供种子语义、统一矩阵比例、设备与统计范围等更完整说明"
            inventory.append(dict(file=path.relative_to(ROOT).as_posix(), status=status, reason=reason, action="keep original"))
        self.write("existing-figure-audit.json", inventory)
        audit = ["# 图表审查与使用清单", "", f"已审查{len(inventory)}张历史图；不删除、不覆盖。新增{len(self.figure_index)}组PNG/PDF。", "",
            table(["历史文件","状态","原因"],[[r["file"],r["status"],r["reason"]] for r in inventory]), "", "## 新图", "",
            table(["图号","主题","用途"],[[r["id"],r["title"],r["role"]] for r in self.figure_index]), "", "## 尚未具备的图", "",
            "- 人工错误主因分布：等待试标和正式标注。", "- BERT/LoRA统一逐步loss—时间曲线：部分源数据仅在外部训练目录，当前不捏造。", "- 纯推理阶段RSS峰值、独立加载耗时：历史边界缺失；需要时先确认补测。", ""]
        (self.report / "图表审查清单.md").write_text("\n".join(audit), encoding="utf-8")
        self.write("figure-index.json", self.figure_index)
        cards = []
        for r in self.figure_index:
            cards.append(f'<section><h2>{html.escape(r["id"]+" · "+r["title"])}</h2><img src="../../figures/{html.escape(self.id)}/{r["png"]}" alt="{html.escape(r["title"])}"><p>{html.escape(r["caption"])}</p><p>数据：{html.escape(r["data"])} ｜ <a href="../../figures/{html.escape(self.id)}/{r["pdf"]}">PDF</a></p></section>')
        gallery = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>实验对比图表</title><style>body{font:16px/1.7 system-ui;background:#f3f6fa;color:#263649;max-width:1200px;margin:auto;padding:24px}section{background:white;padding:20px;margin:24px 0;border-radius:10px}img{width:100%;height:auto}h1{font-size:28px}h2{font-size:20px}p{color:#506378}</style><h1>实验对比图表 · '+html.escape(self.id)+'</h1><p>已有记录的事后分析；无新增训练或推理。人工错例归因尚未完成。正文候选需学生确认取舍。</p>'+"".join(cards)+"</html>"
        (self.report / "图表总览.html").write_text(gallery, encoding="utf-8")
        now = datetime.now(timezone(timedelta(hours=8))).isoformat()
        (self.report / "AI使用记录.md").write_text(f"# AI辅助记录\n\n日期：{now}。工具：Codex会话；具体模型版本未提供。\n\n用途：整理已有预测与配置、重算指标、配对区间、延迟审计、图表生成和证据索引。\n\n本人确认：先复用现有结果；补测逐项确认；多数类仅作单独参考行；先人工试标20条。\n\n采用范围：工程代码与派生图表；未自动完成错误主因标注、学生结果解释或最终报告结论。验证：预测ID/真值/数量、既有指标、统计函数边界、固定延迟样本与哈希；实际验证记录见status.json与测试记录。\n", encoding="utf-8")

    def finish(self, error=None):
        for path in [Path(__file__), ROOT / "src/evaluation/metrics.py"]:
            self.sources[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == sha for p, sha in self.sources.items())
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
        outputs = {}
        for directory in [self.output, self.figures, self.report]:
            for path in directory.iterdir():
                if path.is_file() and path.name not in ("status.json", "stdout.log", "stderr.log"):
                    outputs[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.write("status.json", dict(analysis_id=self.id, status="failed" if error else "completed", error=error,
            completed_at=datetime.now(timezone(timedelta(hours=8))).isoformat(), command=sys.argv,
            git_commit=git.stdout.strip() if git.returncode == 0 else None, working_tree_dirty=bool(dirty.stdout.strip()),
            python=sys.version, platform=platform.platform(), numpy=np.__version__, config=self.config, plot_font=getattr(self,"plot_font",None),
            elapsed_seconds=time.perf_counter()-self.started, input_sha256=self.sources, output_sha256=outputs,
            model_loading=False, training=False, inference=False,
            annotation_counts=getattr(self,"annotation_counts",None),
            human_causes_filled=20 if getattr(self,"annotation_counts",None) else 0,
            student_interpretation_approved=False))
        (self.output / "stdout.log").write_text("\n".join(self.log)+"\n", encoding="utf-8")
        (self.output / "stderr.log").write_text(error or "", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/analysis/comparison-20261028-v1.json")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    report = Report(args.config, args.run_id)
    try:
        report.load_predictions()
        report.pair_analysis()
        report.group_analysis()
        report.costs_and_parameters()
        report.draw()
        if report.config.get("formal_version"):
            from scripts.formal_comparison import finalize_formal
            finalize_formal(report)
        else:
            report.documents()
        report.finish()
    except Exception:
        report.finish(traceback.format_exc())
        raise
    print(f"Ready: {report.report}")


if __name__ == "__main__":
    main()
