"""Acceptance boundaries for the portable, model-free reproduction entry point."""

from pathlib import Path
import tempfile
import unittest

from src.common.io import digest, load, rows, save, save_rows
from src.reproduction import audit, main, verify_data


class ReproductionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.data = self.root / "data/frozen"
        self.data.mkdir(parents=True)
        ids = {}
        for split in ["train", "validation", "test"]:
            records = [dict(sample_id=f"{split}-{c}", split=split, label=c,
                            text=f"{split} example {c}") for c in [0, 1]]
            save_rows(self.data / f"{split}.jsonl", records)
            ids[split] = [row["sample_id"] for row in records]
        save(self.data / "split_ids.json", ids)
        save(self.data / "manifest.json", {"files": {
            p.name: digest(p) for p in self.data.iterdir() if p.is_file()
        }})
        methods = ["tfidf", "bert", "qwen_zero_shot", "qwen_few_shot", "qwen_lora"]
        trained = ["tfidf", "bert", "qwen_lora"]
        index = []
        truth = rows(self.data / "test.jsonl")
        for method in methods:
            for seed in [42, 43, 44] if method in trained else [None]:
                run_id = f"{method}-{seed}"
                predictions = [dict(run_id=run_id, method=method, config_id="fixed", seed=seed,
                                    sample_id=row["sample_id"], split="test", true_label=row["label"],
                                    raw_output=str(row["label"]), parsed_label=row["label"], status="ok",
                                    model_id="frozen-model", checkpoint_id=None,
                                    source_artifact_sha256="a" * 64) for row in truth]
                save_rows(self.root / "results/predictions" / f"{run_id}.jsonl", predictions)
                index.append(dict(run_id=run_id, method=method, seed=seed, split="test",
                                  metrics=dict(accuracy=1.0, macro_f1=1.0, weighted_f1=1.0, invalid_rate=0.0)))
        save_rows(self.root / "results/run_index.jsonl", index)
        self.config = dict(data_directory="data/frozen", data_manifest="data/frozen/manifest.json",
                           split_counts={s: 2 for s in ids}, max_characters=512,
                           run_index="results/run_index.jsonl", run_ids=[r["run_id"] for r in index],
                           method_order=methods, trained_methods=trained, training_seeds=[42, 43, 44],
                           metric_tolerance=1e-12, seed_registry={"training": [42, 43, 44]},
                           expected_test_means={m: dict(accuracy=1.0, macro_f1=1.0, weighted_f1=1.0,
                                                       invalid_rate=0.0) for m in methods})
        self.config["input_sha256"] = {
            p.relative_to(self.root).as_posix(): digest(p)
            for p in (self.root / "results").rglob("*") if p.is_file()
        }
        self.save_config()

    def save_config(self):
        save(self.root / "configs/reproduction.json", self.config)

    def test_saved_results_need_no_personal_model_paths(self):
        self.assertFalse((self.root / "configs/paths.local.json").exists())
        result = audit(self.root, self.config)
        self.assertEqual(len(result["runs"]), 11)
        self.assertEqual(result["max_metric_difference"], 0)

    def test_changed_prediction_is_rejected_before_scoring(self):
        path = next((self.root / "results/predictions").glob("*.jsonl"))
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            audit(self.root, self.config)

    def test_missing_training_seed_is_not_averaged_away(self):
        path = self.root / "results/run_index.jsonl"
        index = [r for r in rows(path) if r["run_id"] != "bert-43"]
        save_rows(path, index)
        self.config["run_ids"].remove("bert-43")
        self.config["input_sha256"]["results/run_index.jsonl"] = digest(path)
        with self.assertRaisesRegex(ValueError, "seed coverage"):
            audit(self.root, self.config)

    def test_cross_split_overlap_is_rejected_even_with_updated_hashes(self):
        path = self.data / "validation.jsonl"
        records = rows(path)
        records[0]["sample_id"] = "train-0"
        save_rows(path, records)
        ids = load(self.data / "split_ids.json")
        ids["validation"][0] = "train-0"
        save(self.data / "split_ids.json", ids)
        manifest = load(self.data / "manifest.json")
        for name in ["validation.jsonl", "split_ids.json"]:
            manifest["files"][name] = digest(self.data / name)
        save(self.data / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "overlapping"):
            verify_data(self.root, self.config)

    def test_existing_output_is_not_overwritten(self):
        output = self.root / "results/reproduction/existing"
        output.mkdir(parents=True)
        (output / "sentinel.txt").write_text("keep", encoding="utf-8")
        with self.assertRaises(SystemExit) as error:
            main(["--output", "results/reproduction/existing"], root=self.root)
        self.assertEqual(error.exception.code, 2)
        self.assertEqual((output / "sentinel.txt").read_text(encoding="utf-8"), "keep")

    def test_failure_has_a_traceable_status_without_rewriting_inputs(self):
        path = next((self.root / "results/predictions").glob("*.jsonl"))
        path.write_text("broken input", encoding="utf-8")
        result = main(["--output", "results/reproduction/failed"], root=self.root)
        self.assertEqual(result, 1)
        status = load(self.root / "results/reproduction/failed/status.json")
        self.assertEqual(status["status"], "failed")
        self.assertFalse(status["inference"])
        self.assertEqual(path.read_text(encoding="utf-8"), "broken input")


if __name__ == "__main__":
    unittest.main()
