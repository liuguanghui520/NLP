from copy import deepcopy
import json
from pathlib import Path
import unittest

from scripts.case_analysis import DEFAULT_CASE_NOTES, enrich_case, render_casebook, render_report_selection


ROOT = Path(__file__).resolve().parents[1]


class CaseAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.originals = json.loads((ROOT / "results/analysis/comparison-formal-20261028-03/casebook.json").read_text(encoding="utf-8"))
        cls.notes = json.loads((ROOT / DEFAULT_CASE_NOTES).read_text(encoding="utf-8"))

    def test_revision_preserves_frozen_labels_predictions_and_attributions(self):
        before = deepcopy(self.originals)
        for original in self.originals:
            revised = enrich_case(original, self.notes)
            for key in ["sample", "all_runs", "error_attributions", "selection"]:
                self.assertEqual(revised[key], original[key])
            self.assertFalse(revised["analysis"]["recommendations_executed"])
            self.assertEqual(revised["analysis"]["student_interpretation_approved"], bool(self.notes["cases"][original["sample"]["sample_id"]].get("approval", {}).get("approved")))
        self.assertEqual(before, self.originals)

    def test_nine_cases_do_not_regress_to_one_recommendation_template(self):
        cases = [enrich_case(c, self.notes) for c in self.originals]
        self.assertEqual(len(cases), 9)
        self.assertEqual(len({c["improvement"] for c in cases}), 9)
        text = render_casebook(cases)
        self.assertEqual(text.count("**本例的局限与不能推出的结论**"), 9)
        self.assertEqual(text.count("**有针对性的改进与验收方式**"), 9)

    def test_invented_evidence_is_rejected(self):
        notes = deepcopy(self.notes)
        notes["cases"][self.originals[0]["sample"]["sample_id"]]["evidence_quotes"] = ["NONEXISTENT_EVIDENCE_7b2e"]
        with self.assertRaises(ValueError):
            enrich_case(self.originals[0], notes)

    def test_unreviewed_case_has_no_generic_fallback(self):
        case = deepcopy(self.originals[0])
        case["sample"]["sample_id"] = "unknown-case"
        with self.assertRaises(ValueError):
            enrich_case(case, self.notes)

    def test_wrong_run_evidence_cannot_be_dropped(self):
        case = deepcopy(self.originals[0])
        case["error_attributions"] = []
        with self.assertRaises(ValueError):
            enrich_case(case, self.notes)

    def test_label_disagreement_is_not_resolved_by_interpretation(self):
        original = next(c for c in self.originals if c["sample"]["queue_index"] == 8)
        revised = enrich_case(original, self.notes)
        self.assertEqual(revised["sample"]["source_label"], 1)
        self.assertEqual(revised["sample"]["assessed_label"], 0)
        self.assertEqual(revised["sample"]["review_status"], "label_disagreement_pending")
        self.assertEqual({e["error_type_v1"] for e in revised["error_attributions"]}, {"T5", "T6"})

    def test_confirmed_selection_merges_cases_without_claiming_experimental_gain(self):
        groups = self.notes['report_selection']
        self.assertEqual([g['case_numbers'] for g in groups], [[1], [2,8], [3,5], [4], [6], [7], [9]])
        self.assertEqual(sum(g['placement']=='正文' for g in groups), 5)
        cases = [enrich_case(c, self.notes) for c in self.originals]
        self.assertTrue(all(c['analysis']['student_interpretation_approved'] for c in cases))
        self.assertTrue(all(not c['analysis']['recommendations_executed'] for c in cases))
        self.assertIn('124条待复核', render_report_selection(cases, self.notes))


if __name__ == "__main__":
    unittest.main()
