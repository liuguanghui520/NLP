"""Boundary tests for CSV parsing, provenance, cleaning and deduplication."""
import csv
from pathlib import Path
import tempfile
import unittest

from process_data import prepare, read_source


class DataRulesTest(unittest.TestCase):
    def test_csv_records_cleaning_duplicates_and_conflicts(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'source.csv'
            texts = [('1', '  带逗号,和"引号"\r\n下一行  '),
                     ('1', '带逗号,和"引号"\n下一行'), ('0', ' \r\n '),
                     ('0', '冲突'), ('1', '冲突'), ('0', '😀' * 513)]
            with source.open('w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['label', 'review'])
                writer.writerows(texts)
            before = source.read_bytes()
            rows, issues, sha = read_source(source)
            valid, excluded, conflicts = prepare(rows, 512)
            self.assertFalse(issues)
            self.assertEqual([r['source_row'] for r in rows], list(range(1, 7)))
            self.assertEqual(rows[0]['sample_id'], sha + ':1')
            self.assertEqual(valid[0]['normalized_text'], '带逗号,和"引号"\n下一行')
            self.assertEqual(valid[0]['raw_text'], texts[0][1])
            self.assertEqual({r['reason'] for r in excluded}, {'empty_text', 'duplicate_same_label'})
            self.assertEqual([r['source_row'] for r in conflicts], [4, 5])
            self.assertEqual(valid[1]['text'], '😀' * 512)
            self.assertEqual(source.read_bytes(), before)

    def test_missing_body_is_blocking_not_empty(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'broken.csv'
            source.write_text('label,review\n1\n2,正文\n', encoding='utf-8')
            rows, issues, _ = read_source(source)
            self.assertEqual(len(issues), 2)
            self.assertEqual(rows, [])


if __name__ == '__main__':
    unittest.main()
