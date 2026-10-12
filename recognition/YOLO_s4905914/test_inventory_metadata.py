"""Synthetic metadata-discovery tests; no real patient records used."""

import tempfile
import unittest
from pathlib import Path

from inventory_metadata import inventory


class InventoryTests(unittest.TestCase):
    def test_headers_only_and_candidate_types(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'metadata.csv').write_text('\ufeffimage_id,patient_id,diagnosis\nSECRET_ROW\n', encoding='utf-8')
            (root / 'labels.tsv').write_text('image_id\tlesion_id\nSECRET_ROW\n', encoding='utf-8')
            (root / 'records.json').write_text('SECRET_ROW', encoding='utf-8')
            (root / 'photo.jpg').write_bytes(b'not an image')
            report = inventory(root)
            self.assertEqual(len(report['candidates']), 3)
            self.assertNotIn('SECRET_ROW', str(report))
            self.assertEqual(report['candidates'][1]['header'], ['image_id', 'patient_id', 'diagnosis'])
            self.assertFalse(report['patient_independence_verified'])

    def test_depth_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'nested').mkdir()
            (root / 'nested' / 'data.csv').write_text('image_id\n', encoding='utf-8')
            report = inventory(root, max_depth=0)
            self.assertEqual(report['top_level_directories'], ['nested'])
            self.assertEqual(report['candidates'], [])

    def test_entry_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ['a.csv', 'b.csv']:
                (root / name).write_text('id\n', encoding='utf-8')
            report = inventory(root, max_entries=1)
            self.assertTrue(report['truncated_by_entry_limit'])
            self.assertEqual(report['entries_examined'], 1)

    def test_long_header_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'bad.csv').write_text('x' * 8193, encoding='utf-8')
            self.assertIn('error', inventory(root)['candidates'][0])

    def test_invalid_root_or_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for kwargs in [{'max_depth': -1}, {'max_entries': 0}]:
                with self.assertRaises(ValueError):
                    inventory(root, **kwargs)
            with self.assertRaises(ValueError):
                inventory(root / 'missing')


if __name__ == '__main__':
    unittest.main()
