"""Audit must report bad data without modifying the source or declaring a split."""

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from audit_data import audit_pairs, save_report


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.images, self.masks = self.root / 'images', self.root / 'masks'
        self.images.mkdir()
        self.masks.mkdir()

    def pair(self, name='ISIC_example', mask_size=(10, 8), value=255):
        Image.new('RGB', (10, 8)).save(self.images / f'{name}.jpg')
        Image.new('L', mask_size, value).save(self.masks / f'{name}_segmentation.png')

    def test_good_pairs_and_metadata_do_not_prove_patient_independence(self):
        self.pair()
        metadata = self.root / 'metadata'
        metadata.mkdir()
        (metadata / 'candidate.csv').write_text('image_id,patient_id,diagnosis\n', encoding='utf-8')
        report = audit_pairs(self.images, self.masks, metadata)
        self.assertTrue(report['pair_checks_passed'])
        self.assertEqual(report['valid_pairs'], 1)
        self.assertEqual(report['paired_image_dimensions'], {'10x8': 1})
        self.assertEqual(report['metadata_csv_candidates'][0]['header'],
                         ['image_id', 'patient_id', 'diagnosis'])
        self.assertFalse(report['patient_independence_verified'])
        self.assertFalse(report['diagnosis_mapping_verified'])
        self.assertFalse(report['split_created'])

    def test_missing_orphan_and_duplicate_ids_are_reported(self):
        self.pair()
        Image.new('RGB', (10, 8)).save(self.images / 'missing.jpg')
        Image.new('L', (10, 8), 255).save(self.masks / 'orphan_segmentation.png')
        duplicate = self.images / 'subfolder'
        duplicate.mkdir()
        Image.new('RGB', (10, 8)).save(duplicate / 'ISIC_example.jpg')
        report = audit_pairs(self.images, self.masks)
        self.assertFalse(report['pair_checks_passed'])
        self.assertEqual(report['missing_mask_ids'], ['missing'])
        self.assertEqual(report['orphan_mask_ids'], ['orphan'])
        self.assertEqual(report['duplicate_image_ids'], ['ISIC_example'])
        self.assertEqual(report['unique_pairs_checked'], 0)

    def test_bad_masks_and_corrupt_files_are_reported(self):
        self.pair('empty', value=0)
        self.pair('nonbinary', value=127)
        self.pair('mismatch', mask_size=(5, 4))
        self.pair('corrupt')
        (self.images / 'corrupt.jpg').write_bytes(b'not an image')
        report = audit_pairs(self.images, self.masks)
        self.assertEqual(len(report['pair_issues']), 4)
        self.assertEqual(report['valid_pairs'], 0)
        self.assertFalse(report['pair_checks_passed'])

    def test_empty_input_cannot_pass(self):
        self.assertFalse(audit_pairs(self.images, self.masks)['pair_checks_passed'])

    def test_report_cannot_modify_input_or_overwrite_previous_evidence(self):
        self.pair()
        report = audit_pairs(self.images, self.masks)
        with self.assertRaises(ValueError):
            save_report(report, self.images / 'report.json', [self.images, self.masks])
        destination = self.root / 'outputs' / 'report.json'
        save_report(report, destination, [self.images, self.masks])
        with self.assertRaises(ValueError):
            save_report(report, destination, [self.images, self.masks])


if __name__ == '__main__':
    unittest.main()
