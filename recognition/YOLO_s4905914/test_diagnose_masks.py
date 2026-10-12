"""Candidate thresholding must not replace source data or the strict converter."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from dataset import mask_to_bbox
from diagnose_masks import diagnose_samples, threshold_diagnostics


class MaskDiagnosisTests(unittest.TestCase):
    def test_threshold_edges_and_nonzero_halo(self):
        values = np.array([[1, 0, 0], [0, 127, 128], [0, 255, 0]], dtype=np.uint8)
        result = threshold_diagnostics(values)
        self.assertEqual(result['boxes_by_threshold']['1'], [0, 0, 3, 3])
        self.assertEqual(result['boxes_by_threshold']['128'], [1, 1, 3, 3])
        self.assertEqual(result['boxes_by_threshold']['255'], [1, 2, 2, 3])
        with self.assertRaises(ValueError):
            mask_to_bbox(values)  # Existing strict production converter is unchanged.

    def test_empty_candidate_is_visible_in_report(self):
        result = threshold_diagnostics(np.array([[0, 100]], dtype=np.uint8))
        self.assertFalse(result['candidate_has_foreground'])
        self.assertIsNone(result['boxes_by_threshold']['128'])

    def test_unsupported_encoding_and_threshold_are_rejected(self):
        for values in (np.zeros((2, 2, 3), dtype=np.uint8),
                       np.array([[0, 65535]], dtype=np.uint16),
                       np.array([[0, 1]], dtype=np.uint8)):
            with self.assertRaises(ValueError):
                threshold_diagnostics(values)
        for threshold in (0, 256, 128.5, True):
            with self.assertRaises(ValueError):
                threshold_diagnostics(np.array([[0, 255]], dtype=np.uint8), threshold)

    def test_preview_is_provisional_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, masks = root / 'images', root / 'masks'
            images.mkdir()
            masks.mkdir()
            Image.new('RGB', (3, 3)).save(images / 'ISIC_example.jpg')
            source = masks / 'ISIC_example_segmentation.png'
            Image.fromarray(np.array([[0, 1, 0], [127, 128, 0], [0, 255, 0]], dtype=np.uint8)).save(source)
            before = source.read_bytes()
            result = diagnose_samples(images, masks, root / 'diagnosis')
            self.assertEqual(source.read_bytes(), before)
            self.assertFalse(result['threshold_approved'])
            self.assertFalse(result['training_labels_exported'])
            self.assertTrue((root / 'diagnosis' / 'overview.png').is_file())
            with self.assertRaises(ValueError):
                diagnose_samples(images, masks, root / 'diagnosis')
            with self.assertRaises(ValueError):
                diagnose_samples(images, masks, masks / 'diagnosis')


if __name__ == '__main__':
    unittest.main()
