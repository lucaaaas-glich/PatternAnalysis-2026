"""Boundary checks that catch annotation errors before GPU training."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from dataset import bbox_to_yolo, inspect_pairs, mask_to_bbox


class AnnotationTests(unittest.TestCase):
    def test_known_rectangle(self):
        mask = np.zeros((80, 100), dtype=np.uint8)
        mask[20:36, 10:30] = 255
        self.assertEqual(mask_to_bbox(mask), (10, 20, 30, 36))
        np.testing.assert_allclose(bbox_to_yolo(mask_to_bbox(mask), 100, 80),
                                   (0.2, 0.35, 0.2, 0.2))

    def test_single_pixel_at_bottom_right_has_positive_area(self):
        mask = np.zeros((3, 4), dtype=np.uint8)
        mask[2, 3] = 1
        self.assertEqual(mask_to_bbox(mask), (3, 2, 4, 3))
        np.testing.assert_allclose(bbox_to_yolo(mask_to_bbox(mask), 4, 3),
                                   (0.875, 5 / 6, 0.25, 1 / 3))

    def test_full_image_and_disconnected_regions(self):
        self.assertEqual(bbox_to_yolo(mask_to_bbox(np.ones((4, 6))), 6, 4),
                         (0.5, 0.5, 1.0, 1.0))
        mask = np.zeros((4, 6), dtype=np.uint8)
        mask[0, 1] = mask[3, 5] = 255
        self.assertEqual(mask_to_bbox(mask), (1, 0, 6, 4))

    def test_bad_annotation_is_rejected(self):
        for mask in (np.zeros((2, 2)), np.array([[127]]), np.zeros((2, 2, 3))):
            with self.assertRaises(ValueError):
                mask_to_bbox(mask)
        with self.assertRaises(ValueError):
            bbox_to_yolo((0, 0, 11, 8), 10, 8)

    def test_pairing_dimensions_and_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, masks, output = root / 'images', root / 'masks', root / 'out'
            images.mkdir()
            masks.mkdir()
            Image.new('RGB', (100, 80)).save(images / 'ISIC_example.jpg')
            with self.assertRaises(ValueError):
                inspect_pairs(images, masks, output)
            Image.new('L', (50, 40), 255).save(masks / 'ISIC_example_segmentation.png')
            with self.assertRaises(ValueError):
                inspect_pairs(images, masks, output)
            self.assertFalse(output.exists())
            mask = np.zeros((80, 100), dtype=np.uint8)
            mask[20:36, 10:30] = 255
            Image.fromarray(mask).save(masks / 'ISIC_example_segmentation.png')
            result = inspect_pairs(images, masks, output, samples=1)
            self.assertEqual(result['valid_pairs'], 1)
            label = (output / 'labels' / 'ISIC_example.txt').read_text().split()
            self.assertEqual(label[0], '0')
            np.testing.assert_allclose([float(v) for v in label[1:]], (0.2, 0.35, 0.2, 0.2))
            self.assertTrue((output / 'previews' / 'ISIC_example.png').is_file())


if __name__ == '__main__':
    unittest.main()
