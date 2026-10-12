"""Reference comparison must never be confused with detector evaluation."""

import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from check_mask_reference import box_iou, compare_reference


class ReferenceTests(unittest.TestCase):
    def test_box_iou_and_invalid_cases(self):
        self.assertEqual(box_iou([0, 0, 2, 2], [0, 0, 2, 2]), 1)
        self.assertEqual(box_iou([0, 0, 2, 2], [2, 0, 4, 2]), 0)
        self.assertAlmostEqual(box_iou([0, 0, 2, 2], [1, 0, 3, 2]), 1 / 3)
        self.assertIsNone(box_iou([0, 0, 2, 2], None))
        with self.assertRaises(ValueError):
            box_iou([0, 0, 0, 2], [0, 0, 2, 2])

    def test_nearest_reference_and_no_automatic_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mask = np.zeros((8, 8), dtype=np.uint8)
            mask[2:6, 2:6] = 255
            png = io.BytesIO()
            Image.fromarray(mask).save(png, format='PNG')
            archive_path = root / 'reference.zip'
            with zipfile.ZipFile(archive_path, 'w') as archive:
                archive.writestr('official/ISIC_example_segmentation.png', png.getvalue())
            sample_path = root / 'samples.json'
            sample_path.write_text(json.dumps({'records': [{
                'image_id': 'ISIC_example', 'width': 4, 'height': 4,
                'candidate_threshold': 128, 'boxes_by_threshold': {'128': [1, 1, 3, 3]},
            }]}), encoding='utf-8')
            result = compare_reference(sample_path, archive_path)
            self.assertEqual(result['records'][0]['official_nearest_box'], [1, 1, 3, 3])
            self.assertEqual(result['records'][0]['bbox_iou_annotation_consistency_only'], 1)
            self.assertFalse(result['threshold_approved'])
            self.assertFalse(result['school_preprocessing_provenance_verified'])
            self.assertFalse(result['representative_full_dataset_check'])


if __name__ == '__main__':
    unittest.main()
