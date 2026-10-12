"""Validate original-mask preparation without a remote job or network download."""

import hashlib
import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from prepare_official_labels import (download_reference, nearest_binary_mask,
                                     preparation_preview_ids, prepare_labels)


class OfficialLabelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.images = self.root / 'images'
        self.images.mkdir()
        self.reference = self.root / 'reference.zip'

    def make_fixture(self, bad_second_geometry=False):
        with zipfile.ZipFile(self.reference, 'w') as archive:
            for identifier in ('ISIC_a', 'ISIC_b'):
                size = (5, 4) if identifier == 'ISIC_b' and bad_second_geometry else (4, 4)
                Image.new('RGB', size).save(self.images / (identifier + '.jpg'))
                values = np.zeros((8, 8), dtype=np.uint8)
                values[2:6, 2:6] = 255
                buffer = io.BytesIO()
                Image.fromarray(values).save(buffer, format='PNG')
                archive.writestr('official/' + identifier + '_segmentation.png', buffer.getvalue())
        return hashlib.sha256(self.reference.read_bytes()).hexdigest()

    def test_nearest_preserves_encoding_and_correct_box(self):
        values = np.zeros((8, 8), dtype=np.uint8)
        values[2:6, 2:6] = 255
        resized, box = nearest_binary_mask(Image.fromarray(values), (4, 4))
        self.assertEqual(set(np.unique(resized).tolist()), {0, 255})
        self.assertEqual(box, (1, 1, 3, 3))
        rounded = Image.fromarray(values[:7])
        self.assertEqual(nearest_binary_mask(rounded, (4, 4))[0].size, (4, 4))

    def test_bad_geometry_nonbinary_and_lost_lesion_are_rejected(self):
        good = Image.new('L', (8, 8), 255)
        with self.assertRaises(ValueError):
            nearest_binary_mask(good, (5, 4))
        for image in (Image.new('L', (8, 8), 127), Image.new('L', (8, 8), 0),
                      Image.new('RGB', (8, 8), 'white')):
            with self.assertRaises(ValueError):
                nearest_binary_mask(image, (4, 4))
        tiny = np.zeros((8, 8), dtype=np.uint8)
        tiny[0, 0] = 255
        with self.assertRaises(ValueError):
            nearest_binary_mask(Image.fromarray(tiny), (4, 4))

    def test_full_export_preserves_sources_and_does_not_claim_split(self):
        checksum = self.make_fixture()
        before = self.reference.read_bytes()
        before_image = (self.images / 'ISIC_a.jpg').read_bytes()
        result = prepare_labels(self.images, self.reference, self.root / 'prepared',
                                expected_pairs=2, expected_hash=checksum, samples=2)
        self.assertTrue(result['all_pairs_validated'])
        self.assertEqual(result['labels_exported'], 2)
        self.assertEqual(result['preview_ids'], ['ISIC_a', 'ISIC_b'])
        label = (self.root / 'prepared' / 'labels' / 'ISIC_a.txt').read_text().split()
        self.assertEqual(label[0], '0')
        np.testing.assert_allclose([float(v) for v in label[1:]], [0.5, 0.5, 0.5, 0.5])
        self.assertFalse(result['split_created'])
        self.assertFalse(result['patient_independence_verified'])
        self.assertFalse(result['diagnosis_mapping_verified'])
        self.assertFalse(result['school_preprocessing_provenance_verified'])
        self.assertEqual(self.reference.read_bytes(), before)
        self.assertEqual((self.images / 'ISIC_a.jpg').read_bytes(), before_image)
        self.assertTrue((self.root / 'prepared' / 'previews' / 'ISIC_a.png').is_file())

    def test_one_bad_pair_prevents_all_label_exports(self):
        checksum = self.make_fixture(bad_second_geometry=True)
        result = prepare_labels(self.images, self.reference, self.root / 'failed',
                                expected_pairs=2, expected_hash=checksum)
        self.assertEqual(result['valid_pairs'], 1)
        self.assertEqual(result['issues_count'], 1)
        self.assertEqual(result['labels_exported'], 0)
        self.assertFalse((self.root / 'failed' / 'labels').exists())
        self.assertTrue((self.root / 'failed' / 'summary.json').is_file())

    def test_no_overwrite_source_output_and_checksum_guards(self):
        checksum = self.make_fixture()
        for destination, expected in ((self.images / 'out', checksum),
                                      (self.root / 'out', 'wrong-checksum')):
            with self.assertRaises(ValueError):
                prepare_labels(self.images, self.reference, destination, 2, expected)
            self.assertFalse(destination.exists())
        output = self.root / 'existing'
        output.mkdir()
        with self.assertRaises(ValueError):
            prepare_labels(self.images, self.reference, output, 2, checksum)

    def test_missing_duplicate_or_extra_ids_are_rejected_before_writing(self):
        checksum = self.make_fixture()
        nested = self.images / 'nested'
        nested.mkdir()
        Image.new('RGB', (4, 4)).save(nested / 'ISIC_a.jpg')
        with self.assertRaises(ValueError):
            prepare_labels(self.images, self.reference, self.root / 'duplicate', 2, checksum)
        other_images = self.root / 'other_images'
        other_images.mkdir()
        Image.new('RGB', (4, 4)).save(other_images / 'ISIC_missing.jpg')
        with self.assertRaises(ValueError):
            prepare_labels(other_images, self.reference, self.root / 'missing', 1, checksum)

    def test_verified_download_cache_and_bad_download(self):
        checksum = self.make_fixture()
        payload = self.reference.read_bytes()
        destination = self.root / 'cache' / 'reference.zip'
        with patch('prepare_official_labels.urllib.request.urlopen', return_value=io.BytesIO(payload)):
            self.assertEqual(download_reference(destination, checksum), destination)
        self.assertEqual(destination.read_bytes(), payload)
        with patch('prepare_official_labels.urllib.request.urlopen') as no_network:
            download_reference(destination, checksum)
            no_network.assert_not_called()
        with self.assertRaises(ValueError):
            download_reference(destination, 'wrong-checksum')
        rejected = self.root / 'rejected' / 'reference.zip'
        with patch('prepare_official_labels.urllib.request.urlopen', return_value=io.BytesIO(b'bad download')):
            with self.assertRaises(ValueError):
                download_reference(rejected, checksum)
        self.assertFalse(rejected.exists())
        self.assertEqual(list(rejected.parent.iterdir()), [])

    def test_preview_selection_has_area_extremes_and_reproducible_randomness(self):
        records = [{'image_id': f'ISIC_{i}', 'box_area_fraction': i / 10} for i in range(7)]
        selected = preparation_preview_ids(records, 6)
        self.assertEqual(selected[:2], ['ISIC_0', 'ISIC_6'])
        self.assertEqual(len(selected), len(set(selected)))
        self.assertEqual(selected, preparation_preview_ids(records, 6))


if __name__ == '__main__':
    unittest.main()
