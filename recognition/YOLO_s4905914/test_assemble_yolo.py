import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from assemble_yolo import assemble, validate_split


class AssemblyTests(unittest.TestCase):
    def fixture(self, root):
        source, splits = root / 'source', root / 'splits'
        source.mkdir()
        splits.mkdir()
        summary = dict(split_unit='image', classes={'Benign': 0, 'Malignant': 1, 'Indeterminate': 2},
                       split_counts=dict(train=1, val=1, test=1), image_count=3)
        (splits / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
        for index, name in enumerate(['train', 'val', 'test']):
            identifier = f'ISIC_{index:07d}'
            image = source / f'{identifier}.jpg'
            image.write_bytes(b'synthetic path fixture; not a decoded photo')
            labels = splits / name / 'labels'
            labels.mkdir(parents=True)
            (labels / f'{identifier}.txt').write_text('0 0.5 0.5 0.2 0.2\n', encoding='utf-8')
            with (splits / f'{name}.csv').open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=['image_id', 'image_path', 'class_id', 'diagnosis'])
                writer.writeheader()
                writer.writerow(dict(image_id=identifier, image_path=str(image.resolve()), class_id=0, diagnosis='Benign'))
        return splits

    def test_validate_and_assemble_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            splits = self.fixture(root)
            self.assertEqual(len(validate_split(splits)[0]), 3)
            # Windows may forbid symlinks; verify calls, not pretend real links exist.
            with patch.object(Path, 'symlink_to') as links:
                report = assemble(splits, root / 'runtime', True)
                self.assertEqual(links.call_count, 3)
            self.assertEqual(report['images_linked'], 3)
            self.assertIn('test: images/test', (root/'runtime'/'data.yaml').read_text())
            self.assertFalse(report['model_trained'])

    def test_missing_image_prevents_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            splits = self.fixture(root)
            (root/'source'/'ISIC_0000000.jpg').unlink()
            with self.assertRaises(ValueError):
                assemble(splits, root/'runtime', True)
            self.assertFalse((root/'runtime').exists())

    def test_label_class_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            splits = self.fixture(Path(directory))
            (splits/'val'/'labels'/'ISIC_0000001.txt').write_text('1 0.5 0.5 0.2 0.2', encoding='utf-8')
            with self.assertRaises(ValueError):
                validate_split(splits)

    def test_acknowledgement_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            splits = self.fixture(root)
            with self.assertRaises(ValueError):
                assemble(splits, root/'runtime')


if __name__ == '__main__':
    unittest.main()
