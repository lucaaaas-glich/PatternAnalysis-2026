import tempfile
import unittest
from pathlib import Path

from modules import load_detector
from train import smoke_config


class SmokeTests(unittest.TestCase):
    def test_config_is_small_validation_only(self):
        options = smoke_config('data.yaml', 'output')
        self.assertEqual(options['epochs'], 2)
        self.assertEqual(options['batch'], 8)
        self.assertEqual(options['imgsz'], 320)
        self.assertFalse(options['amp'])
        self.assertFalse(options['exist_ok'])
        self.assertTrue(options['val'])
        self.assertNotIn('test', options)

    def test_missing_weights_do_not_download(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                load_detector(Path(directory) / 'missing.pt')


if __name__ == '__main__':
    unittest.main()
