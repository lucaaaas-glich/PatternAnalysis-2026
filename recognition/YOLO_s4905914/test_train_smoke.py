import tempfile
import unittest
from pathlib import Path

from modules import load_detector
from train import smoke_config, training_config, verify_initial_weights


class SmokeTests(unittest.TestCase):
    def test_comparison_matches_baseline_budget(self):
        baseline = training_config('data.yaml', 'output', 'baseline')
        comparison = training_config('data.yaml', 'output', 'comparison')
        self.assertEqual(baseline, comparison)

    def test_comparison_rejects_unverified_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / 'wrong.pt'
            weights.write_bytes(b'not YOLOv8s')
            with self.assertRaises(ValueError):
                verify_initial_weights(weights, 'comparison')

    def test_baseline_profile_preserves_validation_boundary(self):
        options = training_config('data.yaml', 'output', 'baseline')
        self.assertEqual(options['epochs'], 30)
        self.assertEqual(options['imgsz'], 640)
        self.assertEqual(options['batch'], 16)
        self.assertEqual(options['workers'], 1)
        self.assertEqual(options['split'], 'val')
        self.assertFalse(options['resume'])
        self.assertFalse(options['amp'])
        self.assertEqual(options['seed'], 3710)
        self.assertEqual(options['close_mosaic'], 10)

    def test_unknown_profile_rejected(self):
        with self.assertRaises(ValueError):
            training_config('data.yaml', 'output', 'test')

    def test_baseline_rejects_unverified_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / 'wrong.pt'
            weights.write_bytes(b'not the original model')
            with self.assertRaises(ValueError):
                verify_initial_weights(weights, 'baseline')
            self.assertEqual(len(verify_initial_weights(weights, 'smoke')), 64)

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
