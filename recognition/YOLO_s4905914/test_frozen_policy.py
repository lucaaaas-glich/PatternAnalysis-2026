import unittest
from evaluate_test import POLICY


class PolicyTests(unittest.TestCase):
    def test_policy_frozen_before_test(self):
        self.assertEqual(POLICY['primary_model'], 'yolov8n')
        self.assertEqual(POLICY['confidence'], .05)
        self.assertEqual(POLICY['selection_data'], 'validation_661469')
        self.assertTrue(POLICY['selection_confirmed_by_user'])
        self.assertEqual(len(POLICY['validation_summary_sha256']), 64)

    def test_separate_nms_and_matching_thresholds(self):
        self.assertEqual(POLICY['matching_iou'], .5)
        self.assertEqual(POLICY['nms_iou'], .7)
        self.assertFalse(POLICY['half'])


if __name__ == '__main__':
    unittest.main()
