import unittest
from report_metrics import localisation_report


class ReportTests(unittest.TestCase):
    def test_misses_are_zero_not_excluded(self):
        rows = [dict(gt_class=1, gt_box=[0,0,10,10], predictions=[]),
                dict(gt_class=1, gt_box=[0,0,10,10], predictions=[
                    dict(class_id=1, confidence=.8, box=[0,0,10,10])])]
        report = localisation_report(rows, .05)['1']
        self.assertEqual(report['mean_same_class_best_iou_all_truths'], .5)
        self.assertEqual(report['fraction_same_class_iou_ge_0_8'], .5)
        self.assertEqual(report['truth_count'], 2)

    def test_wrong_class_localisation_separated(self):
        rows = [dict(gt_class=1, gt_box=[0,0,10,10], predictions=[
            dict(class_id=0, confidence=.8, box=[0,0,10,10])])]
        report = localisation_report(rows, .05)['1']
        self.assertEqual(report['mean_same_class_best_iou_all_truths'], 0)
        self.assertEqual(report['mean_class_agnostic_best_iou_all_truths'], 1)

    def test_confidence_filter_and_empty_class(self):
        rows = [dict(gt_class=0, gt_box=[0,0,10,10], predictions=[
            dict(class_id=0, confidence=.01, box=[0,0,10,10])])]
        report = localisation_report(rows, .05)
        self.assertEqual(report['0']['mean_same_class_best_iou_all_truths'], 0)
        self.assertIsNone(report['2']['mean_same_class_best_iou_all_truths'])


if __name__ == '__main__':
    unittest.main()
