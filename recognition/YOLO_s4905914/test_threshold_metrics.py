import unittest
from threshold_metrics import box_iou, evaluate


def record(identifier, cls, predictions):
    return dict(image_id=identifier, gt_class=cls, gt_box=[0, 0, 10, 10], predictions=predictions)


def prediction(cls=1, confidence=0.8, box=None):
    return dict(class_id=cls, confidence=confidence, box=box or [0, 0, 10, 10])


class ThresholdTests(unittest.TestCase):
    def test_iou(self):
        self.assertEqual(box_iou([0,0,10,10], [0,0,10,10]), 1)
        self.assertEqual(box_iou([0,0,10,10], [20,20,30,30]), 0)
        self.assertAlmostEqual(box_iou([0,0,10,10], [5,0,15,10]), 1/3)

    def test_perfect_and_false_alert(self):
        rows = [record('a', 1, [prediction()]), record('b', 0, [prediction()])]
        result = evaluate(rows, .5)
        self.assertEqual(result['detection_by_class']['1']['tp'], 1)
        self.assertEqual(result['detection_by_class']['1']['fp'], 1)
        self.assertEqual(result['image_level_malignant_alert']['benign_false_alert_rate'], 1)

    def test_duplicate_boxes_not_double_true_positive(self):
        result = evaluate([record('a', 1, [prediction(), prediction(confidence=.7)])], .5)
        self.assertEqual(result['detection_by_class']['1']['tp'], 1)
        self.assertEqual(result['detection_by_class']['1']['fp'], 1)
        self.assertEqual(result['image_level_malignant_alert']['alert_count'], 1)

    def test_alert_is_not_localisation_recall(self):
        result = evaluate([record('a', 1, [prediction(box=[20,20,30,30])])], .5)
        self.assertEqual(result['detection_by_class']['1']['recall'], 0)
        self.assertEqual(result['image_level_malignant_alert']['sensitivity'], 1)

    def test_wrong_class_not_fixed_by_lower_threshold(self):
        result = evaluate([record('a', 1, [prediction(cls=0)])], .01)
        self.assertEqual(result['image_level_malignant_alert']['fn'], 1)
        self.assertEqual(result['detection_by_class']['1']['fn'], 1)

    def test_empty_predictions_and_threshold_boundary(self):
        result = evaluate([record('a', 1, [prediction(confidence=.5)]), record('b', 0, [])], .5)
        self.assertEqual(result['image_level_malignant_alert']['tp'], 1)
        self.assertEqual(result['image_level_malignant_alert']['tn'], 1)
        high = evaluate([record('a', 1, [prediction(confidence=.5)])], .6)
        self.assertIsNone(high['detection_by_class']['1']['precision'])
        self.assertEqual(high['detection_by_class']['1']['recall'], 0)

    def test_invalid_truth_or_duplicates(self):
        for rows in [[record('a', 2, [])], [record('a', 0, []), record('a', 0, [])]]:
            with self.assertRaises(ValueError):
                evaluate(rows, .5)


if __name__ == '__main__':
    unittest.main()
