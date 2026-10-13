import unittest
from split_images import assign_images


def fixture():
    annotations, metadata = [], []
    for index, category in enumerate(['Benign']*20 + ['Malignant']*10 + ['Indeterminate']):
        identifier = f'ISIC_{index:07d}'
        annotations.append(dict(image_id=identifier, image_path='/example.jpg',
                                center_x='0.5', center_y='0.5', box_width='0.2', box_height='0.2'))
        metadata.append(dict(isic_id=identifier, diagnosis_1=category, lesion_id='same'))
    return annotations, metadata


class SplitTests(unittest.TestCase):
    def test_reproducible_complete_disjoint_and_singleton(self):
        a, m = fixture()
        splits, _, _ = assign_images(a, m)
        self.assertEqual(splits, assign_images(list(reversed(a)), list(reversed(m)))[0])
        flattened = sum(splits.values(), [])
        self.assertEqual(len(flattened), 31)
        self.assertEqual(len(set(flattened)), 31)
        self.assertIn(m[-1]['isic_id'], splits['train'])
        self.assertNotEqual(splits, assign_images(a, m, seed=99)[0])

    def test_reject_duplicate_or_mismatch(self):
        a, m = fixture()
        for candidate in [m[:-1], m+[m[0]]]:
            with self.assertRaises(ValueError):
                assign_images(a, candidate)

    def test_reject_unknown_diagnosis(self):
        a, m = fixture()
        m[0]['diagnosis_1'] = 'unknown'
        with self.assertRaises(ValueError):
            assign_images(a, m)

    def test_reject_invalid_box(self):
        a, m = fixture()
        a[0]['center_x'] = 'nan'
        with self.assertRaises(ValueError):
            assign_images(a, m)


if __name__ == '__main__':
    unittest.main()
