import unittest
from failure_cases import select_cases


def row(identifier,cls,predictions,box=None):
    return dict(image_id=identifier,gt_class=cls,gt_box=box or [0,0,10,10],predictions=predictions)


def pred(cls,score=.8,box=None):
    return dict(class_id=cls,confidence=score,box=box or [0,0,10,10])


class CaseTests(unittest.TestCase):
    def test_deterministic_unique_categories(self):
        records=[row('a',1,[pred(0)]),row('b',0,[pred(1)]),
                 row('c',1,[pred(1,box=[20,20,30,30])]),
                 row('d',0,[pred(0),pred(0,.7)]),row('e',1,[],[0,0,2,2])]
        cases=select_cases(records)
        self.assertEqual(len(cases),5)
        self.assertEqual(len({c['image_id'] for c in cases}),5)
        self.assertEqual(cases,select_cases(list(reversed(records))))

    def test_success_not_selected(self):
        self.assertEqual(select_cases([row('a',1,[pred(1)])]),[])

    def test_low_score_false_alert_excluded(self):
        self.assertEqual(select_cases([row('a',0,[pred(1,.01)])]),[])

    def test_missing_cause_not_invented(self):
        cases=select_cases([row('a',1,[pred(0)])])
        self.assertEqual(cases[0]['visual_cause'],'pending manual review')


if __name__=='__main__':
    unittest.main()
