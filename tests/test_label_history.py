import unittest
import numpy as np
import pandas as pd
from spend_course10.label_history import label_history_features,restrict_to_target_cohort


class LabelHistoryTests(unittest.TestCase):
    def objects(self):
        d=pd.to_datetime(['2018-06-01','2018-07-01','2018-08-01','2018-09-01'])
        return pd.DataFrame({'Id':list('abcd'),'customer_id':'a','as_of_date':d,'target_end':d+pd.Timedelta(days=30),
           'official':[True,True,True,False],'is_test':[False,False,False,True],'spend_30d':[0.,100.,20.,np.nan]})

    def test_current_and_future_labels_are_never_features(self):
        a=self.objects();b=a.copy();b.loc[2,'spend_30d']=1e15
        x,l=label_history_features(a);y,m=label_history_features(b)
        np.testing.assert_array_equal(x[:3],y[:3]);np.testing.assert_array_equal(l[:3],m[:3])
        self.assertEqual(l[1,0],0);self.assertAlmostEqual(float(l[2,0]),float(np.log1p(100)),places=6)
        self.assertAlmostEqual(float(l[3,0]),float(np.log1p(20)),places=6)

    def test_incomplete_february_target_is_not_available_on_march_first(self):
        a=self.objects().iloc[:2].copy();a.as_of_date=pd.to_datetime(['2018-02-01','2018-03-01']);a.target_end=a.as_of_date+pd.Timedelta(days=30)
        x,l=label_history_features(a);self.assertEqual(x[1,0],0);self.assertTrue(np.isnan(l[1,0]))

    def test_target_cohort_selection_does_not_read_validation_answers(self):
        a=self.objects();a.loc[0,'customer_id']='other';a.loc[2,'customer_id']='new'
        tr=np.array([0,1]);first=restrict_to_target_cohort(a,tr,'2018-08-01')
        a.loc[2,'spend_30d']=1e15
        np.testing.assert_array_equal(first,restrict_to_target_cohort(a,tr,'2018-08-01'))
        self.assertEqual(len(first),0)


if __name__=='__main__':unittest.main()
