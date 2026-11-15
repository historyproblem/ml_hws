"""Seasonal transaction aggregates respect both cutoff and coverage boundaries."""
import runpy
import unittest
import numpy as np
import pandas as pd
from spend_baselines.data import PurchaseHistory

h=runpy.run_path('scripts/course10_forest_followup.py',run_name='forest_helpers')


def history(dates,amounts):
    return PurchaseHistory(pd.DataFrame({'customer_id':'a','date':pd.to_datetime(dates),
        'amount':amounts,'hour':12.,'dow':0}))


class ForestFeatureTests(unittest.TestCase):
    def test_future_poison_and_yearago_right_edge_do_not_enter_features(self):
        obj=pd.DataFrame({'customer_id':['a'],'as_of_date':pd.to_datetime(['2018-09-01'])})
        dates=['2017-06-01','2017-09-01','2017-09-30','2017-10-01','2018-08-31','2018-09-01','2019-01-01']
        a=history(dates,[1,10,20,5000,7,100,100])
        b=history(dates,[1,10,20,5000,7,1e15,1e15])
        x=h['seasonal_features'](obj,a,'2016-12-01');y=h['seasonal_features'](obj,b,'2016-12-01')
        np.testing.assert_array_equal(x,y)
        self.assertAlmostEqual(float(x[0,0]),float(np.log1p(30)),places=6)
        self.assertAlmostEqual(float(x[0,1]),float(np.log1p(2)),places=6)

    def test_uncovered_year_is_missing_and_covered_empty_window_is_zero(self):
        obj=pd.DataFrame({'customer_id':['a','a'],'as_of_date':pd.to_datetime(['2017-09-01','2018-09-01'])})
        x=h['seasonal_features'](obj,history(['2017-01-01'],[5]),'2016-12-01')
        self.assertTrue(np.isnan(x[0,0]));self.assertEqual(x[1,0],0)
        ci=h['SEASON_COLUMNS'].index('year_coverage30')
        self.assertEqual(x[0,ci],0);self.assertEqual(x[1,ci],1)


if __name__=='__main__':unittest.main()
