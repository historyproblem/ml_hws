import unittest
import numpy as np
import pandas as pd
from spend_baselines.data import PurchaseHistory
from spend_course10.history_features import HISTORY_COLUMNS,monthly_history_features


def history(dates,amounts):
    return PurchaseHistory(pd.DataFrame({'customer_id':'a','date':pd.to_datetime(dates),'amount':amounts,'hour':12.,'dow':0}))


class MonthlyHistoryTests(unittest.TestCase):
    def test_cutoff_and_exact_30day_boundary(self):
        obj=pd.DataFrame({'customer_id':['a'],'as_of_date':pd.to_datetime(['2018-09-01'])})
        h=history(['2018-07-03','2018-08-02','2018-08-31','2018-09-01'],[40,10,20,100000])
        x=monthly_history_features(obj,h,'2016-12-01')
        self.assertAlmostEqual(float(x[0,0]),float(np.log1p(30)),places=6)
        self.assertAlmostEqual(float(x[0,1]),float(np.log1p(40)),places=6)

    def test_future_and_target_poison_leave_features_unchanged(self):
        obj=pd.DataFrame({'customer_id':['a'],'as_of_date':pd.to_datetime(['2018-09-01']),'spend_30d':[1.]})
        h=history(['2018-08-02','2018-09-01','2019-01-01'],[10,20,30])
        poison=history(['2018-08-02','2018-09-01','2019-01-01'],[10,1e15,1e15])
        x=monthly_history_features(obj,h,'2016-12-01');obj.spend_30d=1e15
        np.testing.assert_array_equal(x,monthly_history_features(obj,poison,'2016-12-01'))

    def test_uncovered_months_are_missing_and_covered_empty_months_zero(self):
        obj=pd.DataFrame({'customer_id':['a'],'as_of_date':pd.to_datetime(['2017-02-01'])})
        x=monthly_history_features(obj,history(['2017-01-03'],[3]),'2017-01-01')
        self.assertTrue(np.isnan(x[0,1]));self.assertAlmostEqual(float(x[0,0]),float(np.log1p(3)),places=6)
        self.assertAlmostEqual(float(x[0,HISTORY_COLUMNS.index('covered_fraction_3')]),1/3,places=6)
        y=monthly_history_features(obj,history(['2017-01-01'],[3]),'2017-01-01')
        self.assertEqual(y[0,0],0)


if __name__=='__main__':unittest.main()
