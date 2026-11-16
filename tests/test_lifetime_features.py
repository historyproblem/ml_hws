import unittest
import numpy as np
import pandas as pd
from spend_baselines.data import PurchaseHistory
from spend_course10.lifetime_features import LIFETIME_COLUMNS,lifetime_features


def history(dates,amounts):
    return PurchaseHistory(pd.DataFrame({'customer_id':'a','date':pd.to_datetime(dates),'amount':amounts,'hour':12.,'dow':0}))


class LifetimeTests(unittest.TestCase):
    def test_future_purchase_and_target_poison_are_ignored(self):
        obj=pd.DataFrame({'customer_id':['a'],'as_of_date':pd.to_datetime(['2018-09-01']),'spend_30d':[1.]})
        a=history(['2017-01-01','2018-08-20','2018-09-01','2019-01-01'],[5,10,1,1])
        b=history(['2017-01-01','2018-08-20','2018-09-01','2019-01-01'],[5,10,1e15,1e15])
        x=lifetime_features(obj,a,'2016-12-01');obj.spend_30d=1e15
        np.testing.assert_array_equal(x,lifetime_features(obj,b,'2016-12-01'))
        self.assertAlmostEqual(float(x[0,0]),float(np.log1p(2)),places=6)

    def test_pre_first_observation_windows_do_not_count_as_exposed_months(self):
        obj=pd.DataFrame({'customer_id':['a'],'as_of_date':pd.to_datetime(['2018-09-01'])})
        x=lifetime_features(obj,history(['2018-08-20'],[99]),'2016-12-01')
        self.assertAlmostEqual(float(x[0,LIFETIME_COLUMNS.index('observed_active_mean_log')]),float(np.log1p(99)),places=6)
        self.assertEqual(x[0,LIFETIME_COLUMNS.index('observed_active_zero_fraction')],0)
        self.assertEqual(x[0,LIFETIME_COLUMNS.index('observed_log_repeat_rate30')],0)


if __name__=='__main__':unittest.main()
