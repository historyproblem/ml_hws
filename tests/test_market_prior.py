import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from spend_baselines.data import PurchaseHistory
from spend_course10.centred_neural import CentredNeural
from spend_course10.market_prior import cohort_mean, market_forecasts, centre_features_by_date


def history(dates, amounts):
    return PurchaseHistory(pd.DataFrame({'customer_id':'a','date':pd.to_datetime(dates),'amount':amounts,'hour':12.,'dow':0}))


class MarketPriorTests(unittest.TestCase):
    def test_future_and_current_targets_do_not_change_prior(self):
        o=pd.DataFrame({'customer_id':['a'],'as_of_date':pd.to_datetime(['2018-09-01']),'spend_30d':[0.]})
        dates=['2017-04-01','2017-09-10','2018-06-10','2018-08-31','2018-09-01','2018-12-01']
        a=history(dates,[10,20,30,40,50,60])
        b=history(dates,[10,20,30,40,1e12,1e12])
        x,_=market_forecasts(o,a,'2016-12-01')
        o.spend_30d=1e12
        y,_=market_forecasts(o,b,'2016-12-01')
        np.testing.assert_array_equal(x,y)

    def test_zero_buyers_and_right_open_window(self):
        h=history(['2018-03-05','2018-08-31','2018-09-01','2018-10-01'],[1,5,10,100000])
        self.assertAlmostEqual(cohort_mean(h,'2018-09-01'),np.log1p(10))
        self.assertAlmostEqual(cohort_mean(h,'2018-10-02'),0)

    def test_date_centred_features_do_not_use_future_cohorts(self):
        x=np.array([[1.,np.nan],[3.,np.nan],[100.,9.]])
        dates=np.array(['2018-08-01','2018-08-01','2018-09-01'])
        a=centre_features_by_date(x,dates)
        np.testing.assert_array_equal(a[:2],np.array([[-1.,0.],[1.,0.]]))
        x[-1]=1e15
        np.testing.assert_array_equal(a[:2],centre_features_by_date(x,dates)[:2])

    def test_signed_residual_checkpoint_and_population_centring(self):
        x=np.array([[-1.],[0.],[1.],[2.]])
        z=np.array([-2.,-1.,1.,2.])
        settings=dict(hidden_sizes=[8],dropout=0.,weight_decay=.003,learning_rate=.01,batch_size=4,max_epochs=3,patience=2,min_delta=.0001)
        m=CentredNeural('mlp','direct',settings,42,'cpu',1).fit_residual(x,z,np.ones(4),epochs=3)
        a=m.predict(x)
        self.assertAlmostEqual(a.mean(),0,places=12)
        self.assertLess(a.min(),0)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'model.pt';m.save(p)
            np.testing.assert_array_equal(a,CentredNeural.load(p,threads=1).predict(x))


if __name__=='__main__':
    unittest.main()
