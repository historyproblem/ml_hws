import unittest
import numpy as np
import pandas as pd
from spend_course10.transfer_weights import sample_weights


class TransferWeightsTests(unittest.TestCase):
    def test_cyclic_calendar_treats_december_as_adjacent_to_january(self):
        dates=pd.to_datetime(['2017-12-01','2018-02-01','2018-07-01'])
        w,_=sample_weights({'weighting':'calendar'},dates,np.zeros((3,1)),np.zeros((2,1)),'2018-01-01')
        self.assertEqual(w[0],w[1]);self.assertGreater(w[0],w[2]);self.assertAlmostEqual(w.mean(),1)

    def test_domain_weights_are_bounded_and_use_no_purchase_labels(self):
        x=np.linspace(-2,2,80).reshape(-1,1);target=np.linspace(1,3,20).reshape(-1,1)
        dates=pd.to_datetime(['2018-01-01']*len(x))
        w,m=sample_weights({'weighting':'covariate'},dates,x,target,'2018-09-01')
        self.assertGreater(w[-1],w[0]);self.assertAlmostEqual(w.mean(),1)
        self.assertLessEqual(w.max()/w.min(),25+1e-10)
        self.assertFalse(m['purchase_targets_used'])


if __name__=='__main__':unittest.main()
