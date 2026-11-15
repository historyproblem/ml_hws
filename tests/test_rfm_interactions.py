import unittest
import numpy as np
from spend_course10.rfm_interactions import rfm_interactions


class RFMInteractionTests(unittest.TestCase):
    def test_fixed_boundaries_and_numeric_season_interaction(self):
        names=['recency','n_180','mean_log_spend_6','mean_log_spend_12','positive_mean_log_spend_12']+[f'month_{m}' for m in range(1,13)]
        x=np.zeros((3,len(names)),dtype=np.float32)
        x[:,0]=np.log1p([7,30,91]);x[:,1]=np.log1p([1,5,11]);x[:,2]=[2,3,4];x[:,3]=[1,2,3];x[:,4]=[6,7,8];x[:,names.index('month_11')]=1
        a,columns=rfm_interactions(x,names,'all')
        np.testing.assert_array_equal(a[:,[columns.index(f'recency_bin_{i}') for i in range(4)]],np.eye(4)[[0,1,3]])
        np.testing.assert_array_equal(a[:,columns.index('month_11_mean_log12')],[1,2,3])
        np.testing.assert_array_equal(a[:,columns.index('mean_log_12_recency_0')],[1,0,0])

    def test_missing_positive_mean_does_not_pollute_other_bins(self):
        names=['recency','n_180','mean_log_spend_6','mean_log_spend_12','positive_mean_log_spend_12']+[f'month_{m}' for m in range(1,13)]
        x=np.zeros((1,len(names)),dtype=np.float32);x[0,4]=np.nan;x[0,names.index('month_1')]=1
        a,columns=rfm_interactions(x,names,'personalized')
        self.assertTrue(np.isnan(a[0,columns.index('positive_log_frequency_0')]))
        self.assertEqual(a[0,columns.index('positive_log_frequency_1')],0)


if __name__=='__main__':unittest.main()
