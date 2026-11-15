import runpy,sys,unittest
from pathlib import Path
import numpy as np
import pandas as pd
from spend_baselines.suite_data import split_indices

sys.path.insert(0,str(Path('scripts').resolve()))
try:h=runpy.run_path('scripts/run_snapshot_history.py',run_name='snapshot_helpers')
finally:sys.path.pop(0)


class SnapshotWeightTests(unittest.TestCase):
    def test_future_snapshots_do_not_dilute_completed_customer_month(self):
        dates=pd.to_datetime(['2018-07-01','2018-07-08','2018-07-15','2018-07-22','2018-08-01','2018-08-08','2018-08-15'])
        objects=pd.DataFrame({'Id':list('abcdefg'),'customer_id':'a','as_of_date':dates,
          'target_end':dates+pd.Timedelta(days=30),'official':[True,False,False,False,True,False,False],
          'month':dates.strftime('%Y-%m')})
        p={'dataset':'D2','weights':'customer_month','half_life':None};tr=split_indices(objects,'D2','2018-09-01')
        w=h['weights'](objects,tr,p,'2018-09-01')
        np.testing.assert_array_equal(tr,np.arange(5));np.testing.assert_array_equal(w,[.25,.25,.25,.25,1.])
        shortened=objects.iloc[:5].copy();np.testing.assert_array_equal(w,h['weights'](shortened,np.arange(5),p,'2018-09-01'))

    def test_uniform_normalization_preserves_original_label_weight_sum(self):
        dates=pd.to_datetime(['2018-07-01','2018-07-08','2018-08-01'])
        objects=pd.DataFrame({'Id':list('abc'),'customer_id':'a','as_of_date':dates,'target_end':dates+pd.Timedelta(days=30),'official':[True,False,True],'month':dates.strftime('%Y-%m')})
        p={'dataset':'D2','weights':'normalized_uniform','half_life':None}
        w=h['weights'](objects,np.arange(3),p,'2018-09-01');np.testing.assert_allclose(w,2/3);self.assertAlmostEqual(w.sum(),2)


if __name__=='__main__':unittest.main()
