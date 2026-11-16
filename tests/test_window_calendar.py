import sys,unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path('scripts').resolve()))
from run_augmented_monthly_mlp import window_calendar


class WindowCalendarTests(unittest.TestCase):
    def test_exact_horizon_exposure_and_target_independence(self):
        names=['month_sin','month_cos']+[f'month_{k}' for k in range(1,13)]
        o=pd.DataFrame({'as_of_date':pd.to_datetime(['2018-11-01','2018-10-22']),'spend_30d':[0.,100.]})
        x=np.zeros((2,len(names)))
        a=window_calendar(x,o,names)
        self.assertEqual(a[0,names.index('month_11')],1.)
        self.assertAlmostEqual(a[1,names.index('month_10')],10/30)
        self.assertAlmostEqual(a[1,names.index('month_11')],20/30)
        o.spend_30d=1e15
        np.testing.assert_array_equal(a,window_calendar(x,o,names))


if __name__=='__main__':unittest.main()
