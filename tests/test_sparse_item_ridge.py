import unittest
import numpy as np
import pandas as pd
from scipy import sparse
from spend_course10.sparse_item_ridge import SparseItemDesign
import sys
from pathlib import Path
sys.path.insert(0,str(Path('scripts').resolve()))
from run_customer_svd import build_matrix


class SparseItemTests(unittest.TestCase):
    def test_vocabulary_counts_customers_not_repeated_snapshots(self):
        h=sparse.csr_matrix([[1,1,0],[1,1,0],[1,0,0]],dtype=np.float32)
        d=SparseItemDesign(min_customers=2).fit(np.arange(3.)[:,None],h,['a','a','b'],['common','rare','future'])
        self.assertEqual(d.item_names,['common'])
        np.testing.assert_array_equal(d.support,[2,1,0])
        x=d.transform(np.array([[1.]]),sparse.csr_matrix([[0,0,100]],dtype=np.float32))
        self.assertTrue(np.isfinite(x.data).all())
        self.assertEqual(x[0,1],0) # unknown item cannot get a learned coefficient

    def test_each_raw_bag_excludes_its_current_and_future_events(self):
        o=pd.DataFrame({'customer_id':['a'],'as_of_date':pd.to_datetime(['2018-09-01'])})
        a=pd.DataFrame({'customer_id':['a','a'],'event_ts':pd.to_datetime(['2018-08-31','2018-09-01']),'item_id':['old','future']})
        b=a.copy();b.loc[1,'item_id']='new_future'
        h,names=build_matrix(o,a);k,other=build_matrix(o,b)
        self.assertEqual(h[0,list(names).index('future')],0)
        self.assertEqual(k[0,list(other).index('new_future')],0)
        self.assertEqual(h[0,list(names).index('old')],k[0,list(other).index('old')])


if __name__=='__main__':unittest.main()
