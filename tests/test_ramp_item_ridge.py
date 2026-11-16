import unittest
import numpy as np
from scipy import sparse
from spend_course10.ramp_item_ridge import RampItemDesign
from spend_course10.sparse_item_ridge import SparseItemRidge


class RampItemTests(unittest.TestCase):
    def test_block_scaling_equals_explicit_diagonal_penalties(self):
        rng=np.random.default_rng(42)
        x=rng.normal(size=(50,6));z=3+rng.normal(size=50)
        h=sparse.csr_matrix(rng.poisson(1,size=(50,4)),dtype=np.float32)
        names=['recency','n_90','mean_log_spend_6','year_spend30','month_sin','month_cos']
        d=RampItemDesign(names,min_customers=2,bag_scale=.5).fit(x,h,np.arange(50).astype(str),list('abcd'))
        a=d.transform(x,h);m=SparseItemRidge(d,10).fit_design(a,z)
        original=a.toarray();start=d.ramps.n_output;stop=start+len(d.columns)
        original[:,start:stop]*=10 # return SKU block to its original scale5
        penalty=np.full(original.shape[1],10.);penalty[start:stop]=1000.
        mean=original.mean(axis=0);b=original-mean
        beta=np.linalg.solve(b.T@b+np.diag(penalty),b.T@(z-z.mean()))
        expected=np.maximum((original-mean)@beta+z.mean(),0)
        np.testing.assert_allclose(m.predict(x,h),expected,rtol=0,atol=2e-6)


if __name__=='__main__':unittest.main()
