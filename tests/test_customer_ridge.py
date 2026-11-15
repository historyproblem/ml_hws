import unittest
import numpy as np
from sklearn.linear_model import Ridge
from spend_course10.customer_ridge import CustomerRidge


class CustomerRidgeTests(unittest.TestCase):
    def test_joint_solver_matches_explicit_categorical_ridge_and_unknown_fallback(self):
        rng=np.random.default_rng(8);x=rng.normal(size=(80,5));y=rng.normal(3,2,size=80)
        ids=np.array([str(k%9) for k in range(80)]);names=np.unique(ids)
        alpha,individual=7.,3.;scale=np.sqrt(alpha/individual)
        onehot=(ids[:,None]==names[None,:]).astype(float)*scale
        explicit=Ridge(alpha=alpha).fit(np.c_[x,onehot],y)
        model=CustomerRidge(alpha,individual).fit(x,y,ids)
        test=rng.normal(size=(12,5));customers=np.array(['unknown']+[str(k%9) for k in range(11)])
        dummy=(customers[:,None]==names[None,:]).astype(float)*scale
        np.testing.assert_allclose(model.predict_raw(test,customers),explicit.predict(np.c_[test,dummy]),rtol=1e-11,atol=1e-11)
        self.assertAlmostEqual(model.predict_raw(test[:1],['unknown'])[0],float(test[0]@model.coef_+model.intercept_),places=12)

    def test_row_order_is_invariant_in_float64(self):
        rng=np.random.default_rng(4);x=rng.normal(size=(90,4));y=rng.normal(size=90);ids=np.array([str(k%12) for k in range(90)])
        ix=rng.permutation(90)
        a=CustomerRidge().fit(x,y,ids);b=CustomerRidge().fit(x[ix],y[ix],ids[ix])
        np.testing.assert_allclose(a.predict_raw(x,ids),b.predict_raw(x,ids),rtol=1e-11,atol=1e-11)


if __name__=='__main__':unittest.main()
