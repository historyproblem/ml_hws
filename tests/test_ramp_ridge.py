import tempfile,unittest
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import Ridge
from spend_course10.ramp_ridge import RampDesign,RampRidge


class RampRidgeTests(unittest.TestCase):
    def test_nonlinear_response_and_constant_extrapolation(self):
        x=np.linspace(-2,2,161)[:,None]
        z=1+np.maximum(x[:,0],0)
        d=RampDesign().fit(x,['recency']);m=RampRidge(d,.001).fit_design(d.transform(x),z)
        simple=Ridge(alpha=.001).fit(x,z)
        self.assertLess(np.mean((m.predict(x)-z)**2),np.mean((simple.predict(x)-z)**2)/20)
        np.testing.assert_array_equal(m.predict(np.array([[-1e12],[-2.]])),np.repeat(m.predict(x[:1]),2))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'m.joblib';joblib.dump(m,p)
            np.testing.assert_array_equal(m.predict(x),joblib.load(p).predict(x))

    def test_transform_never_refits_knots_or_uses_a_response(self):
        x=np.column_stack([np.arange(15.),np.ones(15),np.full(15,np.nan)])
        d=RampDesign().fit(x,['recency','month_1','year_spend30'])
        expected={j:k.copy() for j,k in d.knots.items()}
        self.assertTrue(np.isfinite(d.transform(np.array([[1e12,0,np.nan]]))).all())
        for j,k in expected.items():np.testing.assert_array_equal(k,d.knots[j])
        with self.assertRaises(ValueError):RampDesign().fit(x,['recency','target_z','other'])


if __name__=='__main__':unittest.main()
