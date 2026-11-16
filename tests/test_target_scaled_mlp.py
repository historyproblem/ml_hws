import tempfile,unittest
from pathlib import Path
import numpy as np
from spend_course10.target_scaled_mlp import TargetScaledMLP


class TargetScaledTests(unittest.TestCase):
    def test_train_only_affine_stats_and_checkpoint(self):
        x=np.linspace(-1,1,12)[:,None];z=np.arange(12.)/2;y=np.expm1(z)
        settings=dict(hidden_sizes=[8],dropout=.1,weight_decay=.003,learning_rate=.001,batch_size=8,max_epochs=2,patience=1,min_delta=.0001)
        m=TargetScaledMLP('mlp','direct',settings,42,'cpu',1).fit(x,y,np.ones(12),validation=(x,np.full(12,1e10)),epochs=2)
        self.assertAlmostEqual(m.target_mean,z.mean())
        self.assertAlmostEqual(m.target_scale,z.std())
        self.assertTrue(np.isfinite(m.predict(x)).all())
        self.assertTrue((m.predict(x)>=0).all())
        metric=np.sqrt(np.mean((np.log1p(np.full(12,1e10))-m.predict(x))**2))
        selected=next(r['inner_metric'] for r in m.history if r['epoch']==m.best_epoch)
        self.assertAlmostEqual(metric,selected,places=12)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'m.pt';m.save(p)
            n=TargetScaledMLP.load(p,threads=1)
            self.assertEqual(n.target_mean,m.target_mean);self.assertEqual(n.target_scale,m.target_scale)
            np.testing.assert_array_equal(n.predict(x),m.predict(x))


if __name__=='__main__':unittest.main()
