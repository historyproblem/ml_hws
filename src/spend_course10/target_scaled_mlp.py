"""The unchanged direct MLP loss with a TRAIN-only affine target transform."""
import numpy as np
from spend_baselines.suite_models import Neural


class TargetScaledMLP(Neural):
    def fit(self,x,y,w,validation=None,epochs=None):
        if self.task!='direct':raise ValueError('Only plain direct log regression is supported')
        z=np.log1p(np.asarray(y,dtype=float));w=np.asarray(w,dtype=float)
        self.target_mean=float(np.average(z,weights=w))
        variance=float(np.average((z-self.target_mean)**2,weights=w))
        self.target_scale=float(np.sqrt(variance)) if variance>1e-12 else 1.
        transformed=np.expm1((z-self.target_mean)/self.target_scale)
        # The inherited direct MSE sees standardised log targets. Validation
        # retains original spend: _predict_tensor already performs the inverse.
        return super().fit(x,transformed,w,validation=validation,epochs=epochs)

    def _predict_tensor(self,x):
        import torch
        self.network.eval()
        with torch.no_grad():
            raw=torch.cat([self.network(chunk)[:,0] for chunk in x.split(4096)])
        return np.maximum(raw.cpu().numpy().astype(float)*self.target_scale+self.target_mean,0)

    def save(self,path):
        import torch
        super().save(path)
        state=torch.load(path,map_location='cpu',weights_only=True)
        state['target_affine']={'mean':self.target_mean,'scale':self.target_scale}
        torch.save(state,path)

    @classmethod
    def load(cls,path,device='cpu',threads=2):
        import torch
        model=super().load(path,device,threads)
        state=torch.load(path,map_location='cpu',weights_only=True)
        model.target_mean=float(state['target_affine']['mean'])
        model.target_scale=float(state['target_affine']['scale'])
        assert np.isfinite([model.target_mean,model.target_scale]).all() and model.target_scale>0
        return model
