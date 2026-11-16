"""Ordinary Ridge on continuous piecewise-linear functions and a few products.

All knot positions, imputation and input ranges are fitted on TRAIN covariates.
This is a fixed nonlinear feature map (course7) followed by one L2 regression.
"""
import numpy as np
from sklearn.linear_model import Ridge
from spend_baselines.suite_models import Preprocessor


PAIRS=[('recency','n_90'),('recency','mean_log_spend_6'),('year_spend30','recency')]


class RampDesign:
    def __init__(self,interactions=False):
        self.interactions=interactions

    def fit(self,x,names):
        if any(n in names for n in ['spend_30d','target_z']):
            raise ValueError('Responses are not input features')
        self.names=list(names)
        self.preprocessor=Preprocessor().fit(x,np.ones(len(x)))
        a=self.preprocessor.transform(x).astype(np.float64)
        self.low=a.min(axis=0);self.high=a.max(axis=0)
        self.knots={}
        for j,name in enumerate(names):
            if name.startswith('month_') or 'coverage' in name or 'covered_fraction' in name:
                continue
            k=np.unique(np.quantile(a[:,j],[0,.2,.4,.6,.8,1.]))
            if len(k)>2:self.knots[j]=k
        self.pairs=[]
        if self.interactions:
            for left,right in PAIRS:
                i,j=names.index(left),names.index(right)
                if i in self.knots and j in self.knots:self.pairs.append((i,j))
        self.n_original=a.shape[1]
        self.n_output=self.n_original+sum(len(k)-1 for k in self.knots.values())
        self.n_output+=sum((len(self.knots[i])-1)*(len(self.knots[j])-1) for i,j in self.pairs)
        return self

    def transform(self,x):
        a=self.preprocessor.transform(x).astype(np.float64)
        a=np.clip(a,self.low,self.high)
        parts=[a];ramps={}
        for j,k in self.knots.items():
            ramps[j]=np.clip((a[:,j,None]-k[:-1])/(k[1:]-k[:-1]),0,1)
            parts.append(ramps[j])
        for i,j in self.pairs:
            parts.append((ramps[i][:,:,None]*ramps[j][:,None,:]).reshape(len(x),-1))
        result=np.column_stack(parts)
        assert result.shape==(len(x),self.n_output) and np.isfinite(result).all()
        return result


class RampRidge:
    def __init__(self,design,alpha):
        self.design,self.alpha=design,float(alpha)

    def fit_design(self,a,z):
        assert a.dtype==np.float64
        self.estimator=Ridge(alpha=self.alpha,solver='cholesky').fit(a,z)
        assert self.estimator.coef_.dtype==np.float64
        return self

    def predict(self,x):
        return np.maximum(self.estimator.predict(self.design.transform(x)),0)
