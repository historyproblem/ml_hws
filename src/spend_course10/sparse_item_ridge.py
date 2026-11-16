"""One ordinary Ridge with sparse product-identity covariates; no embeddings."""
import numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize
from spend_baselines.suite_models import Preprocessor


class SparseItemDesign:
    def __init__(self,min_customers=10,bag_scale=5.):
        self.min_customers,self.bag_scale=min_customers,bag_scale

    def fit(self,x,histories,customers,item_names):
        self.preprocessor=Preprocessor().fit(x,np.ones(len(x)))
        _,inverse=np.unique(np.asarray(customers,dtype=str),return_inverse=True)
        grouping=sparse.csr_matrix((np.ones(len(x)),(inverse,np.arange(len(x)))),shape=(inverse.max()+1,len(x)))
        present=histories.copy();present.data[:]=1
        self.support=np.asarray((grouping@present).getnnz(axis=0)).ravel()
        self.columns=np.flatnonzero(self.support>=self.min_customers)
        self.item_names=np.asarray(item_names,dtype=str)[self.columns].tolist()
        return self

    def transform(self,x,histories):
        tab=self.preprocessor.transform(x).astype(np.float64)
        selected=histories[:,self.columns].astype(np.float64)
        total=np.asarray(histories.sum(axis=1,dtype=np.float64)).ravel()
        kept=np.asarray(selected.sum(axis=1)).ravel()
        coverage=np.divide(kept,total,out=np.zeros(len(x)),where=total>0)
        diversity=np.log1p(histories.getnnz(axis=1))
        bag=normalize(selected,norm='l2',copy=True)*self.bag_scale if len(self.columns) else selected
        result=sparse.hstack([sparse.csr_matrix(tab),bag,sparse.csr_matrix(np.c_[coverage,diversity])],format='csr')
        assert result.dtype==np.float64 and np.isfinite(result.data).all()
        return result


class SparseItemRidge:
    def __init__(self,design,alpha):self.design,self.alpha=design,float(alpha)

    def fit_design(self,a,z):
        self.estimator=Ridge(alpha=self.alpha,solver='lsqr',tol=1e-7,max_iter=500).fit(a,z)
        if np.max(self.estimator.n_iter_)>=500:raise RuntimeError('Sparse Ridge reached its iteration limit')
        assert self.estimator.coef_.dtype==np.float64
        return self

    def predict(self,x,histories):
        return np.maximum(self.estimator.predict(self.design.transform(x,histories)),0)
