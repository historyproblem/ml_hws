"""One joint Ridge with categorical customer indicators and separate L2 strengths."""
import numpy as np


class CustomerRidge:
    """Equivalent to Ridge on [X, sqrt(alpha/lambda)*one_hot(customer)].

    Eliminate categorical coefficients analytically to avoid a large dense
    design. Effects are jointly fitted model parameters, never target features.
    """
    def __init__(self,alpha=100.,customer_alpha=10.):
        if alpha<=0 or customer_alpha<=0:raise ValueError('Both penalties must be positive')
        self.alpha,self.customer_alpha=float(alpha),float(customer_alpha)

    def fit(self,x,y,customers):
        x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64);customers=np.asarray(customers,dtype=str)
        if x.ndim!=2 or y.shape!=(len(x),) or len(customers)!=len(x) or not len(x):raise ValueError('Inconsistent training arrays')
        if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Finite training arrays required')
        names,group=np.unique(customers,return_inverse=True)
        a=np.c_[np.ones(len(x)),x];n=np.bincount(group).astype(float)
        sx=np.zeros((len(names),a.shape[1]),dtype=float);np.add.at(sx,group,a)
        sy=np.bincount(group,weights=y);denominator=n+self.customer_alpha
        lhs=a.T@a-(sx.T/denominator)@sx
        penalty=np.r_[0.,np.full(x.shape[1],self.alpha)];lhs.flat[::lhs.shape[0]+1]+=penalty
        rhs=a.T@y-sx.T@(sy/denominator)
        beta=np.linalg.solve(lhs,rhs)
        effect=(sy-sx@beta)/denominator
        self.intercept_=float(beta[0]);self.coef_=beta[1:];self.effects_={str(cid):float(u) for cid,u in zip(names,effect)}
        self.counts_={str(cid):int(v) for cid,v in zip(names,n)}
        return self

    def predict_raw(self,x,customers):
        x=np.asarray(x,dtype=np.float64)
        if len(x)!=len(customers):raise ValueError('One customer per row required')
        effects=np.array([self.effects_.get(str(cid),0.) for cid in customers],dtype=float)
        return x@self.coef_+self.intercept_+effects

    def predict(self,x,customers):return np.maximum(self.predict_raw(x,customers),0)
