"""Two fixed sample-weight rules; neither reads a purchase target."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from spend_baselines.suite_models import Preprocessor


def sample_weights(profile, train_dates, source_x, forecast_x, cutoff):
    if profile['weighting']=='calendar':
        months=pd.DatetimeIndex(train_dates).month.to_numpy()
        delta=np.abs(months-pd.Timestamp(cutoff).month)
        delta=np.minimum(delta,12-delta)
        w=.25+.75*np.exp(-.5*(delta/1.5)**2)
        metadata={'rule':'0.25+0.75*exp(-0.5*(cyclic_month_distance/1.5)^2)'}
    elif profile['weighting']=='covariate':
        # Scaling is fitted on source rows; the domain classifier sees only
        # unlabelled, already available covariates of the forecast population.
        pre=Preprocessor().fit(source_x,np.ones(len(source_x)))
        a=pre.transform(source_x).astype(np.float64)
        b=pre.transform(forecast_x).astype(np.float64)
        labels=np.r_[np.zeros(len(a)),np.ones(len(b))]
        classifier=LogisticRegression(C=.1,class_weight='balanced',max_iter=1000,random_state=42)
        classifier.fit(np.r_[a,b],labels)
        decision=classifier.decision_function(a)
        w=np.exp(np.clip(decision,np.log(.2),np.log(5.)))
        metadata={'classifier':'LogisticRegression','C':.1,'class_weight':'balanced','clip':[.2,5.],
                  'iterations':classifier.n_iter_.tolist(), 'purchase_targets_used':False}
    else:
        raise ValueError('Unknown fixed weighting rule')
    w=w/w.mean()
    assert np.isfinite(w).all() and (w>0).all()
    metadata.update(effective_n=float(w.sum()**2/np.sum(w*w)),n_source=len(w),n_forecast=len(forecast_x),
                    min_weight=float(w.min()),max_weight=float(w.max()),weight_quantiles=np.quantile(w,[.1,.5,.9]).tolist())
    return w,metadata
