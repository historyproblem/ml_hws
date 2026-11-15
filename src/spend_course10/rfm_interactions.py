"""Interpretable fixed RFM indicator interactions for ordinary Ridge regression."""
import numpy as np


def rfm_interactions(x,names,variant):
    if variant not in {'none','bins','personalized','all'}:raise ValueError('Unknown interaction block')
    if variant=='none':return np.empty((len(x),0),dtype=np.float32),[]
    if any(n in names for n in ['spend_30d','target_z']):raise ValueError('Targets are not input features')
    recency=np.rint(np.expm1(x[:,names.index('recency')]))
    frequency=np.rint(np.expm1(x[:,names.index('n_180')]))
    r=np.eye(4,dtype=np.float32)[np.digitize(recency,[7,30,90],right=True)]
    f=np.eye(5,dtype=np.float32)[np.digitize(frequency,[1,2,5,10],right=True)]
    blocks=[r,f];columns=[f'recency_bin_{j}' for j in range(4)]+[f'frequency_bin_{j}' for j in range(5)]
    if variant in {'bins','all'}:
        blocks.append((r[:,:,None]*f[:,None,:]).reshape(len(x),-1))
        columns+=[f'recency_{i}_frequency_{j}' for i in range(4) for j in range(5)]
    if variant in {'personalized','all'}:
        for period in [6,12]:
            mean=x[:,names.index(f'mean_log_spend_{period}')]
            blocks.append(np.where(r.astype(bool),mean[:,None],0))
            columns+=[f'mean_log_{period}_recency_{j}' for j in range(4)]
        positive=x[:,names.index('positive_mean_log_spend_12')]
        blocks.append(np.where(f.astype(bool),positive[:,None],0))
        columns+=[f'positive_log_frequency_{j}' for j in range(5)]
        mean=x[:,names.index('mean_log_spend_12')]
        month=x[:,[names.index(f'month_{m}') for m in range(1,13)]]
        blocks.append(np.where(month!=0,month*mean[:,None],0))
        columns+=[f'month_{m}_mean_log12' for m in range(1,13)]
    result=np.column_stack(blocks).astype(np.float32)
    assert result.shape==(len(x),len(columns)) and not np.isinf(result).any()
    return result,columns
