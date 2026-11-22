"""Small causal retrieval MLP; inspired by TabR, not its reference model."""
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import QuantileTransformer
from torch import nn
from torch.nn import functional as F


class Preprocessor:
    def fit(self, frame, categories):
        self.columns=list(frame);self.categories=list(categories)
        self.numeric=[c for c in frame if c not in categories]
        a=frame[self.numeric].to_numpy(dtype=float);a[~np.isfinite(a)]=np.nan
        self.median=pd.DataFrame(a).median().fillna(0).to_numpy()
        a=np.where(np.isnan(a),self.median,a)
        self.active=np.ptp(a,axis=0)>0
        self.quantile=None
        if self.active.any():
            self.quantile=QuantileTransformer(n_quantiles=min(1000,len(a)),output_distribution='normal',subsample=None,random_state=42).fit(a[:,self.active])
        self.vocab={c:{v:i+1 for i,v in enumerate(sorted(frame[c].fillna('<none>').astype(str).unique()))} for c in categories}
        self.cardinalities=[len(self.vocab[c])+1 for c in categories]
        return self

    def transform(self, frame):
        if list(frame)!=self.columns:raise ValueError('Feature schema differs')
        a=frame[self.numeric].to_numpy(dtype=float);a[~np.isfinite(a)]=np.nan
        missing=np.isnan(a);a=np.where(missing,self.median,a);z=np.zeros_like(a)
        if self.quantile is not None:z[:,self.active]=np.clip(self.quantile.transform(a[:,self.active]),-5,5)
        cat=np.column_stack([frame[c].fillna('<none>').astype(str).map(self.vocab[c]).fillna(0).to_numpy(dtype=np.int64) for c in self.categories])
        return torch.from_numpy(np.column_stack([z,missing]).astype(np.float32)),torch.from_numpy(cat)


class RetrievalMLP(nn.Module):
    def __init__(self,n_numeric,cardinalities,retrieval=True):
        super().__init__();self.retrieval=retrieval;self.k=16
        self.embeddings=nn.ModuleList([nn.Embedding(n,4) for n in cardinalities])
        self.encoder=nn.Sequential(nn.Linear(n_numeric+4*len(cardinalities),64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,64),nn.GELU(),nn.Dropout(.1),nn.LayerNorm(64))
        self.key=nn.Linear(64,64,bias=False)
        self.label=nn.Embedding(2,64)
        self.difference=nn.Sequential(nn.Linear(64,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,64))
        self.gain=nn.Parameter(torch.tensor(.1));self.head=nn.Linear(64,1)

    def encode(self,numeric,categories):
        return self.encoder(torch.cat([numeric,*[e(categories[:,i]) for i,e in enumerate(self.embeddings)]],dim=1))

    def keys(self,h):return F.normalize(self.key(h),dim=1,eps=1e-6)

    @torch.no_grad()
    def cache(self,candidates,batch_size=256):
        training=self.training;self.eval()
        result=torch.cat([self.keys(self.encode(candidates['numeric'][i:i+batch_size],candidates['categories'][i:i+batch_size])) for i in range(0,len(candidates['numeric']),batch_size)])
        self.train(training);return result.detach()

    @torch.no_grad()
    def neighbors(self,key,site,period,candidates,cache):
        distance=(key.detach().square().sum(1,keepdim=True)+cache.square().sum(1)[None]-2*key.detach()@cache.T).clamp_min(0)
        forbidden=(site[:,None]==candidates['site'][None])|(candidates['period'][None]>=period[:,None])
        distance.masked_fill_(forbidden,float('inf'))
        values,indices=torch.topk(distance,k=min(self.k,len(cache)),dim=1,largest=False,sorted=True)
        return indices,torch.isfinite(values)

    def forward(self,numeric,categories,site,period,candidates=None,cache=None,return_neighbors=False):
        h=self.encode(numeric,categories)
        if not self.retrieval:return self.head(h).squeeze(1)
        key=self.keys(h);indices,valid=self.neighbors(key,site,period,candidates,cache)
        b,k=indices.shape
        neighbor=self.encode(candidates['numeric'][indices.reshape(-1)],candidates['categories'][indices.reshape(-1)]).reshape(b,k,64)
        neighbor_key=self.keys(neighbor.reshape(-1,64)).reshape(b,k,64)
        attention=(-8*(key[:,None]-neighbor_key).square().sum(2)).masked_fill(~valid,-1e9).softmax(1)
        attention=attention*valid;attention=attention/attention.sum(1,keepdim=True).clamp_min(1e-12)
        value=self.label(candidates['target'][indices].long())+self.difference(neighbor-h[:,None])
        aggregate=(attention[:,:,None]*value).sum(1)
        logits=self.head(h+self.gain*aggregate).squeeze(1)
        return (logits,indices,valid,aggregate) if return_neighbors else logits


def targeted_tests():
    torch.manual_seed(42);a=RetrievalMLP(4,[3],True)
    torch.manual_seed(42);b=RetrievalMLP(4,[3],False)
    assert all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict())
    candidate=dict(numeric=torch.randn(5,4),categories=torch.tensor([[0],[1],[2],[1],[0]]),site=torch.tensor([1,2,3,4,5]),period=torch.tensor([0,0,1,2,3]),target=torch.tensor([0.,1.,0.,1.,0.]))
    a.eval();q=torch.randn(3,4);cat=torch.tensor([[0],[1],[2]]);sites=torch.tensor([1,9,9]);period=torch.tensor([0,1,2])
    cache=a.cache(candidate);p,indices,valid,aggregate=a(q,cat,sites,period,candidate,cache,True)
    assert torch.isfinite(p).all() and torch.equal(aggregate[0],torch.zeros(64)) and valid.sum(1).tolist()==[0,2,3]
    assert all(torch.all(candidate['period'][indices[i,valid[i]]]<period[i]) and torch.all(candidate['site'][indices[i,valid[i]]]!=sites[i]) for i in range(3))
    # Future/same-site labels cannot affect a query, even when topk pads masks.
    original=a(q[:1],cat[:1],sites[:1],torch.tensor([1]),candidate,cache)
    changed={**candidate,'target':candidate['target'].clone()};changed['target'][[0,2,3,4]]=1-changed['target'][[0,2,3,4]]
    assert torch.equal(original,a(q[:1],cat[:1],sites[:1],torch.tensor([1]),changed,cache))
    a.zero_grad();loss=a(q,cat,sites,period,candidate,cache).sum();loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in a.parameters() if p.grad is not None)
    assert a.label.weight.grad.abs().sum()>0 and a.key.weight.grad.abs().sum()>0
    clone=RetrievalMLP(4,[3],True);clone.load_state_dict(a.state_dict());clone.eval()
    assert torch.equal(a(q,cat,sites,period,candidate,a.cache(candidate)),clone(q,cat,sites,period,candidate,clone.cache(candidate)))
    f=pd.DataFrame({'x':[1.,2.,np.nan,2.],'constant':[np.nan]*4,'category':['a','b','a',None]})
    prep=Preprocessor().fit(f,['category']);n,c=prep.transform(pd.DataFrame({'x':[100.,np.nan],'constant':[7.,np.nan],'category':['unseen','a']}))
    assert torch.isfinite(n).all() and c[0,0]==0 and prep.median.tolist()==[2.,0.]
    return {'status':'PASS','checks':['same_initialized_control','empty_zero_aggregate','fewer_than_k_mask','same_site_future_exclusion','forbidden_label_mutation_invariant','finite_gradients','key_label_gradients','state_reload_exact','train_only_unknown_preprocessing']}
