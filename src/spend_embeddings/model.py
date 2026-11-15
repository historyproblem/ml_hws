"""End-to-end weighted product embeddings plus the existing Joint MLP loss."""

import os
import random

import numpy as np

from spend_baselines.suite_models import Preprocessor, neural_loss, output_prediction


def make_network(features,vocabulary,settings):
    import torch
    nn=torch.nn

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding=nn.EmbeddingBag(vocabulary,settings['embedding_dim'],mode='sum',
                                           padding_idx=0,include_last_offset=True)
            nn.init.normal_(self.embedding.weight,std=settings['embedding']['initial_std'])
            with torch.no_grad():
                self.embedding.weight[0].zero_()
            layers=[]
            previous=features+settings['embedding_dim']
            for width in settings['hidden_sizes']:
                layers.extend([nn.Linear(previous,width),nn.ReLU(),nn.Dropout(settings['dropout'])])
                previous=width
            self.body=nn.Sequential(*layers)
            self.output=nn.Linear(previous,2)

        def forward(self,tab,ids,weights):
            k=ids.shape[1]
            offsets=torch.arange(0,ids.numel()+1,k,dtype=torch.int64,device=ids.device)
            pooled=self.embedding(ids.reshape(-1),offsets,per_sample_weights=weights.reshape(-1))
            return self.output(self.body(torch.cat([tab,pooled],dim=1)))

    return Network()


class ProductJoint:
    def __init__(self,settings,seed,device='cpu',threads=2):
        self.settings,self.seed,self.device,self.threads=settings,seed,device,threads

    def tensors(self,x,ids,weights):
        import torch
        a=self.preprocessor.transform(x)
        mapped=self.mapping[ids]
        return (torch.from_numpy(a).to(self.device),
                torch.as_tensor(mapped,dtype=torch.int64,device=self.device),
                torch.as_tensor(weights.copy(),dtype=torch.float32,device=self.device))

    def fit(self,x,ids,weights,y,row_weights,mapping,validation=None,epochs=None):
        import torch
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        if self.device=='cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.set_num_threads(self.threads)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(self.seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False
        if not (row_weights>0).all():
            raise ValueError('Product model requires positive purged D2 weights')
        self.mapping=mapping
        self.preprocessor=Preprocessor().fit(x,row_weights)
        tab,ti,tw=self.tensors(x,ids,weights)
        target=torch.as_tensor(y,dtype=torch.float32,device=self.device)
        rw=torch.as_tensor(row_weights,dtype=torch.float32,device=self.device)
        self.network=make_network(tab.shape[1],int(mapping.max())+1,self.settings).to(self.device)
        dense=[p for name,p in self.network.named_parameters() if name!='embedding.weight']
        optimizer=torch.optim.Adam([
            {'params':dense,'weight_decay':self.settings['weight_decay']},
            {'params':[self.network.embedding.weight],'weight_decay':self.settings['embedding']['weight_decay']},
        ],lr=self.settings['learning_rate'])
        if validation is not None:
            vx,vi,vw,vy=validation
            va=self.tensors(vx,vi,vw)
        self.history=[]
        self.best_epoch=0
        best,stale,state=np.inf,0,None
        for epoch in range(epochs or self.settings['max_epochs']):
            self.network.train()
            total,mass=0.,0
            for ix in torch.randperm(len(target),device=self.device).split(self.settings['batch_size']):
                optimizer.zero_grad(set_to_none=True)
                raw=self.network(tab[ix],ti[ix],tw[ix])
                loss=neural_loss(raw,target[ix],rw[ix],'hurdle_joint')
                if not torch.isfinite(loss):raise FloatingPointError('Non-finite embedding loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(),10.)
                optimizer.step()
                total+=float(loss.detach())*len(ix)
                mass+=len(ix)
            record={'epoch':epoch+1,'train_loss':total/mass}
            if validation is not None:
                prediction=self.predict_tensors(va)
                score=float(np.sqrt(np.mean((prediction-np.log1p(vy))**2)))
                if not np.isfinite(score):raise FloatingPointError('Non-finite embedding inner metric')
                record['inner_metric']=score
                if score<best-self.settings['min_delta']:
                    best,stale,self.best_epoch=score,0,epoch+1
                    state={k:v.detach().cpu().clone() for k,v in self.network.state_dict().items()}
                else:stale+=1
            self.history.append(record)
            if (epoch+1)%10==0:print(f'    epoch {epoch+1}: loss={record["train_loss"]:.5f}',flush=True)
            if validation is not None and stale>=self.settings['patience']:break
        if state is not None:self.network.load_state_dict(state)
        else:self.best_epoch=len(self.history)
        self.network.eval()
        return self

    def predict_tensors(self,tensors):
        import torch
        self.network.eval()
        with torch.no_grad():
            return torch.cat([output_prediction(self.network(*(a[ix:ix+2048] for a in tensors)),'hurdle_joint')
                              for ix in range(0,len(tensors[0]),2048)]).cpu().numpy().astype(float)

    def predict(self,x,ids,weights):
        z=self.predict_tensors(self.tensors(x,ids,weights))
        if not np.isfinite(z).all() or (z<0).any():raise FloatingPointError('Invalid log prediction')
        return z

    def save(self,path,item_names):
        import torch
        kept=np.flatnonzero(self.mapping>0)
        torch.save({'format':1,'settings':self.settings,'seed':self.seed,'mapping':self.mapping.tolist(),
                    'item_names':[item_names[i-1] for i in kept],
                    'preprocessing':self.preprocessor.state(),
                    'state_dict':{k:v.detach().cpu() for k,v in self.network.state_dict().items()}},path)

    @classmethod
    def load(cls,path,device='cpu',threads=2):
        import torch
        state=torch.load(path,map_location='cpu',weights_only=True)
        if state['format']!=1:raise ValueError('Unknown embedding checkpoint')
        model=cls(state['settings'],state['seed'],device,threads)
        model.mapping=np.asarray(state['mapping'],dtype=np.int64)
        class LoadedPreprocessor:
            def transform(self,x):return Preprocessor.apply_state(x,state['preprocessing'])
        model.preprocessor=LoadedPreprocessor()
        model.network=make_network(len(state['preprocessing']['mean']),int(model.mapping.max())+1,model.settings).to(device)
        model.network.load_state_dict(state['state_dict'])
        return model
