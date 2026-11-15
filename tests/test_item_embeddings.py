"""Future poison, fold-local vocabularies, pooling, gradients and checkpoints."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from spend_embeddings.data import build_bags,fit_vocabulary
from spend_embeddings.model import ProductJoint,make_network
from spend_embeddings.suite import read_config


class ProductHistoryTests(unittest.TestCase):
    def setUp(self):
        self.settings=read_config('configs/item_embeddings.json')['embedding']

    def test_future_items_do_not_change_training_bags_or_local_codes(self):
        objects=pd.DataFrame({'customer_id':['a'],'as_of_date':pd.to_datetime(['2018-08-01'])})
        past=pd.DataFrame({'customer_id':['a','a'],'item_id':['b','z'],
                           'event_ts':pd.to_datetime(['2018-07-01','2018-07-03'])})
        future=pd.concat([past,pd.DataFrame({'customer_id':['a'],'item_id':['a'],
                                            'event_ts':pd.to_datetime(['2018-08-01'])})],ignore_index=True)
        a,w,_=build_bags(objects,past,['b','z'],self.settings)
        b,v,_=build_bags(objects,future,['a','b','z'],self.settings)
        np.testing.assert_allclose(w,v,atol=0,rtol=0)
        ma,_=fit_vocabulary(a,np.array(['a']),2,1)
        mb,_=fit_vocabulary(b,np.array(['a']),3,1)
        np.testing.assert_array_equal(ma[a],mb[b])
        self.assertEqual(mb[1],0)  # future-only product has no learned embedding

    def test_right_cutoff_and_left_window_boundary(self):
        t=pd.Timestamp('2018-08-01')
        objects=pd.DataFrame({'customer_id':['a'],'as_of_date':[t]})
        lines=pd.DataFrame({'customer_id':['a']*4,'item_id':['left','outside','inside','right'],
                            'event_ts':[t-pd.Timedelta(days=180),t-pd.Timedelta(days=181),
                                        t-pd.Timedelta(days=1),t]})
        names=sorted(lines.item_id)
        ids,w,_=build_bags(objects,lines,names,self.settings)
        found={names[i-1] for i in ids[0] if i}
        self.assertEqual(found,{'left','inside'})
        self.assertAlmostEqual(w.sum(),1.,places=6)

    def test_product_day_is_counted_once_and_topk_is_deterministic(self):
        objects=pd.DataFrame({'customer_id':['a'],'as_of_date':pd.to_datetime(['2018-08-01'])})
        lines=pd.DataFrame({'customer_id':['a']*4,'item_id':['x','x','y','z'],
                            'event_ts':pd.to_datetime(['2018-07-01 10:00','2018-07-01 11:00',
                                                       '2018-07-01 10:00','2018-07-01 10:00'])})
        s={**self.settings,'max_items':2}
        ids,w,diag=build_bags(objects,lines,['x','y','z'],s)
        np.testing.assert_array_equal(ids[0],[1,2])
        np.testing.assert_allclose(w[0],[.5,.5])
        self.assertEqual(diag[0,0],3)
        self.assertAlmostEqual(diag[0,1],2/3,places=6)

    def test_vocabulary_support_counts_customers_not_snapshots(self):
        ids=np.array([[1,2,0],[1,2,0],[2,0,0]],dtype=np.uint32)
        mapping,support=fit_vocabulary(ids,np.array(['a','a','b']),3,2)
        np.testing.assert_array_equal(support,[0,1,2,0])
        np.testing.assert_array_equal(mapping,[0,0,1,0])
        self.assertTrue(np.all(mapping[np.array([0,1,3])]==0))


class EmbeddingModelTests(unittest.TestCase):
    def setUp(self):
        c=read_config('configs/item_embeddings.json')
        self.settings={**c['base_settings'],'embedding_dim':16,'embedding':c['embedding']}
        self.settings.update(max_epochs=2,patience=1,batch_size=8)

    def test_pooling_is_invariant_to_product_order(self):
        import torch
        torch.manual_seed(1)
        model=make_network(3,4,self.settings).eval()
        x=torch.randn(2,3)
        ids=torch.tensor([[1,2,3],[2,3,0]])
        weights=torch.tensor([[.2,.3,.5],[.4,.6,0.]])
        order=torch.tensor([2,0,1])
        with torch.no_grad():
            a=model(x,ids,weights)
            b=model(x,ids[:,order],weights[:,order])
        np.testing.assert_allclose(a.numpy(),b.numpy(),atol=1e-7,rtol=0)

    def test_embedding_gradients_padding_oov_and_checkpoint_parity(self):
        rng=np.random.default_rng(3)
        x=rng.normal(size=(24,4)).astype(np.float32)
        x[0,0]=np.nan
        ids=np.tile(np.array([1,2,3,0],dtype=np.uint32),(24,1))
        w=np.tile(np.array([.2,.3,.5,0.],dtype=np.float32),(24,1))
        mapping=np.array([0,1,0,2],dtype=np.int64)
        y=np.where(np.arange(24)%3,10.,0.)
        model=ProductJoint(self.settings,42).fit(x,ids,w,y,np.ones(24),mapping,epochs=2)
        self.assertGreater(float(model.network.embedding.weight.grad[1:].abs().sum()),0.)
        self.assertTrue(np.all(model.network.embedding.weight.detach().numpy()[0]==0))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.pt'
            model.save(path,['one','unknown','three'])
            loaded=ProductJoint.load(path)
            np.testing.assert_allclose(model.predict(x,ids,w),loaded.predict(x,ids,w),atol=1e-7,rtol=0)
        unknown=np.full_like(ids,2)
        padding=np.zeros_like(ids)
        np.testing.assert_allclose(model.predict(x,unknown,w),model.predict(x,padding,w),atol=0,rtol=0)

    def test_training_is_repeatable_with_the_same_seed(self):
        rng=np.random.default_rng(4)
        x=rng.normal(size=(16,3)).astype(np.float32)
        ids=np.tile(np.array([1,2,0],np.uint32),(16,1))
        w=np.tile(np.array([.5,.5,0.],np.float32),(16,1))
        mapping=np.array([0,1,2],np.int64)
        outputs=[]
        for _ in range(2):
            model=ProductJoint(self.settings,42).fit(x,ids,w,np.arange(16,dtype=float),np.ones(16),mapping,epochs=2)
            outputs.append(model.predict(x,ids,w))
        np.testing.assert_array_equal(*outputs)

    def test_config_retains_unconfirmed_hpo_baseline(self):
        c=read_config('configs/item_embeddings.json')
        self.assertEqual(c['base_settings']['learning_rate'],.001)
        bad=copy.deepcopy(c);bad['base_settings']['learning_rate']=.002
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'config.json';p.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError,'did not confirm'):
                read_config(p)


if __name__=='__main__':unittest.main()
