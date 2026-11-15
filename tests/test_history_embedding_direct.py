import runpy,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from spend_embeddings.direct import ProductMLP,make_network

settings=runpy.run_path('scripts/run_history_embedding_direct.py',run_name='direct_helpers')['SETTINGS']


class DirectHistoryTests(unittest.TestCase):
    def test_history_vector_is_linear_weighted_sum_and_output_is_scalar(self):
        s={**settings,'hidden_sizes':[],'embedding_dim':2}
        net=make_network(1,3,s)
        with torch.no_grad():
            net.embedding.weight.copy_(torch.tensor([[0.,0.],[2.,4.],[6.,8.]]))
            net.output.weight.copy_(torch.tensor([[0.,1.,1.]]));net.output.bias.zero_()
        raw=net(torch.zeros(1,1),torch.tensor([[1,2,0]]),torch.tensor([[.25,.75,0.]]))
        self.assertEqual(tuple(raw.shape),(1,1));self.assertEqual(float(raw.detach().item()),12.)

    def test_direct_training_and_checkpoint_reproduce_without_probability_head(self):
        x=np.arange(16,dtype=np.float32).reshape(8,2)/16
        ids=np.tile(np.array([1,2,0]),(8,1));weights=np.tile(np.array([.4,.6,0.],dtype=np.float32),(8,1))
        y=np.arange(8,dtype=float);mapping=np.arange(3)
        s={**settings,'hidden_sizes':[4],'max_epochs':2,'batch_size':4}
        model=ProductMLP(s,42,'cpu',2).fit(x,ids,weights,y,np.ones(8),mapping,epochs=2)
        self.assertEqual(model.network.output.out_features,1)
        np.testing.assert_array_equal(model.network.embedding.weight[0].detach().numpy(),np.zeros(32))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'model.pt';model.save(p,np.array(['a','b']))
            loaded=ProductMLP.load(p)
            np.testing.assert_array_equal(model.predict(x,ids,weights),loaded.predict(x,ids,weights))


if __name__=='__main__':unittest.main()
