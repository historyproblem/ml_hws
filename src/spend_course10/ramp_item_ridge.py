"""A single Ridge combining the two separately regularised input blocks."""
from scipy import sparse
from .ramp_ridge import RampDesign
from .sparse_item_ridge import SparseItemDesign


class RampItemDesign(SparseItemDesign):
    def __init__(self,feature_names,min_customers=10,bag_scale=.5):
        super().__init__(min_customers,bag_scale)
        self.feature_names=list(feature_names)

    def fit(self,x,histories,customers,item_names):
        super().fit(x,histories,customers,item_names)
        self.ramps=RampDesign(interactions=True).fit(x,self.feature_names)
        return self

    def transform(self,x,histories):
        original=super().transform(x,histories)
        tabular_width=len(self.preprocessor.state()['mean'])
        return sparse.hstack([sparse.csr_matrix(self.ramps.transform(x)),original[:,tabular_width:]],format='csr')
