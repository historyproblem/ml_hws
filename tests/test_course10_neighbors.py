import runpy
import unittest
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

h=runpy.run_path('scripts/course10_neighbors.py',run_name='neighbor_helpers')


class NeighborTests(unittest.TestCase):
    def test_shared_query_matches_single_knn_regression_with_exact_duplicates(self):
        x=np.array([[0.],[0.],[1.],[2.],[5.]])
        y=np.array([2.,6.,10.,12.,20.]);test=np.array([[0.],[.3],[3.]])
        for k in [2,4]:
            for w in ['uniform','distance']:
                model=KNeighborsRegressor(n_neighbors=k,weights=w,algorithm='brute').fit(x,y)
                d,i=model.kneighbors(test)
                actual=h['neighbor_average'](d,i,y,k,w)
                np.testing.assert_allclose(actual,model.predict(test),rtol=1e-14,atol=1e-14)


if __name__=='__main__':unittest.main()
