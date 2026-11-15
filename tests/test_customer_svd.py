"""History cutoffs and train-only SVD must survive future-data poisoning."""

import runpy
import unittest

import numpy as np
import pandas as pd

helpers = runpy.run_path('scripts/run_customer_svd.py', run_name='customer_svd_helpers')


class CustomerSVDTests(unittest.TestCase):
    def test_boundaries_and_one_product_day_event(self):
        t = pd.Timestamp('2018-08-01')
        objects = pd.DataFrame({'customer_id': ['c'], 'as_of_date': [t]})
        lines = pd.DataFrame({'customer_id': ['c']*5, 'item_id': ['left', 'old', 'x', 'x', 'future'],
                              'event_ts': [t-pd.Timedelta(days=180), t-pd.Timedelta(days=181),
                                           t-pd.Timedelta(hours=24), t-pd.Timedelta(hours=23), t]})
        matrix, names = helpers['build_matrix'](objects, lines)
        row = dict(zip(names, matrix.toarray()[0]))
        self.assertEqual(row['old'], 0)
        self.assertEqual(row['future'], 0)
        self.assertAlmostEqual(row['left'], np.log1p(2**(-180/90)), places=7)
        self.assertAlmostEqual(row['x'], np.log1p(2**(-1/90)), places=7)

    def fixture(self):
        objects = pd.DataFrame({'customer_id': ['a', 'b', 'c', 'd', 'e'],
                                'as_of_date': pd.to_datetime(['2018-08-01']*4+['2018-09-01'])})
        lines = pd.DataFrame({'customer_id': ['a', 'a', 'b', 'b', 'c', 'c', 'd', 'e'],
                              'item_id': ['x', 'y', 'y', 'z', 'x', 'z', 'z', 'unknown'],
                              'event_ts': pd.to_datetime(['2018-07-01']*8)})
        return objects, lines

    def test_future_and_validation_items_do_not_enter_vocabulary(self):
        objects, lines = self.fixture()
        matrix, names = helpers['build_matrix'](objects, lines)
        first = helpers['HistorySVD']().fit(matrix, np.arange(4), names, components=2)
        self.assertNotIn('unknown', first.names)
        np.testing.assert_allclose(first.transform(matrix, [4]), 0, rtol=0, atol=0)
        poisoned = pd.concat([lines, pd.DataFrame({'customer_id': ['a'], 'item_id': ['aaa_future'],
                              'event_ts': pd.to_datetime(['2018-08-01'])})], ignore_index=True)
        other, new_names = helpers['build_matrix'](objects, poisoned)
        second = helpers['HistorySVD']().fit(other, np.arange(4), new_names, components=2)
        np.testing.assert_array_equal(first.names, second.names)
        np.testing.assert_allclose(first.transform(matrix, np.arange(5)),
                                   second.transform(other, np.arange(5)), atol=0, rtol=0)

    def test_target_values_are_not_used_by_history_encoder(self):
        objects, lines = self.fixture()
        a, _ = helpers['build_matrix'](objects.assign(spend_30d=0.), lines)
        b, _ = helpers['build_matrix'](objects.assign(spend_30d=1e12), lines)
        self.assertEqual((a != b).nnz, 0)

    def test_purge_and_paired_comparison(self):
        objects = pd.DataFrame({'target_end': pd.to_datetime(['2018-07-31', '2018-08-01', '2018-08-02'])})
        np.testing.assert_array_equal(helpers['indices'](objects, '2018-08-01'), [0, 1])
        frame = pd.DataFrame({'customer_id': ['a', 'a', 'b', 'b'],
                              'date': ['2018-09-01', '2018-10-01']*2,
                              'target_z': [0., 1., 2., 3.], 'z_control': [1., 2., 3., 4.],
                              'z_svd': [.5, 1.5, 2.5, 3.5]})
        scores = helpers['paired_scores'](frame, repeats=30)
        np.testing.assert_allclose(scores.delta, -.5)
        np.testing.assert_allclose(scores.conditional_ci_low, -.5)
        np.testing.assert_allclose(scores.conditional_ci_high, -.5)


if __name__ == '__main__':
    unittest.main()
