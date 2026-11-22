"""Target-free rank invariance and explicit unchanged-own-feature checks."""
import numpy as np
import pandas as pd

from hw4_inspections.quarter_context_rank import quarter_rank_context


def test_context_rank_independent_quarter_levels_and_missingness():
    values=np.array([10.,20.,np.nan,1000.,2000.,np.nan])
    frame=pd.DataFrame({'own':np.arange(6)})
    for i in range(72):frame[f'ctx_{i}']=values
    periods=pd.Series(['2023-01-01']*3+['2023-04-01']*3)
    ranked=quarter_rank_context(frame,periods)
    assert np.array_equal(ranked.own,frame.own)
    for i in range(72):
        np.testing.assert_array_equal(ranked[f'ctx_{i}'].to_numpy(),[.5,1.,np.nan,.5,1.,np.nan])
    order=np.array([5,1,3,0,4,2])
    shuffled=quarter_rank_context(frame.iloc[order].reset_index(drop=True),periods.iloc[order].reset_index(drop=True))
    pd.testing.assert_frame_equal(shuffled.iloc[np.argsort(order)].reset_index(drop=True),ranked)


def test_context_rank_rejects_raw_identity_or_misalignment():
    frame=pd.DataFrame({f'ctx_{i}':[1.,2.]for i in range(72)})
    for bad,period in [(frame.assign(target=[0,1]),['a','a']),(frame,['a'])]:
        try:
            quarter_rank_context(bad,period)
            raise AssertionError('Invalid metadata leaked into rank transform')
        except ValueError:
            pass
