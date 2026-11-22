"""Boundary, unknown-first, incomplete-quarter and target-free own summaries."""
import numpy as np
import pandas as pd

from hw4_inspections.own_first_quarter import build_own_first_quarter


def fixture():
    base=dict(forecast_period='2024-01-01',facility_type='restaurant',risk_level='high',area_id='a')
    rows=pd.DataFrame([dict(base,site_id='own',history_before='2023-12-25',target=1)])
    data=[]
    for site,week,result,codes in [
        ('peer','2019-12-30','pass',''),('own','2022-04-11','fail','1 2 3'),
        ('own','2022-05-16','pass','1'),('own','2022-07-04','unknown',''),
        ('own','2022-08-01','fail','1 2'),('own','2022-09-26','fail','1'),
        ('own','2022-10-10','pass',''),('own','2023-01-02','pass',''),
        ('own','2023-04-03','fail','1 2'),('own','2023-07-03','conditional','2 3'),
        ('own','2023-10-02','fail','1 2 3'),('own','2023-12-25','fail','1 2 3')]:
        data.append(dict(base,site_id=site,inspection_week=week,inspection_type='routine',result=result,violation_codes=codes))
    return rows,pd.DataFrame(data)


def test_own_firstq_excludes_unknown_boundary_and_incomplete_without_substitution():
    rows,h=fixture();x=build_own_first_quarter(rows,h).iloc[0]
    assert x.own_firstq_n==4 and x.own_firstq_fail_n==2 and x.own_firstq_fail_mean==.5
    assert x.own_firstq_last_fail==0 and x.own_firstq_last_load==2
    assert x.own_firstq_last2_fail_mean==x.own_firstq_last4_fail_mean==.5
    assert x.own_firstq_same_season_n==1 and x.own_firstq_same_season_fail_mean==0


def test_own_firstq_target_and_future_and_peer_outcome_invariance():
    rows,h=fixture();original=build_own_first_quarter(rows,h)
    modified=h.copy();modified.loc[modified.site_id.eq('peer'),'result']='fail'
    modified.loc[modified.inspection_week.eq('2023-12-25'),'violation_codes']='44'
    changed=build_own_first_quarter(rows.assign(target=0),modified)
    pd.testing.assert_frame_equal(original,changed)
    clipped=h.loc[h.inspection_week.ge('2023-04-03')]
    low=build_own_first_quarter(rows,clipped).iloc[0]
    assert low.own_firstq_n==1 and low.own_firstq_same_season_n==0
