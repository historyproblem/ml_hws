"""Identical-population mean3 and fixed-blend comparisons to Joint+history."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from spend_behavior.report import comparison
from spend_tuning.suite import write_csv


def report(root,cfg):
    root=Path(root)
    paths=sorted((root/'predictions').glob('*/*.csv.gz'))
    original=[pd.read_csv(p,dtype={'Id':str,'customer_id':str}) for p in paths if not p.parent.name.startswith('blend_')]
    if not original:return
    frames=list(original)
    for a in original:
        if a.candidate.eq('baseline').all():continue
        for weight in cfg['blend_weights']:
            f=a.copy()
            name=f"blend_{a.candidate.iloc[0]}_{int(round(weight*100))}pct"
            f['candidate']=name
            f['z']=weight*a.z+(1-weight)*a.reference_z
            f['delta_squared_log_error']=(f.z-f.target_z)**2-(f.reference_z-f.target_z)**2
            write_csv(root/'predictions'/name/f'{a.date.iloc[0]}.csv.gz',f)
            frames.append(f)
    rows=pd.concat(frames,ignore_index=True)
    if rows.duplicated(['candidate','Id']).any():raise ValueError('Duplicate embedding predictions')
    scores,slices=[],[]
    for name,frame in rows.groupby('candidate'):
        groups=list(frame.groupby('date'))
        late=['2018-09-01','2018-10-01']
        for label,dates in [('pooled_SepOct',late),('pooled_5_dates',cfg['validation_dates'])]:
            if set(dates)<=set(frame.date):groups.append((label,frame[frame.date.isin(dates)]))
        for date,g in groups:
            scores.append({'candidate':name,'date':date,**comparison(g,cfg['bootstrap_repeats'])})
            n=g.purchase_days_180.to_numpy()
            for label,mask in [('zero',g.target_z.to_numpy()==0),('positive',g.target_z.to_numpy()>0),
                               ('days_0',n==0),('days_1_2',(n>=1)&(n<=2)),('days_3_plus',n>=3)]:
                slices.append({'candidate':name,'date':date,'slice':label,'n':int(mask.sum()),
                               'RMSLE':float(np.sqrt(np.mean((g.z.to_numpy()[mask]-g.target_z.to_numpy()[mask])**2))) if mask.any() else None})
    table=pd.DataFrame(scores)
    write_csv(root/'scores.csv',table)
    write_csv(root/'error_slices.csv',pd.DataFrame(slices))
    timing=[]
    for path in sorted((root/'jobs').glob('*/*/*/result.json')):
        r=json.loads(path.read_text())
        timing.append({**{k:r['job'][k] for k in ['profile','date','seed']},
                       **{k:r[k] for k in ['train_seconds','inference_seconds','vocabulary_items','known_weight_fraction','mean_retained_topk_mass']}})
    write_csv(root/'timing.csv',pd.DataFrame(timing))
    lines=['# Product embeddings + Joint history','',
           'Control: the frozen lr=0.001 Joint+history mean3. lr=0.002 did not improve late confirmation in the preceding HP study.','',
           'Vocabulary is fitted only on purged training bags; support counts unique customers. Unknown/rare products have fixed zero embeddings. Product bags use positive purchase days strictly before each row cutoff.','',
           'Customer-cluster bootstrap intervals are conditional on reused dates, without adaptive-selection or drift corrections. November quality is unobserved.','',
           '| Candidate | Sep–Oct RMSLE | Delta | 95% CI |','|---|---:|---:|---|']
    for r in table[table.date.eq('pooled_SepOct')].sort_values('RMSLE').itertuples():
        lines.append(f'| {r.candidate} | {r.RMSLE:.6f} | {r.delta:+.6f} | [{r.delta_ci_low:+.6f}, {r.delta_ci_high:+.6f}] |')
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
