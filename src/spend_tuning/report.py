"""Reports and a checksum-verified tar export without weights or feature caches."""

import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

from spend_baselines.suite_data import atomic_json
from spend_behavior.report import comparison


def report(root, cfg):
    from .suite import write_csv
    root = Path(root)
    frames = [pd.read_csv(p,dtype={'Id':str,'customer_id':str}) for p in sorted((root/'predictions').glob('*/*.csv.gz'))]
    if not frames:
        return
    all_rows = pd.concat(frames,ignore_index=True)
    if all_rows.duplicated(['candidate','Id']).any():
        raise ValueError('Duplicate mean3 predictions')
    scores, slices = [], []
    for name, rows in all_rows.groupby('candidate'):
        groups = [(date,g) for date,g in rows.groupby('date')]
        for label,dates in [('pooled_tuning',cfg['tuning_dates']),('pooled_confirmation',cfg['confirmation_dates']),
                            ('pooled_5_dates',cfg['tuning_dates']+cfg['confirmation_dates'])]:
            if set(dates) <= set(rows.date):
                groups.append((label,rows[rows.date.isin(dates)]))
        for date,g in groups:
            scores.append({'candidate':name,'date':date,**comparison(g,cfg['bootstrap_repeats'])})
            n=g.purchase_days_180.to_numpy()
            for label,mask in [('zero',g.target_z.to_numpy()==0),('positive',g.target_z.to_numpy()>0),
                               ('days_0',n==0),('days_1_2',(n>=1)&(n<=2)),('days_3_plus',n>=3)]:
                slices.append({'candidate':name,'date':date,'slice':label,'n':int(mask.sum()),
                               'RMSLE':float(np.sqrt(np.mean((g.z.to_numpy()[mask]-g.target_z.to_numpy()[mask])**2))) if mask.any() else None,
                               'reference_RMSLE':float(np.sqrt(np.mean((g.reference_z.to_numpy()[mask]-g.target_z.to_numpy()[mask])**2))) if mask.any() else None})
    table=pd.DataFrame(scores)
    write_csv(root/'scores.csv',table)
    write_csv(root/'error_slices.csv',pd.DataFrame(slices))
    jobs=[]
    for path in sorted((root/'jobs').glob('*/*/*/result.json')):
        r=json.loads(path.read_text())
        jobs.append({**{k:r['job'][k] for k in ('profile','date','seed')},
                     **{k:r[k] for k in ('selected_epoch','inner_date','train_seconds','inference_seconds','n_train')}})
    write_csv(root/'timing.csv',pd.DataFrame(jobs))
    lines=['# Joint + history hyperparameter study','',
           'Eight fixed profiles; seed42 screening and mean3 shortlisting use June–August only. '
           'The winner is frozen before September/October confirmation. Those dates were used in previous studies; bootstrap intervals do not correct adaptive selection or temporal drift.','',
           'Architecture 128/64, F2 + the same four causal history features, loss and data weights are unchanged. No embeddings or submission CSV.','']
    for label in ['pooled_tuning','pooled_confirmation','pooled_5_dates']:
        group=table[table.date.eq(label)].sort_values('RMSLE')
        lines.extend([f'## {label}','','| Profile | RMSLE | Delta to current Joint+history | 95% CI |','|---|---:|---:|---|'])
        for r in group.itertuples():
            lines.append(f'| {r.candidate} | {r.RMSLE:.6f} | {r.delta:+.6f} | [{r.delta_ci_low:+.6f}, {r.delta_ci_high:+.6f}] |')
        lines.append('')
    winner_path=root/'winner.json'
    if winner_path.exists():
        winner=json.loads(winner_path.read_text())
        lines.append(f"Frozen early-data winner: `{winner['profile']}`. This does not automatically replace the current best model.")
        selected=table[table.candidate.eq(winner['profile']) & table.date.eq('pooled_confirmation')]
        if len(selected):
            r=selected.iloc[0]
            verdict=('improved' if r.delta_ci_high<0 else 'worse' if r.delta_ci_low>0 else 'inconclusive')
            atomic_json(root/'confirmation.json',{'profile':winner['profile'],'verdict':verdict,
                                                  **{k:float(r[k]) for k in ['RMSLE','reference_RMSLE','delta','delta_ci_low','delta_ci_high']},
                                                  'baseline_reused':True,'test_quality_observed':False})
            lines.append(f'Confirmation: {verdict}. November test quality remains unobserved.')
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


def export(root, output):
    """Snapshot complete jobs only. Original weight hashes remain in metadata."""
    root,output=Path(root),Path(output)
    if not (root/'manifest.json').exists():
        raise FileNotFoundError('No tuning manifest')
    if output.exists():
        raise FileExistsError(f'Choose a new archive path: {output}')
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_name(output.name+'.tmp')
    receipt={'format_version':1,'completed_jobs':[],'unfinished_jobs':[],'invalid_jobs':[],
             'sha256':{},'weights_omitted':True,'cache_omitted':True,
             'summary_files':'advisory live snapshots; recompute scores from committed predictions'}
    with tarfile.open(temporary,'w:gz') as archive:
        def add(name,payload):
            member=tarfile.TarInfo(name)
            member.size=len(payload)
            archive.addfile(member,io.BytesIO(payload))
            receipt['sha256'][name]=hashlib.sha256(payload).hexdigest()
        for name in ['manifest.json','status.json','shortlist.json','winner.json','confirmation.json','screen_scores.csv',
                     'scores.csv','error_slices.csv','timing.csv','REPORT.md']:
            if (root/name).is_file():
                add(name,(root/name).read_bytes())
        for parent in ['jobs','refit_jobs']:
            for directory in sorted((root/parent).glob('*/*/*')):
                if not directory.is_dir():
                    continue
                key=str(directory.relative_to(root))
                marker=directory/'result.json'
                if not marker.exists():
                    receipt['unfinished_jobs'].append(key)
                    continue
                try:
                    raw=marker.read_bytes()
                    state=json.loads(raw)
                    files={'result.json':raw}
                    for name,sha in state['sha256'].items():
                        if Path(name).name != name:
                            raise ValueError('Unexpected artifact path')
                        if name.endswith('.pt'):
                            continue
                        payload=(directory/name).read_bytes()
                        if hashlib.sha256(payload).hexdigest()!=sha:
                            raise ValueError(f'Corrupt artifact {name}')
                        files[name]=payload
                except (OSError,ValueError,KeyError,TypeError) as error:
                    receipt['invalid_jobs'].append({'job':key,'error':str(error)})
                    continue
                for name,payload in files.items():
                    add(key+'/'+name,payload)
                receipt['completed_jobs'].append(key)
        for path in sorted((root/'predictions').glob('*/*.csv.gz')):
            add(str(path.relative_to(root)),path.read_bytes())
        payload=json.dumps(receipt,indent=2,sort_keys=True).encode()
        member=tarfile.TarInfo('export_manifest.json')
        member.size=len(payload)
        archive.addfile(member,io.BytesIO(payload))
    temporary.replace(output)
    return {'archive':str(output),'completed_jobs':len(receipt['completed_jobs']),
            'unfinished_jobs':len(receipt['unfinished_jobs']),'invalid_jobs':len(receipt['invalid_jobs']),
            'files':len(receipt['sha256'])}
