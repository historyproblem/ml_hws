"""Export the one fixed50/50 PROBABILITY blend from verified full-fit components.

No fit, weight search, new reserved-quarter score or automatic submission.
"""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from hw4_inspections.delivery import validate_submission

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 a=argparse.ArgumentParser(description=__doc__)
 a.add_argument('--run',type=Path,required=True);a.add_argument('--raw',type=Path)
 a.add_argument('--out',type=Path,required=True);args=a.parse_args()
 manifest=json.loads((args.run/'manifest.json').read_text());frozen=json.loads((args.run/'frozen_recipe.json').read_text())
 if manifest.get('status')!='complete' or manifest.get('smoke'):raise ValueError('Completed real full refit required')
 raw=args.raw or Path(frozen['config']['raw'])
 for name in ['test_student.csv','sample_submission.csv']:
  if sha(raw/name)!=frozen['input_sha256'][name]:raise ValueError('Test/sample SHA changed')
 test=pd.read_csv(raw/'test_student.csv');sample=pd.read_csv(raw/'sample_submission.csv');values=[];parents={}
 for name in ['context_cb','compact_cb']:
  item=manifest['submissions'][name];path=args.run/item['file']
  if sha(path)!=item['sha256']:raise ValueError('Full component CSV changed')
  frame=pd.read_csv(path,float_precision='round_trip')
  if frame.row_id.duplicated().any() or set(frame.row_id)!=set(test.row_id):raise ValueError('Component IDs differ')
  p=frame.set_index('row_id').loc[test.row_id].target.to_numpy(dtype=np.float64)
  if not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Bad probabilities')
  values.append(p);parents[item['file']]=item['sha256']
 probability=.5*values[0]+.5*values[1]
 args.out.mkdir(parents=True,exist_ok=True);file=args.out/'submission_06_context_compact_probability50.csv'
 validate_submission(test,sample,probability,expected_rows=2246).to_csv(file,index=False)
 proof={'file':file.name,'sha256':sha(file),'rows':len(test),'formula':'Equal PROBABILITIES, not percentile ranks','parent_manifest_sha256':sha(args.run/'manifest.json'),'parents':parents,'new_fits':0,'new_closed_quarter_metrics':False,'kaggle_uploads':0,'public_ap':None}
 (args.out/'probability_blend_manifest.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
if __name__=='__main__':main()
