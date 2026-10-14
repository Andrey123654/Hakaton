"""Coarse OOF blend selection; no public-label feedback or manual test edits."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'));from improve_v1 import best
out=ROOT/'improved_v3';base=np.load(ROOT/'improved_v2/blend_predictions.npz');y=base['y'];n=len(y);nt=len(base['test'])
names=['baseline'];rs=[base['oof']];ts=[base['test']]
for name in ['shallow','deep','words']:
 d=np.load(out/f'{name}_predictions.npz');assert np.array_equal(y,d['y']);names.append(name);rs.append(rankdata(d['oof'])/n);ts.append(rankdata(d['test'])/nt)
R=np.column_stack(rs);T=np.column_stack(ts)
weights=[(1,0,0,0)]
for w in [.1,.2,.35,.5]:
 for k in range(1,4):
  a=np.zeros(4);a[0]=1-w;a[k]=w;weights.append(tuple(a))
 for pair in [(1,2),(1,3),(2,3)]:
  a=np.zeros(4);a[0]=1-w;a[list(pair)]=w/2;weights.append(tuple(a))
 a=np.array([1-w,w/3,w/3,w/3]);weights.append(tuple(a))
splits=list(StratifiedKFold(5,shuffle=True,random_state=42).split(y,y));rows=[];scores=[]
for w in weights:
 p=R@w;f,th=best(y,p);pr=np.zeros(n,dtype=int);folds=[]
 for a,b in splits:
  _,t=best(y[a],p[a]);pr[b]=p[b]>=t;folds.append(float(f1_score(y[b],pr[b])))
 rows.append({'weights':list(w),'oof_f1':f,'crossfit_f1':float(f1_score(y,pr)),'threshold':th,'fold_f1':folds});scores.append(p)
order=sorted(range(len(rows)),key=lambda j:rows[j]['crossfit_f1'],reverse=True);choice=order[0]
nested=np.zeros(n,dtype=int);nw=[]
for a,b in splits:
 local=[best(y[a],p[a]) for p in scores];j=max(range(len(local)),key=lambda j:local[j][0]);nested[b]=scores[j][b]>=local[j][1];nw.append(rows[j]['weights'])
tr=pd.read_csv(ROOT/'data/train.csv');g=tr.groupby('url').result.nunique();tr=tr[~tr.url.isin(g[g>1].index)].drop_duplicates('url');known=pd.read_csv(ROOT/'data/test.csv').url.map(tr.set_index('url').result);from url_identity_v3 import lookup
known=lookup(tr,pd.read_csv(ROOT/'data/test.csv'));mask=known.notna().to_numpy()
old=pd.read_csv(ROOT/'improved_v2/submission_v2.csv').Predicted.to_numpy();prev=base['oof']>=rows[0]['threshold'];reports=[]
# First and a distinct second candidate, both locally above baseline.
chosen=[choice]
for j in order[1:]:
 if rows[j]['crossfit_f1']<=rows[0]['crossfit_f1']:continue
 pred=T@np.array(rows[j]['weights'])>=rows[j]['threshold'];main=T@np.array(rows[choice]['weights'])>=rows[choice]['threshold']
 if np.sum(pred!=main)>=5:chosen.append(j);break
for k,j in enumerate(chosen):
 r=rows[j];pred=(T@r['weights']>=r['threshold']).astype(int);pred[mask]=known[mask].astype(int);s=pd.read_csv(ROOT/'data/sample.csv');s['Predicted']=pred;filename='submission_v3.csv' if k==0 else 'submission_v3_alternative.csv';s.to_csv(out/filename,index=False)
 new=scores[j]>=r['threshold'];reports.append({**r,'file':filename,'test_changes':int(np.sum(pred!=old)),'corrected_errors':int(np.sum((prev!=y)&(new==y))),'introduced_errors':int(np.sum((prev==y)&(new!=y)))})
np.savez(out/'blend_predictions.npz',y=y,oof=scores[choice],test=T@rows[choice]['weights'])
report={'sources':names,'baseline':rows[0],'selected':reports,'nested_selection_f1':float(f1_score(y,nested)),'nested_weights':nw,'candidates':rows,'canonical_test_matches':int(mask.sum()),'baseline_public_f1':.994747,'new_public_f1':None,'caveat':'OOF stacking evaluation; upstream models not fully nested retrained.'};(out/'blend_report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:report[k] for k in ['baseline','selected','nested_selection_f1']},indent=2))
