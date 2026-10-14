"""Coarse selection of independent new models, preserve confirmed baseline."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'));from improve_v1 import best
out=ROOT/'improved_v5';base=np.load(ROOT/'improved_v2/blend_predictions.npz');y=base['y'];n=len(y);nt=len(base['test'])
names=['baseline','hierarchy','nbsvm'];rs=[base['oof']];ts=[base['test']]
for name in names[1:]:
 a=np.load(out/f'{name}_predictions.npz');assert np.array_equal(y,a['y']);rs.append(rankdata(a['oof'])/n);ts.append(rankdata(a['test'])/nt)
R=np.column_stack(rs);T=np.column_stack(ts);weights=[(1.,0.,0.)]
for w in [.05,.1,.2,.35,.5]:weights.extend([(1-w,w,0),(1-w,0,w),(1-w,w/2,w/2)])
splits=list(StratifiedKFold(5,shuffle=True,random_state=42).split(y,y));rows=[];scores=[]
for w in weights:
 p=R@w;f,th=best(y,p);pr=np.zeros(n,int);fs=[]
 for a,b in splits:_,t=best(y[a],p[a]);pr[b]=p[b]>=t;fs.append(float(f1_score(y[b],pr[b])))
 rows.append({'weights':list(w),'oof_f1':f,'crossfit_f1':float(f1_score(y,pr)),'threshold':th,'fold_f1':fs});scores.append(p)
order=sorted(range(len(rows)),key=lambda j:rows[j]['crossfit_f1'],reverse=True);choice=order[0]
nested=np.zeros(n,int)
for a,b in splits:
 local=[best(y[a],p[a]) for p in scores];j=max(range(len(local)),key=lambda j:local[j][0]);nested[b]=scores[j][b]>=local[j][1]
# Exact raw training URL identities only; no broadened canonical overrides.
tr=pd.read_csv(ROOT/'data/train.csv');g=tr.groupby('url').result.nunique();tr=tr[~tr.url.isin(g[g>1].index)].drop_duplicates('url');known=pd.read_csv(ROOT/'data/test.csv').url.map(tr.set_index('url').result);mask=known.notna().to_numpy()
old=pd.read_csv(ROOT/'improved_v2/submission_v2.csv').Predicted.to_numpy();selected=[]
for j in order[:2]:
 if rows[j]['crossfit_f1']<=rows[0]['crossfit_f1']:continue
 r=rows[j];pred=(T@r['weights']>=r['threshold']).astype(int);pred[mask]=known[mask].astype(int);changes=int(np.sum(pred!=old))
 if changes==0:continue
 if selected and np.array_equal(pred,pd.read_csv(out/selected[0]['file']).Predicted):continue
 filename='submission_v5.csv' if not selected else 'submission_v5_alternative.csv';s=pd.read_csv(ROOT/'data/sample.csv');s['Predicted']=pred;s.to_csv(out/filename,index=False);selected.append({**r,'file':filename,'test_changes':changes})
report={'sources':names,'baseline':rows[0],'selected':selected,'nested_selection_f1':float(f1_score(y,nested)),'candidates':rows,'baseline_public_f1':.994747,'new_public_f1':None,'caveat':'Prior baseline OOF reused; new models independently exclude validation labels. OOF blend selection not fully nested upstream retraining.'};(out/'blend_report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:report[k] for k in ['baseline','selected','nested_selection_f1']},indent=2))
