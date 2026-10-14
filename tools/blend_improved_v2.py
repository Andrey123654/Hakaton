"""Evaluate coarse, reproducible mixtures; never use tournament feedback to tune."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score,confusion_matrix
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'));from improve_v1 import best
out=ROOT/'improved_v2'
datasets=[np.load(ROOT/'improved_v1/blend_predictions.npz'),np.load(out/'linear_predictions.npz'),np.load(out/'resources_predictions.npz')]
y=np.load(out/'linear_predictions.npz')['y']
assert np.array_equal(y,np.load(out/'resources_predictions.npz')['y'])
# Current baseline is already a rank mixture; preserve its scale.
R=np.column_stack([datasets[0]['oof'],rankdata(datasets[1]['oof'])/len(y),rankdata(datasets[2]['oof'])/len(y)])
nt=len(datasets[0]['test'])
T=np.column_stack([datasets[0]['test'],rankdata(datasets[1]['test'])/nt,rankdata(datasets[2]['test'])/nt])
assert R.shape==(len(y),3) and T.shape==(nt,3)
weights=[(1.,0.,0.)]
for remaining in [.1,.2,.35,.5]:
    for frac in [0.,.5,1.]:weights.append((1-remaining,remaining*frac,remaining*(1-frac)))
splits=list(StratifiedKFold(5,shuffle=True,random_state=42).split(y,y));rows=[];scores=[]
for w in weights:
    p=R@np.array(w);f,th=best(y,p);pred=np.zeros(len(y),dtype=int);folds=[]
    for a,b in splits:
        _,t=best(y[a],p[a]);pred[b]=p[b]>=t;folds.append(float(f1_score(y[b],pred[b])))
    rows.append({'weights':list(w),'optimized_oof_f1':f,'crossfit_threshold_f1':float(f1_score(y,pred)),'threshold':th,'fold_f1':folds})
    scores.append(p)
choice=int(np.argmax([r['crossfit_threshold_f1'] for r in rows]));selected=rows[choice]
# Outer selection of both weights and threshold estimates tuning sensitivity.
nested=np.zeros(len(y),dtype=int);nested_weights=[]
for a,b in splits:
    local=[best(y[a],p[a]) for p in scores];j=max(range(len(local)),key=lambda i:local[i][0])
    nested[b]=scores[j][b]>=local[j][1];nested_weights.append(rows[j]['weights'])
pt=T@np.array(selected['weights']);pred=pt>=selected['threshold']
tr=pd.read_csv(ROOT/'data/train.csv');g=tr.groupby('url').result.nunique()
tr=tr[~tr.url.isin(g[g>1].index)].drop_duplicates('url')
known=pd.read_csv(ROOT/'data/test.csv').url.map(tr.set_index('url').result)
known_mask=known.notna().to_numpy();identity_corrections=int(np.sum(pred[known_mask]!=known[known_mask].to_numpy()))
pred[known_mask]=known[known_mask].to_numpy().astype(bool)
sub=pd.read_csv(ROOT/'data/sample.csv');assert len(sub)==nt;sub['Predicted']=pred.astype(int)
sub.to_csv(out/'submission_v2.csv',index=False)
# Conservative recipe: 80% current best, 10% each independent new signal.
conservative=next(r for r in rows if np.allclose(r['weights'],[.8,.2,0]))
backup=pd.read_csv(ROOT/'data/sample.csv');backup['Predicted']=(T@np.array(conservative['weights'])>=conservative['threshold']).astype(int)
backup.loc[known_mask,'Predicted']=known[known_mask].to_numpy().astype(int)
backup.to_csv(out/'submission_v2_conservative.csv',index=False)
np.savez(out/'blend_predictions.npz',oof=scores[choice],test=pt,y=y)
oldtest=pd.read_csv(ROOT/'improved_v1/submission_blend_improved.csv').Predicted.values
oldpred=R[:,0]>=rows[0]['threshold'];newpred=scores[choice]>=selected['threshold'];wrong=oldpred!=y
report={'sources':['current_best','text_char_word_svm','resource_tail_svm'],'selected':selected,'baseline':rows[0],'candidates':rows,'nested_weight_and_threshold_f1':float(f1_score(y,nested)),'nested_fold_weights':nested_weights,'corrected_baseline_errors':int(np.sum(wrong & (newpred==y))),'introduced_errors':int(np.sum(~wrong & (newpred!=y))),'test_changes':int(np.sum(pred!=oldtest)),'conservative_test_changes':int(np.sum(backup.Predicted.values!=oldtest)),'public_f1':None,'caveat':'Standard OOF stacking with reused baseline, not full nested retraining of all upstream models; tournament improvement unverified.'}
report['exact_train_url_matches']=int(known_mask.sum());report['identity_corrections']=identity_corrections
(out/'blend_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
