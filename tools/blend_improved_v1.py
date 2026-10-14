"""Blend new independent model with frozen previous ensemble; keep diagnostics."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
sys.path.insert(0,str(Path(__file__).resolve().parent))
from improve_v1 import best
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'improved_v1'
d=np.load(out/'predictions.npz');y=d['y'];new=d['oof'];newte=d['test']
pairs=[('oof_final_final.npy','test_proba_final.npy',.2),('oof_final_nn2.npy','test_proba_nn2.npy',.2),('meta_oof_nn.npy','meta_proba_test_nn.npy',.3),('meta_oof_nn2.npy','meta_proba_test_nn2.npy',.3)]
old=sum(w*rankdata(np.load(ROOT/'models'/a))/len(y) for a,b,w in pairs)
oldte=sum(w*rankdata(np.load(ROOT/'models'/b))/len(newte) for a,b,w in pairs)
nr=rankdata(new)/len(y);nrt=rankdata(newte)/len(newte)
rows=[]
splits=list(StratifiedKFold(5,shuffle=True,random_state=42).split(y,y))
for weight in [0,.25,.5,.75,1]:
    p=weight*nr+(1-weight)*old;te=weight*nrt+(1-weight)*oldte
    f,t=best(y,p);fixed=np.zeros(len(y),dtype=int)
    for a,b in splits:
        _,th=best(y[a],p[a]);fixed[b]=(p[b]>=th)
    rows.append({'new_weight':weight,'optimized_oof_f1':f,'crossfit_threshold_f1':f1_score(y,fixed),'threshold':t})
# Coarse, predeclared candidates; select by cross-fitted threshold performance.
selected=max(rows,key=lambda r:r['crossfit_threshold_f1']);w=selected['new_weight']
p=w*nr+(1-w)*old;pt=w*nrt+(1-w)*oldte
sub=pd.read_csv(ROOT/'data/sample.csv');sub['Predicted']=(pt>=selected['threshold']).astype(int)
sub.to_csv(out/'submission_blend_improved.csv',index=False)
conservative=next(r for r in rows if r['new_weight']==.25)
backup=pd.read_csv(ROOT/'data/sample.csv')
backup['Predicted']=(.25*nrt+.75*oldte>=conservative['threshold']).astype(int)
backup.to_csv(out/'submission_blend_conservative.csv',index=False)
np.savez(out/'blend_predictions.npz',oof=p,test=pt)
result={'selected':selected,'candidates':rows,'test_differences_vs_blend4':int((sub.Predicted!=pd.read_csv(ROOT/'submission_blend4.csv').Predicted).sum()),'caveat':'Existing base/meta OOF arrays reused; this blend is exploratory, not fully nested independent validation. Public score unknown.'}
(out/'blend_report.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
