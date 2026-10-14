"""Independent boosting algorithms on frozen leakage-controlled V1 features."""
from pathlib import Path
import os,sys,json,time
ROOT=Path(__file__).resolve().parents[1]
# Load the already installed OpenMP runtime before optional boosting engines.
import ctypes
ctypes.CDLL(str(ROOT/'.venv/lib/python3.9/site-packages/sklearn/.dylibs/libomp.dylib'),mode=ctypes.RTLD_GLOBAL)
import numpy as np,pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import joblib
sys.path.insert(0,str(ROOT/'tools'));from improve_v1 import best
out=ROOT/'improved_v10';out.mkdir(exist_ok=True);base=np.load(ROOT/'improved_v2/blend_predictions.npz');y=base['y'];nt=len(base['test']);n=len(y)
for name in ['lightgbm','xgboost']:
 oof=np.zeros(n);test=np.zeros(nt)
 for fold in range(5):
  a=np.load(ROOT/f'improved_v1/features_fold_{fold}.npz');path=out/f'{name}_fold_{fold}.joblib';start=time.time()
  if path.exists():m=joblib.load(path)
  else:
   if name=='lightgbm':m=LGBMClassifier(n_estimators=1200,learning_rate=.035,num_leaves=31,max_depth=-1,min_child_samples=50,reg_lambda=8,reg_alpha=.1,colsample_bytree=.85,subsample=.85,subsample_freq=1,n_jobs=6,random_state=3042+fold,verbosity=-1)
   else:m=XGBClassifier(n_estimators=900,learning_rate=.04,max_depth=5,min_child_weight=5,reg_lambda=10,reg_alpha=.1,colsample_bytree=.9,subsample=.9,n_jobs=6,random_state=4042+fold,tree_method='hist',eval_metric='logloss')
   m.fit(a['train'],y[a['itr']]);joblib.dump(m,path,compress=3)
  oof[a['iva']]=m.predict_proba(a['valid'])[:,1];test+=m.predict_proba(a['test'])[:,1]/5;print(name,'fold',fold+1,'seconds',round(time.time()-start,1),flush=True)
 np.savez(out/f'{name}_predictions.npz',y=y,oof=oof,test=test);print(name,'standalone',best(y,oof),flush=True)
names=['baseline','lightgbm','xgboost'];R=[base['oof']];T=[base['test']]
for name in names[1:]:a=np.load(out/f'{name}_predictions.npz');R.append(rankdata(a['oof'])/n);T.append(rankdata(a['test'])/nt)
R=np.column_stack(R);T=np.column_stack(T);weights=[(1.,0.,0.)]
for w in [.05,.1,.2,.35]:weights.extend([(1-w,w,0),(1-w,0,w),(1-w,w/2,w/2)])
splits=list(StratifiedKFold(5,shuffle=True,random_state=42).split(y,y));rows=[];scores=[]
for w in weights:
 p=R@w;f,th=best(y,p);pr=np.zeros(n,int)
 for a,b in splits:_,t=best(y[a],p[a]);pr[b]=p[b]>=t
 rows.append({'weights':list(w),'oof_f1':f,'crossfit_f1':float(f1_score(y,pr)),'threshold':th});scores.append(p)
choice=max(rows,key=lambda r:r['crossfit_f1']);nested=np.zeros(n,int)
for a,b in splits:
 local=[best(y[a],p[a]) for p in scores];j=max(range(len(local)),key=lambda j:local[j][0]);nested[b]=scores[j][b]>=local[j][1]
report={'sources':names,'baseline':rows[0],'candidates':rows,'selected':choice if choice['weights']!=[1.,0.,0.] else None,'nested_weight_threshold_f1':float(f1_score(y,nested)),'public_f1':None,'caveat':'New boosting models independently validated. Prior baseline OOF reused; blend selection not full nested upstream retraining. Final V5 full-training component has no independent OOF.'};(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
if report['selected']:
 tree=np.load(ROOT/'improved_v5/full_predictions.npz')['test'];v5=.9*base['test']+.1*rankdata(tree)/nt;w=choice['weights'];pred=(w[0]*v5+w[1]*T[:,1]+w[2]*T[:,2]>=choice['threshold']).astype(int);s=pd.read_csv(ROOT/'data/sample.csv');s['Predicted']=pred;s.to_csv(out/'submission_v10.csv',index=False);old=pd.read_csv(ROOT/'improved_v5/submission_v5_refit.csv').Predicted.to_numpy();ix=np.flatnonzero(pred!=old);te=pd.read_csv(ROOT/'data/test.csv');pd.DataFrame({'Id':ix,'url':te.url.iloc[ix].to_numpy(),'V5':old[ix],'V10':pred[ix]}).to_csv(out/'changes.csv',index=False);report['changes_vs_v5']=len(ix);(out/'report.json').write_text(json.dumps(report,indent=2));print('changes',len(ix),flush=True)
