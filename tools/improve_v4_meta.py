"""Cross-fitted OOF stacking; upstream retraining is not fully nested."""
from pathlib import Path
import sys,json
import numpy as np,joblib
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools')];from improve_v1 import best
out=ROOT/'improved_v4';out.mkdir(exist_ok=True);y=np.load(ROOT/'improved_v1/predictions.npz')['y'];rs=[];ts=[];names=[]
for tag in ['nn','nn2','nn3']:
 a=np.load(ROOT/f'models/oof_{tag}.npz');b=np.load(ROOT/f'models/test_base_{tag}.npz');assert np.array_equal(y,a['y'])
 for k in a.files:
  if k=='y':continue
  rs.append(logit(np.clip(a[k],1e-6,1-1e-6)));ts.append(logit(np.clip(b[k],1e-6,1-1e-6)));names.append(tag+':'+k)
for folder,file in [('improved_v1','predictions'),('improved_v2','linear_predictions'),('improved_v2','resources_predictions'),('improved_v3','shallow_predictions'),('improved_v3','deep_predictions'),('improved_v3','words_predictions')]:
 a=np.load(ROOT/f'{folder}/{file}.npz');assert np.array_equal(y,a['y']);
 if file.startswith('linear') or file.startswith('resources'):r=a['oof'];t=a['test']
 else:r=logit(np.clip(a['oof'],1e-6,1-1e-6));t=logit(np.clip(a['test'],1e-6,1-1e-6))
 rs.append(r);ts.append(t);names.append(folder+':'+file)
a=np.load(ROOT/'improved_v2/blend_predictions.npz');rs.append(logit(np.clip(a['oof'],1e-6,1-1e-6)));ts.append(logit(np.clip(a['test'],1e-6,1-1e-6)));names.append('baseline_rank_blend')
X=np.column_stack(rs);T=np.column_stack(ts);np.savez(out/'meta_features.npz',train=X,test=T,y=y);(out/'meta_columns.json').write_text(json.dumps(names,indent=2))
for name,params in [('meta_lr',None),('meta_tree',dict(iterations=650,depth=4,learning_rate=.025,l2_leaf_reg=30,random_seed=812,thread_count=4,verbose=False,allow_writing_files=False))]:
 oof=np.zeros(len(y));test=np.zeros(16000)
 for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(y,y)):
  if params is None:
   sc=StandardScaler().fit(X[a]);m=LogisticRegression(C=.3,max_iter=2000);m.fit(sc.transform(X[a]),y[a]);oof[b]=m.predict_proba(sc.transform(X[b]))[:,1];test+=m.predict_proba(sc.transform(T))[:,1]/5;joblib.dump((sc,m),out/f'{name}_fold_{fold}.joblib',compress=3)
  else:
   m=CatBoostClassifier(**params);m.fit(X[a],y[a]);oof[b]=m.predict_proba(X[b])[:,1];test+=m.predict_proba(T)[:,1]/5;m.save_model(str(out/f'{name}_fold_{fold}.cbm'))
  print(name,fold+1,flush=True)
 f,th=best(y,oof);np.savez(out/f'{name}_predictions.npz',oof=oof,test=test,y=y);(out/f'{name}_report.json').write_text(json.dumps({'oof_f1':f,'threshold':th,'caveat':'OOF stacking; upstream model retraining not fully nested'},indent=2));print(name,f,flush=True)
