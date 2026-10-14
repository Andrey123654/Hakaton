"""Fold-excluded hierarchical host/path label counts, no held-out labels."""
from pathlib import Path
import sys,json,html,re,time,warnings
from urllib.parse import urlsplit,unquote
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from improve_v1 import best,aggregate
warnings.filterwarnings('ignore',message='Mean of empty slice')
out=ROOT/'improved_v5';out.mkdir(exist_ok=True)
tr=pd.read_csv(ROOT/'data/train.csv');g=tr.groupby('url').result.nunique();tr=tr[~tr.url.isin(g[g>1].index)].drop_duplicates('url').reset_index(drop=True);te=pd.read_csv(ROOT/'data/test.csv');y=tr.result.to_numpy();n=len(y);nt=len(te)
assert np.array_equal(y,np.load(ROOT/'improved_v1/predictions.npz')['y'])
keys=[[] for _ in range(9)]
for u in pd.concat([tr.url,te.url]):
 try:
  p=urlsplit(html.unescape(u if '://' in u else 'http://'+u));host=(p.hostname or '').lower().removeprefix('www.');parts=host.split('.');path=unquote(p.path).lower();segs=[re.sub(r'\d+','0',s) for s in path.split('/') if s];p1='/'.join(segs[:1]);p2='/'.join(segs[:2]);d2='.'.join(parts[-2:]);d3='.'.join(parts[-3:]);d4='.'.join(parts[-4:]);last=segs[-1] if segs else ''
  values=[d3,d4,host+'|'+p1,host+'|'+p2,d3+'|'+p1,d2+'|'+p1,p1,p2,last]
 except ValueError:values=['']*9
 for k,v in zip(keys,values):k.append(v)
keys=[np.array(k) for k in keys]
def extra(ref,q):return np.hstack([aggregate(k,y,ref,q) for k in keys]).astype(np.float32)
oof=np.zeros(n);test=np.zeros(nt)
for fold in range(5):
 a=np.load(ROOT/f'improved_v1/features_fold_{fold}.npz');itr=a['itr'];iva=a['iva'];ex=np.zeros((len(itr),18),np.float32)
 for aa,bb in StratifiedKFold(4,shuffle=True,random_state=100+fold).split(itr,y[itr]):ex[bb]=extra(itr[aa],itr[bb])
 X=np.hstack([a['train'],ex]);V=np.hstack([a['valid'],extra(itr,iva)]);T=np.hstack([a['test'],extra(itr,np.arange(n,n+nt))]);np.savez_compressed(out/f'hierarchy_features_fold_{fold}.npz',train=X,valid=V,test=T,itr=itr,iva=iva)
 t=time.time();m=CatBoostClassifier(depth=7,iterations=1500,learning_rate=.04,l2_leaf_reg=7,random_seed=1242+fold,thread_count=6,verbose=500,allow_writing_files=False);m.fit(X,y[itr]);m.save_model(str(out/f'hierarchy_fold_{fold}.cbm'));oof[iva]=m.predict_proba(V)[:,1];test+=m.predict_proba(T)[:,1]/5;print('fold',fold+1,time.time()-t,flush=True)
f,th=best(y,oof);np.savez(out/'hierarchy_predictions.npz',y=y,oof=oof,test=test);(out/'hierarchy_report.json').write_text(json.dumps({'oof_f1':f,'threshold':th,'validation':'5 outer folds;4 inner folds for all label counts; fixed parameters'},indent=2));print('finished',f,flush=True)
