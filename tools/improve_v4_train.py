"""Add new nearest-resource/tail signals with outer/inner label exclusion."""
from pathlib import Path
import sys,json,time,hashlib,warnings
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from train import nn_feature_block
from improve_v1 import best,aggregate
warnings.filterwarnings('ignore',message='Mean of empty slice')
out=ROOT/'improved_v4';out.mkdir(exist_ok=True)
tr=pd.read_csv(ROOT/'data/train.csv');g=tr.groupby('url').result.nunique();tr=tr[~tr.url.isin(g[g>1].index)].drop_duplicates('url').reset_index(drop=True);te=pd.read_csv(ROOT/'data/test.csv');y=tr.result.to_numpy();n=len(y);nt=len(te)
assert np.array_equal(y,np.load(ROOT/'improved_v1/predictions.npz')['y'])
views={}
for field in ['resources','tail']:
 arr=[np.load(out/f'nn_{field}_{s}.npz') for s in ['train','test']];views[field]=(np.vstack([a['idx'] for a in arr]),np.vstack([a['sim'] for a in arr]));assert not np.any((arr[0]['idx']==np.arange(n)[:,None])&(arr[0]['sim']>0))
frames=[]
for s,df in [('train',tr),('test',te)]:frames.append(pd.read_pickle(ROOT/f'improved_v2/resources_{s}.pkl').drop_duplicates('url').set_index('url').reindex(df.url).reset_index(drop=True))
a=pd.concat(frames,ignore_index=True);keys={field:np.array([hashlib.sha256(t.encode()).hexdigest() if len(t)>40 else '' for t in a[field]]) for field in ['resources','tail']}
del frames,a
def extra(reference,query):
 allowed=np.zeros(n,dtype=np.int16);allowed[reference]=1;chunks=[]
 for idx,sim in views.values():
  f,_=nn_feature_block(idx[query],sim[query],y,fold_of=allowed,current_fold=0);chunks.append(f)
 for k in keys.values():chunks.append(aggregate(k,y,reference,query))
 return np.hstack(chunks).astype(np.float32)
oof=np.zeros(n);test=np.zeros(nt)
for fold in range(5):
 d=np.load(ROOT/f'improved_v1/features_fold_{fold}.npz');itr=d['itr'];iva=d['iva'];ex=np.zeros((len(itr),36),dtype=np.float32)
 for aa,bb in StratifiedKFold(4,shuffle=True,random_state=100+fold).split(itr,y[itr]):ex[bb]=extra(itr[aa],itr[bb])
 X=np.hstack([d['train'],ex]);V=np.hstack([d['valid'],extra(itr,iva)]);T=np.hstack([d['test'],extra(itr,np.arange(n,n+nt))]);del d
 np.savez_compressed(out/f'features_fold_{fold}.npz',train=X,valid=V,test=T,itr=itr,iva=iva)
 t=time.time();m=CatBoostClassifier(iterations=1600,depth=7,learning_rate=.04,l2_leaf_reg=7,loss_function='Logloss',random_seed=842+fold,thread_count=6,verbose=400,allow_writing_files=False);m.fit(X,y[itr]);m.save_model(str(out/f'neighbors_fold_{fold}.cbm'));oof[iva]=m.predict_proba(V)[:,1];test+=m.predict_proba(T)[:,1]/5;print('fold',fold+1,'seconds',time.time()-t,flush=True)
f,th=best(y,oof);np.savez(out/'neighbors_predictions.npz',oof=oof,test=test,y=y);(out/'neighbors_report.json').write_text(json.dumps({'oof_f1':f,'threshold':th,'validation':'5 outer folds, 4 inner label exclusions; fixed iterations'},indent=2));print('finished',f,flush=True)
