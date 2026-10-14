"""Independent NB-SVM lexical classifier, fold-local class feature ratios."""
from pathlib import Path
import sys,json,html,time
import numpy as np,pandas as pd,joblib
from scipy.sparse import hstack,save_npz
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from train import align_text
from improve_v1 import best
out=ROOT/'improved_v5';out.mkdir(exist_ok=True)
tr=pd.read_csv(ROOT/'data/train.csv');g=tr.groupby('url').result.nunique();tr=tr[~tr.url.isin(g[g>1].index)].drop_duplicates('url').reset_index(drop=True);te=pd.read_csv(ROOT/'data/test.csv');y=tr.result.to_numpy();n=len(y)
texts=[];titles=[]
for name,df in [('train',tr),('test',te)]:
 texts.extend(align_text(df.url,str(ROOT/f'cache/content_text_{name}.pkl'),field='ctext'));titles.extend(align_text(df.url,str(ROOT/f'cache/content_title_{name}.pkl'),field='title'))
texts=[html.unescape(t) for t in texts];titles=[html.unescape(t) for t in titles];urls=pd.concat([tr.url,te.url]).astype(str).tolist()
vecs=joblib.load(ROOT/'improved_v3/words_vectorizers.joblib');blocks=[]
for name,txt in [('word',texts),('title',titles),('url',urls)]:
 v,w=vecs[name];x=v.transform(txt);blocks.append(x);print(name,x.shape,x.nnz,flush=True)
X=hstack(blocks,format='csr');del blocks,texts,titles,urls;save_npz(out/'nbsvm_test_features.npz',X[n:]);joblib.dump(vecs,out/'vectorizers.joblib',compress=3)
oof=np.zeros(n);test=np.zeros(len(te))
for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(y,y)):
 t=time.time();xa=X[a];pos=np.asarray(xa[y[a]==1].sum(axis=0)).ravel()+1;neg=np.asarray(xa[y[a]==0].sum(axis=0)).ravel()+1;r=np.log((pos/pos.sum())/(neg/neg.sum())).astype(np.float32);r=np.clip(r,-5,5)
 m=LinearSVC(C=.5,dual=True,max_iter=3000,tol=1e-4,random_state=912+fold);m.fit(xa.multiply(r).tocsr(),y[a]);oof[b]=m.decision_function(X[b].multiply(r).tocsr());test+=m.decision_function(X[n:].multiply(r).tocsr())/5;joblib.dump((r,m),out/f'nbsvm_fold_{fold}.joblib',compress=3);print('fold',fold+1,time.time()-t,flush=True)
f,th=best(y,oof);np.savez(out/'nbsvm_predictions.npz',y=y,oof=oof,test=test);(out/'nbsvm_report.json').write_text(json.dumps({'oof_f1':f,'threshold':th,'validation':'5 folds; class log-count ratios fit only on training labels; fixed C=.5'},indent=2));print('finished',f,flush=True)
