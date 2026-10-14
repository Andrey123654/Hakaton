"""Independent lexical LogisticRegression with supplied offline text."""
from pathlib import Path
import sys,json,html,time
import numpy as np,pandas as pd,joblib
from scipy.sparse import hstack,save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from train import align_text
from improve_v1 import best
out=ROOT/'improved_v3';out.mkdir(exist_ok=True)
tr=pd.read_csv(ROOT/'data/train.csv');g=tr.groupby('url').result.nunique();tr=tr[~tr.url.isin(g[g>1].index)].drop_duplicates('url').reset_index(drop=True);te=pd.read_csv(ROOT/'data/test.csv');y=tr.result.to_numpy();n=len(y)
texts=[];titles=[]
for name,df in [('train',tr),('test',te)]:
 texts.extend(align_text(df.url,str(ROOT/f'cache/content_text_{name}.pkl'),field='ctext'));titles.extend(align_text(df.url,str(ROOT/f'cache/content_title_{name}.pkl'),field='title'))
texts=[html.unescape(t) for t in texts];titles=[html.unescape(t) for t in titles];urls=pd.concat([tr.url,te.url]).astype(str).tolist()
specs=[('word',texts,dict(analyzer='word',ngram_range=(1,3),min_df=2,max_features=180000),1.),('title',titles,dict(analyzer='word',ngram_range=(1,2),min_df=2,max_features=30000),.7),('url',urls,dict(analyzer='char',ngram_range=(3,6),min_df=3,max_features=100000),.8)]
blocks=[];vecs={}
for name,txt,params,w in specs:
 v=TfidfVectorizer(dtype=np.float32,sublinear_tf=True,strip_accents='unicode',**params);x=v.fit_transform(txt);blocks.append(x*w);vecs[name]=(v,w);print(name,x.shape,x.nnz,flush=True)
x=hstack(blocks,format='csr');del blocks;save_npz(out/'words_test_features.npz',x[n:]);joblib.dump(vecs,out/'words_vectorizers.joblib',compress=3)
oof=np.zeros(n);test=np.zeros(len(te))
for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(y,y)):
 t=time.time();m=LogisticRegression(C=12,solver='liblinear',max_iter=300,random_state=73+fold);m.fit(x[a],y[a]);oof[b]=m.predict_proba(x[b])[:,1];test+=m.predict_proba(x[n:])[:,1]/5;joblib.dump(m,out/f'words_fold_{fold}.joblib',compress=3);print('fold',fold+1,time.time()-t,flush=True)
f,th=best(y,oof);np.savez(out/'words_predictions.npz',y=y,oof=oof,test=test);(out/'words_report.json').write_text(json.dumps({'oof_f1':f,'threshold':th,'public_f1':None},indent=2));print('finished',f,flush=True)
