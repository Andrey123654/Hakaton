"""Independent new resource/tail nearest-neighbor views; no query labels."""
from pathlib import Path
import sys,json,time
import numpy as np,pandas as pd,joblib
from sklearn.feature_extraction.text import TfidfVectorizer
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from nn_features import topk_similarity,l2_normalize
out=ROOT/'improved_v4';out.mkdir(exist_ok=True)
tr=pd.read_csv(ROOT/'data/train.csv');g=tr.groupby('url').result.nunique();tr=tr[~tr.url.isin(g[g>1].index)].drop_duplicates('url').reset_index(drop=True);te=pd.read_csv(ROOT/'data/test.csv');n=len(tr)
frames=[]
for name,df in [('train',tr),('test',te)]:
 f=pd.read_pickle(ROOT/f'improved_v2/resources_{name}.pkl').drop_duplicates('url').set_index('url').reindex(df.url);assert f.notna().all().all();frames.append(f.reset_index(drop=True))
a=pd.concat(frames,ignore_index=True)
for field in ['resources','tail']:
 if (out/f'nn_{field}_test.npz').exists():continue
 t=time.time();v=TfidfVectorizer(dtype=np.float32,analyzer='word',ngram_range=(1,2),min_df=3,max_df=.2,max_features=30000,sublinear_tf=True,token_pattern=r'\S+' if field=='resources' else r'(?u)\b\w\w+\b')
 X=l2_normalize(v.fit_transform(a[field]));joblib.dump(v,out/f'{field}_vectorizer.joblib',compress=3);print(field,'matrix',X.shape,X.nnz,flush=True)
 for name,q,offset in [('train',X[:n],0),('test',X[n:],None)]:
  path=out/f'nn_{field}_{name}.npz'
  if path.exists():continue
  idx,sim=topk_similarity(q,X[:n],100,exclude_self_offset=offset,chunk=256)
  np.savez_compressed(path,idx=idx,sim=sim);print(field,name,'finished seconds',time.time()-t,flush=True)
