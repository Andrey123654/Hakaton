"""Native categorical URL/title model; outer validation excludes held-out labels."""
from pathlib import Path
import sys,json,html,re,time
from urllib.parse import urlsplit,unquote
import numpy as np,pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')];from train import align_text;from improve_v1 import best
out=ROOT/'improved_v8';out.mkdir(exist_ok=True)
tr=pd.read_csv(ROOT/'data/train.csv');g=tr.groupby('url').result.nunique();tr=tr[~tr.url.isin(g[g>1].index)].drop_duplicates('url').reset_index(drop=True);te=pd.read_csv(ROOT/'data/test.csv');y=tr.result.to_numpy();n=len(y);nt=len(te);print('data loaded',n,flush=True)
titles=[]
for name,df in [('train',tr),('test',te)]:titles+=align_text(df.url,str(ROOT/f'cache/content_title_{name}.pkl'),field='title')
def category(u,t):
 try:
  p=urlsplit(html.unescape(u if '://' in u else 'http://'+u));h=(p.hostname or '').lower().removeprefix('www.');parts=[re.sub(r'\d+','0',s)[:60] for s in unquote(p.path).lower().split('/') if s];domain='.'.join(h.split('.')[-2:]);first=parts[0] if parts else '';two='/'.join(parts[:2]);ext=parts[-1].rsplit('.',1)[-1] if parts and '.' in parts[-1] else '';title=re.sub(r'\s+',' ',html.unescape(t).lower()).strip()[:100]
  return [h,domain,first,two,ext,title,h+'|'+first,domain+'|'+first]
 except ValueError:return ['']*8
C=pd.DataFrame([category(u,t) for u,t in zip(pd.concat([tr.url,te.url]),titles)],columns=['host','domain','path1','path2','extension','title','host_path1','domain_path1']).replace('','<empty>');C.to_pickle(out/'categories.pkl');del titles
cats=list(range(288,296))
def frame(x,c):
 a=pd.DataFrame(x,columns=[f'f{i}' for i in range(288)]);return pd.concat([a,c.reset_index(drop=True)],axis=1)
def model(seed):return CatBoostClassifier(iterations=1200,depth=6,learning_rate=.05,l2_leaf_reg=10,random_seed=seed,thread_count=6,verbose=400,allow_writing_files=False,max_ctr_complexity=2,cat_features=cats)
oof=np.zeros(n);test=np.zeros(nt)
for fold in range(5):
 a=np.load(ROOT/f'improved_v1/features_fold_{fold}.npz');itr=a['itr'];iva=a['iva'];X=frame(a['train'],C.iloc[itr]);V=frame(a['valid'],C.iloc[iva]);T=frame(a['test'],C.iloc[n:]);path=out/f'fold_{fold}.cbm';m=model(2042+fold)
 if path.exists():m.load_model(str(path))
 else:m.fit(X,y[itr]);m.save_model(str(path))
 oof[iva]=m.predict_proba(V)[:,1];test+=m.predict_proba(T)[:,1]/5;print('fold',fold+1,'complete',flush=True);del X,V,T,a
np.savez(out/'predictions.npz',y=y,oof=oof,test=test);base=np.load(ROOT/'improved_v2/blend_predictions.npz');r=rankdata(oof)/n;rows=[];splits=list(StratifiedKFold(5,shuffle=True,random_state=42).split(y,y))
for w in [0,.05,.1,.2]:
 p=(1-w)*base['oof']+w*r;f,t=best(y,p);pr=np.zeros(n,int)
 for aa,bb in splits:_,th=best(y[aa],p[aa]);pr[bb]=p[bb]>=th
 rows.append({'weight':w,'oof_f1':f,'threshold':t,'crossfit_f1':float(f1_score(y,pr))})
report={'standalone':best(y,oof),'blends':rows,'public_f1':None,'validation':'Native categories raw URL/title only; fold matrices exclude validation labels; fixed parameters; prior baseline OOF reused.'};(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
