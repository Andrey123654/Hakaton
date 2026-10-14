"""Validate unlabeled peer consistency on held-out groups, no test ground truth."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')];from improve_v1 import best;from train import align_text
out=ROOT/'improved_v9';out.mkdir(exist_ok=True);base=np.load(ROOT/'improved_v2/blend_predictions.npz');y=base['y'];n=len(y);nt=len(base['test']);C=pd.read_pickle(ROOT/'improved_v8/categories.pkl');tr=pd.read_csv(ROOT/'data/train.csv');g=tr.groupby('url').result.nunique();tr=tr[~tr.url.isin(g[g>1].index)].drop_duplicates('url').reset_index(drop=True);te=pd.read_csv(ROOT/'data/test.csv')
keys={'host':C.host.to_numpy()};print('loaded',flush=True)
text=[]
for name,df in [('train',tr),('test',te)]:text+=align_text(df.url,str(ROOT/f'cache/content_text_{name}.pkl'),field='ctext')
keys['text']=np.array([hashlib.sha256(t.encode()).hexdigest() if len(t.strip())>=100 else '' for t in text]);del text
splits=list(StratifiedKFold(5,shuffle=True,random_state=42).split(y,y))
def smooth(keys,p,w):
 # All source decisions are out-of-sample; use peers in same held-out block.
 confident=(p<=.4)|(p>=.8);d=pd.DataFrame({'key':keys,'p':p,'conf':confident});refs=d.loc[confident & (d.key!='')].groupby('key').p.agg(['sum','count','min','max']);mapped=refs.reindex(keys);count=mapped['count'].fillna(0).to_numpy()-confident.astype(int);total=mapped['sum'].fillna(0).to_numpy()-p*confident
 avg=total/np.maximum(count,1);same=(mapped['min'].to_numpy()>=.8)|(mapped['max'].to_numpy()<=.4);eligible=(count>=2)&same&(keys!='');q=p.copy();q[eligible]=(1-w)*p[eligible]+w*avg[eligible];return q
rows=[];scores=[]
for name,k in keys.items():
 for w in [.05,.1,.2]:
  p=base['oof'].copy()
  for a,v in splits:p[v]=smooth(k[v],base['oof'][v],w)
  f,t=best(y,p);pred=np.zeros(n,int)
  for a,v in splits:_,th=best(y[a],p[a]);pred[v]=p[v]>=th
  rows.append({'group':name,'weight':w,'oof_f1':f,'crossfit_f1':float(f1_score(y,pred)),'threshold':t});scores.append(p)
bestrow=max(rows,key=lambda z:z['crossfit_f1']);report={'baseline_crossfit_f1':.9901159753421795,'candidates':rows,'selected':None,'public_f1':None,'caveat':'Unlabeled peer scores in each outer validation block; prior OOF models reused; weights selected from OOF, not full nested retraining.'}
if bestrow['crossfit_f1']>report['baseline_crossfit_f1']:
 tree=np.load(ROOT/'improved_v5/full_predictions.npz')['test'];v5=.9*base['test']+.1*rankdata(tree)/nt;p=smooth(keys[bestrow['group']][n:],v5,bestrow['weight']);pred=(p>=bestrow['threshold']).astype(int);old=pd.read_csv(ROOT/'improved_v5/submission_v5_refit.csv').Predicted.to_numpy();changes=int(np.sum(old!=pred));report['selected']={**bestrow,'changes_vs_v5':changes};np.savez(out/'consistency_inputs.npz',keys=keys[bestrow['group']][n:],v5=v5,score=p)
 if changes:
  s=pd.read_csv(ROOT/'data/sample.csv');s['Predicted']=pred;s.to_csv(out/'submission_v9.csv',index=False);ix=np.flatnonzero(old!=pred);pd.DataFrame({'Id':ix,'url':te.url.iloc[ix].to_numpy(),'V5':old[ix],'V9':pred[ix]}).to_csv(out/'changes.csv',index=False)
(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
