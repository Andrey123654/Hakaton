"""Refit independently validated V2 text model on all training examples."""
from pathlib import Path
import sys,json,html,time
import numpy as np,pandas as pd,joblib
from scipy.sparse import hstack,save_npz
from scipy.stats import rankdata
from sklearn.svm import LinearSVC
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));from train import align_text
out=ROOT/'improved_v7';out.mkdir(exist_ok=True)
tr=pd.read_csv(ROOT/'data/train.csv');g=tr.groupby('url').result.nunique();tr=tr[~tr.url.isin(g[g>1].index)].drop_duplicates('url').reset_index(drop=True);te=pd.read_csv(ROOT/'data/test.csv');y=tr.result.to_numpy();n=len(y)
texts=[];titles=[]
for name,df in [('train',tr),('test',te)]:
 texts+=align_text(df.url,str(ROOT/f'cache/content_text_{name}.pkl'),field='ctext');titles+=align_text(df.url,str(ROOT/f'cache/content_title_{name}.pkl'),field='title')
texts=[html.unescape(a) for a in texts];titles=[html.unescape(a) for a in titles];urls=pd.concat([tr.url,te.url]).astype(str).tolist();vecs=joblib.load(ROOT/'improved_v2/linear_vectorizers.joblib');blocks=[]
for name,txt in [('body_char',texts),('body_word',texts),('url_char',urls),('title_char',titles)]:
 v,w=vecs[name];blocks.append(v.transform(txt)*w);print('transform',name,flush=True)
X=hstack(blocks,format='csr',dtype=np.float32);del blocks,texts,titles,urls;save_npz(out/'linear_test_features.npz',X[n:]);assert np.array_equal(y,np.load(ROOT/'improved_v2/linear_predictions.npz')['y'])
m=LinearSVC(C=1.,dual=True,max_iter=2500,tol=1e-4,random_state=42);m.fit(X[:n],y);joblib.dump(m,out/'linear_full.joblib',compress=3);p=m.decision_function(X[n:]);np.savez(out/'linear_predictions.npz',test=p)
# Replace half of the existing SVM contribution with its full-data refit.
# V2=0.9*V1blend+0.1*fold-SVM; V5=0.9*V2+0.1*full-CatBoost.
b=np.load(ROOT/'improved_v2/blend_predictions.npz')['test'];oldlinear=np.load(ROOT/'improved_v2/linear_predictions.npz')['test'];fulltree=np.load(ROOT/'improved_v5/full_predictions.npz')['test'];nt=len(te)
oldrank=rankdata(oldlinear)/nt;newrank=rankdata(p)/nt
v5=.9*b+.1*rankdata(fulltree)/nt
score=v5+.045*(newrank-oldrank);threshold=.6257334898278559;pred=(score>=threshold).astype(int);sample=pd.read_csv(ROOT/'data/sample.csv');sample['Predicted']=pred;sample.to_csv(out/'submission_v7.csv',index=False);old=pd.read_csv(ROOT/'improved_v5/submission_v5_refit.csv').Predicted.to_numpy();ix=np.flatnonzero(pred!=old);pd.DataFrame({'Id':ix,'url':te.url.iloc[ix].to_numpy(),'V5':old[ix],'V7':pred[ix]}).to_csv(out/'changes.csv',index=False)
report={'method':'Preserve V5 tree weight and threshold; replace half of original fold-SVM contribution with its full-data refit','replaced_rank_weight':.045,'train_rows':n,'changed_answers_vs_v5':len(ix),'best_confirmed_public':.995076,'public_f1':None,'local_f1':None,'validation':'Original SVM recipe validated on 5 folds in V2; final refit has no independent OOF evaluation. Fixed replacement fraction, no test labels.'};(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
