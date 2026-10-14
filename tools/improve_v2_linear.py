"""Independent character/word linear model; supplied offline page text only."""
from pathlib import Path
import sys,json,time,html
import numpy as np
import pandas as pd
from scipy.sparse import hstack,save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
import joblib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tools'))
from train import align_text
from improve_v1 import best

def main():
    out=ROOT/'improved_v2';out.mkdir(exist_ok=True)
    tr=pd.read_csv(ROOT/'data/train.csv');g=tr.groupby('url').result.nunique();tr=tr[~tr.url.isin(g[g>1].index)].drop_duplicates('url').reset_index(drop=True)
    te=pd.read_csv(ROOT/'data/test.csv');y=tr.result.to_numpy();n=len(tr)
    texts=[];titles=[]
    for name,df in [('train',tr),('test',te)]:
        texts+=align_text(df.url,str(ROOT/f'cache/content_text_{name}.pkl'),field='ctext')
        titles+=align_text(df.url,str(ROOT/f'cache/content_title_{name}.pkl'),field='title')
    texts=[html.unescape(a) for a in texts];titles=[html.unescape(a) for a in titles]
    urls=pd.concat([tr.url,te.url]).astype(str).tolist()
    specs=[('body_char',texts,dict(analyzer='char',ngram_range=(3,5),min_df=4,max_features=160000),1.),
           ('body_word',texts,dict(analyzer='word',ngram_range=(1,2),min_df=3,max_features=70000),.65),
           ('url_char',urls,dict(analyzer='char',ngram_range=(2,5),min_df=3,max_features=100000),.85),
           ('title_char',titles,dict(analyzer='char',ngram_range=(2,5),min_df=3,max_features=50000),.6)]
    blocks=[];vecs={}
    for name,txt,params,weight in specs:
        print('vectorize',name,flush=True);v=TfidfVectorizer(dtype=np.float32,sublinear_tf=True,strip_accents='unicode',**params)
        X=v.fit_transform(txt);blocks.append(X*weight);vecs[name]=(v,weight)
        print(name,X.shape,X.nnz,flush=True)
    X=hstack(blocks,format='csr',dtype=np.float32);del blocks
    save_npz(out/'linear_test_features.npz',X[n:]);joblib.dump(vecs,out/'linear_vectorizers.joblib',compress=3)
    print('matrix',X.shape,'nnz',X.nnz,flush=True)
    oof=np.zeros(n);test=np.zeros(len(te))
    for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(X[:n],y)):
        t=time.time();m=LinearSVC(C=1.0,dual=True,max_iter=2500,tol=1e-4,random_state=42+fold)
        m.fit(X[a],y[a]);oof[b]=m.decision_function(X[b]);test+=m.decision_function(X[n:])/5
        joblib.dump(m,out/f'linear_fold_{fold}.joblib',compress=3)
        print('fold',fold+1,'seconds',time.time()-t,'F1',f1_score(y[b],oof[b]>=0),flush=True)
    f,t=best(y,oof);np.savez(out/'linear_predictions.npz',oof=oof,test=test,y=y)
    (out/'linear_report.json').write_text(json.dumps({'oof_f1':f,'threshold':t,'features':int(X.shape[1]),'public_f1':None},indent=2))
    print('finished',f,t,flush=True)

if __name__=='__main__':main()
