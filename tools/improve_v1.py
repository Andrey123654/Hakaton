"""Independent outer-fold validation; no held-out labels in any feature.

Run from project root: .venv/bin/python tools/improve_v1.py
Uses supplied offline feature caches; preserves all previous submissions.
"""
from pathlib import Path
import sys, json, hashlib, warnings
import numpy as np
import pandas as pd
from urllib.parse import urlsplit
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve, f1_score
from catboost import CatBoostClassifier
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from train import attach_content_features, align_text, nn_feature_block
warnings.filterwarnings('ignore', message='Mean of empty slice')

def best(y,p):
    pr,re,th=precision_recall_curve(y,p)
    f=2*pr[:-1]*re[:-1]/np.maximum(pr[:-1]+re[:-1],1e-15)
    j=int(np.argmax(f));return float(f[j]),float(th[j])

def groups(urls,part):
    out=[]
    for u in urls:
        try:
            p=urlsplit(u if '://' in u else 'http://'+u)
            h=(p.hostname or '').lower().removeprefix('www.')
            out.append(h if part=='host' else '.'.join(h.split('.')[-2:]))
        except ValueError:out.append('')
    return np.array(out)

def aggregate(keys, target, reference, query):
    d=pd.DataFrame({'k':keys[reference],'y':target[reference]}).groupby('k').y.agg(['sum','count'])
    a=d.reindex(keys[query]).fillna(0).to_numpy()
    missing=np.array([not bool(k) for k in keys[query]])
    a[missing]=0
    return np.column_stack([np.log1p(a[:,1]),(a[:,0]+0.5)/(a[:,1]+1)])

def main():
    out=ROOT/'improved_v1';out.mkdir(exist_ok=True)
    raw=pd.read_csv(ROOT/'data/train.csv')
    bad=raw.groupby('url').result.nunique();tr=raw[~raw.url.isin(bad[bad>1].index)].drop_duplicates('url').reset_index(drop=True)
    te=pd.read_csv(ROOT/'data/test.csv');y=tr.result.to_numpy();n=len(tr);nt=len(te)
    assert np.array_equal(y,np.load(ROOT/'models/oof_nn2.npz')['y']), 'Cache label order mismatch'
    frames=[]
    for name,df in [('train',tr),('test',te)]:
        f=pd.read_pickle(ROOT/f'cache/feat_nn3_{name}.pkl').copy()
        assert len(f)==len(df)
        f=attach_content_features(f,df.url,str(ROOT/f'cache/content_{name}.pkl'),name)
        frames.append(f.select_dtypes(include=[np.number]).fillna(0).to_numpy(dtype=np.float32))
    base=np.vstack(frames)
    allurl=pd.concat([tr.url,te.url],ignore_index=True)
    keys={k:groups(allurl,k) for k in ['host','domain']}
    for view,field in [('title','title'),('text','ctext'),('markup','markup'),('fingerprint','fingerprint')]:
        filename='content_text' if view=='text' else 'content_'+view
        txt=sum([align_text(df.url,str(ROOT/f'cache/{filename}_{name}.pkl'),name,field=field) for name,df in [('train',tr),('test',te)]],[])
        keys[view]=np.array([hashlib.sha256(t.encode()).hexdigest() if len(t.strip())>=12 else '' for t in txt])
    views={}
    for view,folder in [('word','cache_k100'),('markup','cache_k100'),('urlpath','cache_k100'),('title','cache_k100'),('fingerprint','cache')]:
        arr=[np.load(ROOT/f'{folder}/nn_{view}_{s}.npz') for s in ['train','test']]
        assert arr[0]['idx'].shape[0]==n and arr[1]['idx'].shape[0]==nt
        assert not np.any((arr[0]['idx']==np.arange(n)[:,None]) & (arr[0]['sim']>0))
        views[view]=(np.vstack([a['idx'] for a in arr]),np.vstack([a['sim'] for a in arr]))
    outer=list(StratifiedKFold(5,shuffle=True,random_state=42).split(base[:n],y))
    oof=np.zeros(n);pred=np.zeros(nt)
    def features(reference,query):
        allowed=np.zeros(n,dtype=np.int16);allowed[reference]=1
        chunks=[base[query]]
        for idx,sim in views.values():
            b,_=nn_feature_block(idx[query],sim[query],y,fold_of=allowed,current_fold=0)
            chunks.append(b)
        for k in keys.values():chunks.append(aggregate(k,y,reference,query))
        return np.hstack(chunks).astype(np.float32)
    for fold,(itr,iva) in enumerate(outer):
        print('Fold',fold+1,flush=True)
        # Inner cross-fitting prevents training rows from using their own labels.
        X=np.zeros((len(itr),features(itr,iva[:1]).shape[1]),dtype=np.float32)
        for a,b in StratifiedKFold(4,shuffle=True,random_state=100+fold).split(itr,y[itr]):
            X[b]=features(itr[a],itr[b])
        V=features(itr,iva);T=features(itr,np.arange(n,n+nt))
        m=CatBoostClassifier(iterations=1400,depth=7,learning_rate=.045,l2_leaf_reg=5,
                            loss_function='Logloss',random_seed=42+fold,thread_count=6,
                            verbose=200,allow_writing_files=False)
        # Fixed iteration count: outer validation is not used for early stopping.
        m.fit(X,y[itr]);oof[iva]=m.predict_proba(V)[:,1];pred+=m.predict_proba(T)[:,1]/5
        m.save_model(str(out/f'fold_{fold}.cbm'))
        np.savez_compressed(out/f'features_fold_{fold}.npz',train=X,valid=V,test=T,itr=itr,iva=iva)
        print('fold F1 at 0.5',f1_score(y[iva],oof[iva]>=.5),flush=True)
        np.savez(out/'progress.npz',oof=oof,test=pred)
    score,threshold=best(y,oof)
    sub=pd.read_csv(ROOT/'data/sample.csv');sub['Predicted']=(pred>=threshold).astype(int)
    assert len(sub)==nt and sub.Id.tolist()==list(range(nt))
    sub.to_csv(out/'submission_improved.csv',index=False)
    np.savez(out/'predictions.npz',oof=oof,test=pred,y=y)
    report={'oof_f1':score,'threshold':threshold,'f1_at_0.5':f1_score(y,oof>=.5),
            'validation':'5 outer folds; 4 inner folds for label-derived training features; fixed iterations',
            'public_score':None,'train_rows':n,'test_rows':nt,
            'differences_vs_blend4':int((sub.Predicted!=pd.read_csv(ROOT/'submission_blend4.csv').Predicted).sum())}
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(report,flush=True)

if __name__=='__main__':main()
