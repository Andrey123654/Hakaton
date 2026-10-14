"""Extract resource/form/style tokens and page tail from supplied CSV snapshots."""
from pathlib import Path
import re,html,sys,json,time
from urllib.parse import urlsplit,unquote
import numpy as np
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from scipy.sparse import hstack,save_npz
from multiprocessing import Pool
import joblib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'));from improve_v1 import best
TAGS=re.compile(r'<(script|link|form|input|meta|iframe|img|html|body|div)\b([^<>]{0,6000})>',re.I)
ATTR=re.compile(r'''([\w-]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))''')
STRIP=re.compile(r'<(script|style|noscript)\b[^>]*>.*?</\1>',re.I|re.S)
COMMENT=re.compile(r'<!--.*?-->',re.S)
TAG=re.compile(r'<[^>]+>')
WS=re.compile(r'\s+')

def extract(pair):
    url,page=pair;page=page[:350000];tokens=[]
    for tag,attrs in TAGS.findall(page):
        tag=tag.lower()
        for attr,a,b,c in ATTR.findall(attrs):
            value=html.unescape(a or b or c).lower();attr=attr.lower()
            if attr in ['src','href','action'] and tag in ['script','link','form','iframe','img']:
                # Host and resource filename; remove query IDs and session strings.
                try:
                    p=urlsplit(value);host=p.hostname or ''
                    path=unquote(p.path);basename=path.split('/')[-1]
                    if host:tokens.append(tag+'_'+attr+'_host='+host)
                    for part in re.findall(r'[a-z][a-z0-9_-]{2,40}',basename):tokens.append(tag+'_'+attr+'_file='+re.sub(r'\d+','0',part))
                    if tag=='form':tokens.extend('form_path='+a for a in re.findall(r'[a-z]{3,}',path))
                except ValueError:pass
            elif attr in ['name','type','method','id','class','http-equiv','lang']:
                if attr=='class' and tag not in ['html','body','form','input','div']:continue
                for v in value.split()[:6]:
                    if 1<len(v)<=60:tokens.append(tag+'_'+attr+'='+re.sub(r'\d+','0',v))
        if len(tokens)>650:break
    visible=html.unescape(WS.sub(' ',TAG.sub(' ',COMMENT.sub(' ',STRIP.sub(' ',page))))).strip().lower()
    return url,' '.join(tokens),visible[-3500:] if len(visible)>3000 else visible

def build(name,out):
    dest=out/f'resources_{name}.pkl'
    if dest.exists():return pd.read_pickle(dest)
    rows=[]
    src=Path(os.environ.get('CONTENT_DIR','./Архив 3'))/f'{name}_with_content.csv'
    with Pool(3) as pool:
        for chunk in pd.read_csv(src,usecols=['url','content'],dtype=str,keep_default_na=False,chunksize=400):
            rows.extend(pool.map(extract,zip(chunk.url,chunk.content),chunksize=20))
            if len(rows)%4000==0:print(name,'extracted',len(rows),flush=True)
    df=pd.DataFrame(rows,columns=['url','resources','tail']);df.to_pickle(dest);return df

def main():
    out=ROOT/'improved_v2';out.mkdir(exist_ok=True)
    frames={s:build(s,out) for s in ['train','test']}
    tr=pd.read_csv(ROOT/'data/train.csv');g=tr.groupby('url').result.nunique();tr=tr[~tr.url.isin(g[g>1].index)].drop_duplicates('url').reset_index(drop=True)
    te=pd.read_csv(ROOT/'data/test.csv');y=tr.result.to_numpy();n=len(y)
    aligned=[]
    for s,df in [('train',tr),('test',te)]:
        f=frames[s].drop_duplicates('url').set_index('url').reindex(df.url)
        assert f.resources.notna().all();aligned.append(f.reset_index(drop=True))
    a=pd.concat(aligned,ignore_index=True);blocks=[];vecs={}
    for field,analyzer,ngram,maxf,weight in [('resources','word',(1,2),90000,1.),('tail','word',(1,2),90000,.9),('tail','char',(3,5),100000,.9)]:
        name=field+'_'+analyzer;print('vectorize',name,flush=True)
        params={'token_pattern':r'\S+'} if field=='resources' else {}
        v=TfidfVectorizer(dtype=np.float32,sublinear_tf=True,min_df=3,ngram_range=ngram,max_features=maxf,analyzer=analyzer,**params)
        X=v.fit_transform(a[field]);blocks.append(X*weight);vecs[name]=(v,weight)
        print(name,X.shape,X.nnz,flush=True)
    X=hstack(blocks,format='csr',dtype=np.float32);del blocks
    save_npz(out/'resources_test_features.npz',X[n:]);joblib.dump(vecs,out/'resources_vectorizers.joblib',compress=3)
    oof=np.zeros(n);test=np.zeros(len(te))
    for fold,(itr,iva) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(X[:n],y)):
        t=time.time();m=LinearSVC(C=1.,dual=True,max_iter=2500,tol=1e-4,random_state=42+fold)
        m.fit(X[itr],y[itr]);oof[iva]=m.decision_function(X[iva]);test+=m.decision_function(X[n:])/5
        joblib.dump(m,out/f'resources_fold_{fold}.joblib',compress=3)
        print('fold',fold+1,'seconds',time.time()-t,flush=True)
    f,t=best(y,oof);np.savez(out/'resources_predictions.npz',oof=oof,test=test,y=y)
    (out/'resources_report.json').write_text(json.dumps({'oof_f1':f,'threshold':t,'public_f1':None},indent=2));print('finished',f,t,flush=True)

if __name__=='__main__':main()
