"""Check saved-model inference and byte-identical final CSV regeneration."""
from pathlib import Path
import sys,json,hashlib,subprocess
import numpy as np,pandas as pd,joblib
from scipy.sparse import load_npz
from catboost import CatBoostClassifier
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'improved_v3';report={}
for name in ['shallow','deep']:
 saved=np.load(out/f'{name}_predictions.npz');test=np.zeros(16000);oof=np.zeros(len(saved['y']))
 for fold in range(5):
  a=np.load(ROOT/f'improved_v1/features_fold_{fold}.npz');m=CatBoostClassifier();m.load_model(str(out/f'{name}_fold_{fold}.cbm'));test+=m.predict_proba(a['test'])[:,1]/5;oof[a['iva']]=m.predict_proba(a['valid'])[:,1]
 assert np.allclose(test,saved['test'],rtol=0,atol=1e-12);assert np.allclose(oof,saved['oof'],rtol=0,atol=1e-12);report[name]={'max_score_difference':float(np.max(abs(test-saved['test'])))}
x=load_npz(out/'words_test_features.npz');saved=np.load(out/'words_predictions.npz');test=np.zeros(16000)
for fold in range(5):test+=joblib.load(out/f'words_fold_{fold}.joblib').predict_proba(x)[:,1]/5
assert np.allclose(test,saved['test'],rtol=0,atol=1e-12);report['words']={'max_score_difference':float(np.max(abs(test-saved['test'])))}
files=list(out.glob('submission_v3*.csv'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();before={p.name:sha(p) for p in files}
subprocess.run([sys.executable,str(ROOT/'tools/blend_improved_v3.py')],check=True,stdout=subprocess.DEVNULL)
sys.path.insert(0,str(ROOT/'tools'))
from url_identity_v3 import lookup
tr=pd.read_csv(ROOT/'data/train.csv');g=tr.groupby('url').result.nunique();tr=tr[~tr.url.isin(g[g>1].index)].drop_duplicates('url');te=pd.read_csv(ROOT/'data/test.csv');k=lookup(tr,te);identity=pd.read_csv(ROOT/'improved_v2/submission_v2.csv');identity.loc[k.notna(),'Predicted']=k[k.notna()].astype(int);identity.to_csv(out/'submission_v3_identity.csv',index=False)
sample=pd.read_csv(ROOT/'data/sample.csv')
for p in files:
 s=pd.read_csv(p);assert s.columns.tolist()==['Id','Predicted'];assert s.Id.equals(sample.Id);assert s.Predicted.isin([0,1]).all();assert sha(p)==before[p.name];report[p.name]={'rows':len(s),'sha256':sha(p)}
baseline=ROOT/'improved_v2/submission_v2.csv';assert sha(baseline)=='f28adee1f5dd1ba5982726f1d36e3dd9558df4476cd130ea927455ae2aedad94';report['baseline_unchanged']=True
(out/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
