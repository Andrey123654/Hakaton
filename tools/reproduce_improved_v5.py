"""Verify saved models and regenerate submissions exactly."""
from pathlib import Path
import sys,json,hashlib,subprocess
import numpy as np,pandas as pd,joblib
from scipy.sparse import load_npz
from catboost import CatBoostClassifier
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'improved_v5';report={}
saved=np.load(out/'hierarchy_predictions.npz');test=np.zeros(16000);oof=np.zeros(len(saved['y']))
for fold in range(5):
 a=np.load(out/f'hierarchy_features_fold_{fold}.npz');m=CatBoostClassifier();m.load_model(str(out/f'hierarchy_fold_{fold}.cbm'));test+=m.predict_proba(a['test'])[:,1]/5;oof[a['iva']]=m.predict_proba(a['valid'])[:,1]
assert np.allclose(test,saved['test'],rtol=0,atol=1e-12);assert np.allclose(oof,saved['oof'],rtol=0,atol=1e-12);report['hierarchy']={'max_score_difference':float(np.max(abs(test-saved['test'])))}
saved=np.load(out/'nbsvm_predictions.npz');X=load_npz(out/'nbsvm_test_features.npz');test=np.zeros(16000)
for fold in range(5):
 r,m=joblib.load(out/f'nbsvm_fold_{fold}.joblib');test+=m.decision_function(X.multiply(r).tocsr())/5
assert np.allclose(test,saved['test'],rtol=0,atol=1e-12);report['nbsvm']={'max_score_difference':float(np.max(abs(test-saved['test'])))}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();files=list(out.glob('submission_v5*.csv'));before={p.name:sha(p) for p in files}
subprocess.run([sys.executable,str(ROOT/'tools/blend_improved_v5.py')],check=True,stdout=subprocess.DEVNULL)
sample=pd.read_csv(ROOT/'data/sample.csv')
for p in files:
 s=pd.read_csv(p);assert s.columns.tolist()==['Id','Predicted'];assert s.Id.equals(sample.Id);assert len(s)==16000;assert s.Predicted.isin([0,1]).all();assert sha(p)==before[p.name];report[p.name]={'rows':len(s),'sha256':sha(p)}
assert sha(ROOT/'improved_v2/submission_v2.csv')=='f28adee1f5dd1ba5982726f1d36e3dd9558df4476cd130ea927455ae2aedad94';report['baseline_unchanged']=True
(out/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
