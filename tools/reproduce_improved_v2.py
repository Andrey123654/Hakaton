"""Verify exact inference from saved SVMs, then reproducibly rebuild the mixture."""
from pathlib import Path
import sys,subprocess,hashlib,json
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
import joblib
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'improved_v2'
checks={}
for kind in ['linear','resources']:
    X=load_npz(out/f'{kind}_test_features.npz');p=np.zeros(X.shape[0])
    for f in range(5):
        m=joblib.load(out/f'{kind}_fold_{f}.joblib');p+=m.decision_function(X)/5
    expected=np.load(out/f'{kind}_predictions.npz')['test']
    deviation=float(np.max(np.abs(p-expected)));assert np.allclose(p,expected,rtol=0,atol=1e-10)
    checks[kind]={'max_score_difference':deviation,'rows':len(p)}
original=hashlib.sha256((out/'submission_v2.csv').read_bytes()).hexdigest()
subprocess.run([sys.executable,str(ROOT/'tools/blend_improved_v2.py')],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
assert hashlib.sha256((out/'submission_v2.csv').read_bytes()).hexdigest()==original
sample=pd.read_csv(ROOT/'data/sample.csv')
for name in ['submission_v2.csv','submission_v2_conservative.csv']:
    df=pd.read_csv(out/name);assert list(df.columns)==['Id','Predicted'];assert df.Id.equals(sample.Id)
    assert df.Predicted.isin([0,1]).all() and not df.isna().any().any()
    checks[name]={'rows':len(df),'sha256':hashlib.sha256((out/name).read_bytes()).hexdigest()}
(out/'verification.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
