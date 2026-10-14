"""Re-run saved fold models on frozen, independently generated test features."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'improved_v1'
report=json.loads((out/'report.json').read_text())
manifest=json.loads((out/'input_manifest.json').read_text())
for f in ['data/train.csv','data/test.csv','data/sample.csv']:
    assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==manifest[f],f
proba=np.zeros(report['test_rows'])
for fold in range(5):
    m=CatBoostClassifier();m.load_model(str(out/f'fold_{fold}.cbm'))
    with np.load(out/f'features_fold_{fold}.npz') as d:
        proba+=m.predict_proba(d['test'])[:,1]/5
sub=pd.read_csv(ROOT/'data/sample.csv');sub['Predicted']=(proba>=report['threshold']).astype(int)
sub.to_csv(out/'submission_reproduced.csv',index=False)
expected=pd.read_csv(out/'submission_improved.csv')
assert sub.equals(expected),'Predictions differ!'
saved=np.load(out/'predictions.npz')['test']
print('Reproduction: identical labels;',len(sub),'rows; max probability deviation',np.max(np.abs(proba-saved)))
