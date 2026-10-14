"""Deliver only validation-improving categorical blend; validate model parity."""
from pathlib import Path
import sys,json,hashlib,shutil
import numpy as np,pandas as pd
from scipy.stats import rankdata
from catboost import CatBoostClassifier
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'improved_v8';report=json.loads((out/'report.json').read_text());base=report['blends'][0];choice=max(report['blends'],key=lambda r:r['crossfit_f1'])
if choice['weight']==0:
 report['selected']=None;report['conclusion']='No crossfit improvement; no new submission generated.';(out/'report.json').write_text(json.dumps(report,indent=2));print(report['conclusion']);sys.exit(0)
C=pd.read_pickle(out/'categories.pkl');saved=np.load(out/'predictions.npz');n=len(saved['y']);nt=len(saved['test']);test=np.zeros(nt);oof=np.zeros(n)
def frame(x,c):return pd.concat([pd.DataFrame(x,columns=[f'f{i}' for i in range(288)]),c.reset_index(drop=True)],axis=1)
for fold in range(5):
 a=np.load(ROOT/f'improved_v1/features_fold_{fold}.npz');m=CatBoostClassifier();m.load_model(str(out/f'fold_{fold}.cbm'));test+=m.predict_proba(frame(a['test'],C.iloc[n:]))[:,1]/5;oof[a['iva']]=m.predict_proba(frame(a['valid'],C.iloc[a['iva']]))[:,1]
assert np.allclose(test,saved['test'],rtol=0,atol=1e-12);assert np.allclose(oof,saved['oof'],rtol=0,atol=1e-12)
b=np.load(ROOT/'improved_v2/blend_predictions.npz')['test'];tree=np.load(ROOT/'improved_v5/full_predictions.npz')['test'];v5=.9*b+.1*rankdata(tree)/nt;w=choice['weight'];score=(1-w)*v5+w*rankdata(test)/nt;pred=(score>=choice['threshold']).astype(int);s=pd.read_csv(ROOT/'data/sample.csv');s['Predicted']=pred;s.to_csv(out/'submission_v8.csv',index=False);old=pd.read_csv(ROOT/'improved_v5/submission_v5_refit.csv');ix=np.flatnonzero(pred!=old.Predicted.to_numpy());te=pd.read_csv(ROOT/'data/test.csv');pd.DataFrame({'Id':ix,'url':te.url.iloc[ix].to_numpy(),'V5':old.Predicted.iloc[ix].to_numpy(),'V8':pred[ix]}).to_csv(out/'changes.csv',index=False)
assert s.columns.tolist()==['Id','Predicted'];assert len(s)==16000;assert s.Id.equals(pd.read_csv(ROOT/'data/sample.csv').Id);assert s.Predicted.isin([0,1]).all();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert sha(ROOT/'improved_v5/submission_v5_refit.csv')=='110bec074aff3462323c6acaa8713bd8103818965b52a4c0a7730eade5f50c0a';target=Path.home()/'Downloads'/'submission_v8.csv';shutil.copy2(out/'submission_v8.csv',target);assert sha(target)==sha(out/'submission_v8.csv')
report['selected']=choice;report['test_changes_vs_v5']=len(ix);report['caveat']='OOF blend assessed against V2. Final test blend retains V5 full-refit component, which has no independent OOF estimate; V8 public gain unverified.';(out/'report.json').write_text(json.dumps(report,indent=2));v={'model_test_parity':float(np.max(abs(test-saved['test']))),'model_oof_parity':float(np.max(abs(oof-saved['oof']))),'rows':len(s),'sha256':sha(target),'delivery':str(target),'V5_unchanged':True};(out/'verification.json').write_text(json.dumps(v,indent=2));print(json.dumps(report,indent=2));print(json.dumps(v,indent=2))
