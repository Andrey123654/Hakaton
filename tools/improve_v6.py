"""Reduce variance in full-data refit with fixed seeds; preserve known winner."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np,pandas as pd
from scipy.stats import rankdata
from catboost import CatBoostClassifier
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'improved_v6';out.mkdir(exist_ok=True)
history=ROOT/'reports/public_scores_all.json';h=json.loads(history.read_text());h['scores']['improved_v5/submission_v5_refit.csv']=.995076;h['best']='improved_v5/submission_v5_refit.csv';history.write_text(json.dumps(h,indent=2))
rpath=ROOT/'improved_v5/refit_report.json';r=json.loads(rpath.read_text());r['public_f1']=.995076;rpath.write_text(json.dumps(r,indent=2))
a=np.load(ROOT/'improved_v5/full_features.npz');p=[np.load(ROOT/'improved_v5/full_predictions.npz')['test']]
for seed in [43,44]:
 modelpath=out/f'full_seed_{seed}.cbm';m=CatBoostClassifier()
 if modelpath.exists():m.load_model(str(modelpath))
 else:
  m=CatBoostClassifier(iterations=1400,depth=7,learning_rate=.045,l2_leaf_reg=5,loss_function='Logloss',random_seed=seed,thread_count=6,verbose=500,allow_writing_files=False);m.fit(a['train'],a['y']);m.save_model(str(modelpath))
 p.append(m.predict_proba(a['test'])[:,1]);print('seed',seed,'done',flush=True)
P=np.column_stack(p);np.savez(out/'full_predictions.npz',test=P)
# Rank each seed individually, then average to match established rank blending.
q=np.mean(np.column_stack([rankdata(v)/len(v) for v in p]),axis=1);b=np.load(ROOT/'improved_v2/blend_predictions.npz')['test'];old=pd.read_csv(ROOT/'improved_v5/submission_v5_refit.csv');sample=pd.read_csv(ROOT/'data/sample.csv');te=pd.read_csv(ROOT/'data/test.csv');threshold=.6257334898278559
rows=[]
for name,w in [('main',.25),('alternative',.1)]:
 pred=((1-w)*b+w*q>=threshold).astype(int);s=sample.copy();s['Predicted']=pred;s.to_csv(out/f'submission_v6_{name}.csv',index=False);ix=np.flatnonzero(pred!=old.Predicted.to_numpy());pd.DataFrame({'Id':ix,'url':te.url.iloc[ix].to_numpy(),'V5':old.Predicted.iloc[ix].to_numpy(),'V6':pred[ix]}).to_csv(out/f'changes_{name}.csv',index=False)
 rows.append({'file':f'submission_v6_{name}.csv','full_refit_weight':w,'changes_vs_v5':len(ix),'public_f1':None,'local_f1':None})
report={'method':'3 full-training V1 models, seeds 42 43 44, per-model ranks averaged; blended with confirmed V2 score','train_rows':len(a['y']),'threshold':threshold,'candidates':rows,'best_confirmed_public':.995076,'best_confirmed_file':'improved_v5/submission_v5_refit.csv','validation':'Same cross-fitted training features and fixed recipe as V5; no independent OOF score for final refit. Weights motivated by V5 reported result; unknown private performance.'};(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
