"""Full-data LightGBM refit; preserve confirmed V10 weights and threshold."""
from pathlib import Path
import ctypes,json,time,hashlib
ROOT=Path(__file__).resolve().parents[1];ctypes.CDLL(str(ROOT/'.venv/lib/python3.9/site-packages/sklearn/.dylibs/libomp.dylib'),mode=ctypes.RTLD_GLOBAL)
import numpy as np,pandas as pd,joblib
from lightgbm import LGBMClassifier
from scipy.stats import rankdata
out=ROOT/'improved_v11';out.mkdir(exist_ok=True)
hpath=ROOT/'reports/public_scores_all.json';h=json.loads(hpath.read_text());h['scores']['improved_v10/submission_v10.csv']=.995404;h['best']='improved_v10/submission_v10.csv';hpath.write_text(json.dumps(h,indent=2));rpath=ROOT/'improved_v10/report.json';r=json.loads(rpath.read_text());r['public_f1']=.995404;rpath.write_text(json.dumps(r,indent=2))
a=np.load(ROOT/'improved_v5/full_features.npz');assert np.array_equal(a['y'],np.load(ROOT/'improved_v10/lightgbm_predictions.npz')['y']);P=[]
for seed in [3042,3043,3044]:
 t=time.time();path=out/f'full_seed_{seed}.joblib'
 if path.exists():m=joblib.load(path)
 else:
  m=LGBMClassifier(n_estimators=1200,learning_rate=.035,num_leaves=31,max_depth=-1,min_child_samples=50,reg_lambda=8,reg_alpha=.1,colsample_bytree=.85,subsample=.85,subsample_freq=1,n_jobs=6,random_state=seed,verbosity=-1);m.fit(a['train'],a['y']);joblib.dump(m,path,compress=3)
 P.append(m.predict_proba(a['test'])[:,1]);print('full seed',seed,'seconds',round(time.time()-t,1),flush=True)
P=np.column_stack(P);np.savez(out/'full_predictions.npz',test=P);nt=len(P);fullrank=np.mean(np.column_stack([rankdata(P[:,j])/nt for j in range(P.shape[1])]),axis=1)
b=np.load(ROOT/'improved_v2/blend_predictions.npz')['test'];tree=np.load(ROOT/'improved_v5/full_predictions.npz')['test'];foldlgb=np.load(ROOT/'improved_v10/lightgbm_predictions.npz')['test'];foldrank=rankdata(foldlgb)/nt;v5=.9*b+.1*rankdata(tree)/nt;v10=.8*v5+.2*foldrank;threshold=.6250374960876368
old=pd.read_csv(ROOT/'improved_v10/submission_v10.csv');assert np.array_equal((v10>=threshold).astype(int),old.Predicted.to_numpy());sample=pd.read_csv(ROOT/'data/sample.csv');te=pd.read_csv(ROOT/'data/test.csv');rows=[]
for name,replaceweight in [('half',.1),('full',.2)]:
 score=v10+replaceweight*(fullrank-foldrank);pred=(score>=threshold).astype(int);s=sample.copy();s['Predicted']=pred;filename=f'submission_v11_{name}.csv';s.to_csv(out/filename,index=False);ix=np.flatnonzero(pred!=old.Predicted.to_numpy());pd.DataFrame({'Id':ix,'url':te.url.iloc[ix].to_numpy(),'V10':old.Predicted.iloc[ix].to_numpy(),'V11':pred[ix]}).to_csv(out/f'changes_{name}.csv',index=False);rows.append({'file':filename,'replaced_weight':replaceweight,'changes_vs_v10':len(ix),'public_f1':None,'local_f1':None})
report={'method':'Three full-data LightGBM refits, rank averaging; replace half or all of 20 percent LightGBM component in confirmed V10; preserve remaining weights and threshold','seeds':[3042,3043,3044],'train_rows':len(a['y']),'threshold':threshold,'candidates':rows,'best_confirmed_public':.995404,'public_f1':None,'validation':'LightGBM recipe validated previously on five outer folds in V10; all-label final refit has no independent OOF. Training label-derived features cross-fitted; no test labels. Replacement fractions fixed before public evaluation.'};(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
