"""Fixed-parameter diverse trees on independently cross-fitted v1 features."""
from pathlib import Path
import sys,json,time
import numpy as np
from catboost import CatBoostClassifier
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from improve_v1 import best
out=ROOT/'improved_v3';out.mkdir(exist_ok=True)
y=np.load(ROOT/'improved_v1/predictions.npz')['y'];nt=16000
for name,depth,iters,lr in [('shallow',5,2000,.045),('deep',8,1600,.035)]:
 oof=np.zeros(len(y));test=np.zeros(nt)
 for fold in range(5):
  path=out/f'{name}_fold_{fold}.cbm';a=np.load(ROOT/f'improved_v1/features_fold_{fold}.npz');t=time.time()
  m=CatBoostClassifier(depth=depth,iterations=iters,learning_rate=lr,l2_leaf_reg=7,loss_function='Logloss',random_seed=173+fold,thread_count=6,verbose=400,allow_writing_files=False)
  if path.exists():m.load_model(str(path))
  else:m.fit(a['train'],y[a['itr']]);m.save_model(str(path))
  oof[a['iva']]=m.predict_proba(a['valid'])[:,1];test+=m.predict_proba(a['test'])[:,1]/5
  print(name,'fold',fold+1,'seconds',time.time()-t,flush=True)
  del a,m
 f,th=best(y,oof);np.savez(out/f'{name}_predictions.npz',y=y,oof=oof,test=test)
 (out/f'{name}_report.json').write_text(json.dumps({'oof_f1':f,'threshold':th,'depth':depth,'iterations':iters,'learning_rate':lr,'public_f1':None},indent=2));print(name,f,flush=True)
