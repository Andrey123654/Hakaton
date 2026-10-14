from pathlib import Path
import json
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from url_identity_v3 import lookup
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'improved_v3'
tr=pd.read_csv(ROOT/'data/train.csv');g=tr.groupby('url').result.nunique();tr=tr[~tr.url.isin(g[g>1].index)].drop_duplicates('url').reset_index(drop=True);te=pd.read_csv(ROOT/'data/test.csv');d=np.load(ROOT/'improved_v2/blend_predictions.npz');y=d['y'];old=d['oof']>=.6257334898278559;new=old.copy();matches=0
for a,b in StratifiedKFold(5,shuffle=True,random_state=42).split(y,y):
 m=lookup(tr.iloc[a],tr.iloc[b]);mask=m.notna().to_numpy();new[b[mask]]=m[mask].to_numpy();matches+=mask.sum()
m=lookup(tr,te);oldtest=pd.read_csv(ROOT/'improved_v2/submission_v2.csv').Predicted;changes=m.notna()&(m!=oldtest)
te.loc[changes].assign(known_label=m[changes],previous_label=oldtest[changes]).to_csv(out/'identity_changes.csv',index=False)
r={'oof_matches':int(matches),'baseline_oof_f1':float(f1_score(y,old)),'identity_oof_f1':float(f1_score(y,new)),'corrected':int(((old!=y)&(new==y)).sum()),'introduced':int(((old==y)&(new!=y)).sum()),'test_matches':int(m.notna().sum()),'test_changes':int(changes.sum())};(out/'identity_report.json').write_text(json.dumps(r,indent=2));print(r)
