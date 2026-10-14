"""Refit established V1 recipe on all labels; training features cross-fitted."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=(ROOT/'tools/improve_v1.py').read_text()
s=s[:s.index('    outer=list(')]
s=s.replace("out=ROOT/'improved_v1'","out=ROOT/'improved_v5'")
s+='''    def features(reference,query):
        allowed=np.zeros(n,dtype=np.int16);allowed[reference]=1
        chunks=[base[query]]
        for idx,sim in views.values():
            b,_=nn_feature_block(idx[query],sim[query],y,fold_of=allowed,current_fold=0);chunks.append(b)
        for k in keys.values():chunks.append(aggregate(k,y,reference,query))
        return np.hstack(chunks).astype(np.float32)
    X=np.zeros((n,288),dtype=np.float32)
    for a,b in StratifiedKFold(5,shuffle=True,random_state=142).split(X,y):X[b]=features(a,b)
    T=features(np.arange(n),np.arange(n,n+nt))
    np.savez_compressed(out/'full_features.npz',train=X,test=T,y=y)
    print('Full training features',X.shape,flush=True)
    m=CatBoostClassifier(iterations=1400,depth=7,learning_rate=.045,l2_leaf_reg=5,loss_function='Logloss',random_seed=42,thread_count=6,verbose=500,allow_writing_files=False)
    m.fit(X,y);m.save_model(str(out/'full_model.cbm'));pred=m.predict_proba(T)[:,1];np.savez(out/'full_predictions.npz',test=pred)
    from scipy.stats import rankdata
    baseline=np.load(ROOT/'improved_v2/blend_predictions.npz');prior=pd.read_csv(ROOT/'improved_v2/submission_v2.csv');threshold=.6257334898278559
    score=.9*baseline['test']+.1*rankdata(pred)/nt
    sub=pd.read_csv(ROOT/'data/sample.csv');sub['Predicted']=(score>=threshold).astype(int)
    sub.to_csv(out/'submission_v5_refit.csv',index=False)
    report={'method':'90 percent confirmed V2 + 10 percent full-data V1 refit rank','threshold':threshold,'weight_selected_without_public_feedback':.1,'train_rows':n,'test_changes':int((sub.Predicted!=prior.Predicted).sum()),'public_f1':None,'local_f1':None,'validation':'V1 recipe validated previously; final refit has no independent OOF estimate; no training F1 reported.'}
    (out/'refit_report.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
if __name__=='__main__':main()
'''
# Execute unchanged V1 feature construction plus explicit full-refit stage.
exec(compile(s,str(__file__),'exec'))
