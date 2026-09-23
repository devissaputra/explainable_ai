from pathlib import Path
import json, numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score,roc_curve
X,y=load_breast_cancer(return_X_y=True,as_frame=True); Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=y); m=RandomForestClassifier(n_estimators=450,random_state=42,n_jobs=-1).fit(Xtr,ytr); p=m.predict_proba(Xte)[:,1]; imp=permutation_importance(m,Xte,yte,n_repeats=16,random_state=42,n_jobs=-1); order=np.argsort(imp.importances_mean)[-10:][::-1]; out={'roc_auc':float(roc_auc_score(yte,p)),'top_features':[(str(X.columns[i]),float(imp.importances_mean[i])) for i in order]}; Path('results').mkdir(exist_ok=True); Path('results/metrics.json').write_text(json.dumps(out,indent=2))
plt.figure(figsize=(8,5)); names=[X.columns[i] for i in order][::-1]; vals=[imp.importances_mean[i] for i in order][::-1]; plt.barh(names,vals); plt.xlabel('Permutation importance'); plt.title('Held-out feature importance'); plt.tight_layout(); plt.savefig('assets/03_data_or_model.png',dpi=150); plt.close()
fpr,tpr,_=roc_curve(yte,p); plt.figure(figsize=(7,5)); plt.plot(fpr,tpr); plt.plot([0,1],[0,1]); plt.xlabel('False positive rate'); plt.ylabel('True positive rate'); plt.title('Model discrimination'); plt.tight_layout(); plt.savefig('assets/04_evaluation_or_results.png',dpi=150); plt.close(); print(json.dumps(out,indent=2))