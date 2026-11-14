import warnings, numpy as np, pandas as pd, sys
sys.path.insert(0,'.')
import combined_analysis as ca
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
d,feats=ca.load_data()
X=d[feats].to_numpy(float)
print('n rows',len(d),'n feats',len(feats))
rows=[]
Xs=StandardScaler().fit_transform(np.nan_to_num(X,nan=0.0))
for j,f in enumerate(feats):
    x=X[:,j]; s=Xs[:,j]
    rows.append(dict(feature=f,min=np.nanmin(x),max=np.nanmax(x),n_nan=int(np.isnan(x).sum()),n_abs_gt1e3_log=int((np.abs(x)>1e3).sum()),n_abs_gt1e3_scaled=int((np.abs(s)>1e3).sum()),max_abs_scaled=np.abs(s).max(),std=np.nanstd(x),constant=np.nanstd(x)==0))
r=pd.DataFrame(rows); r.to_csv('/private/tmp/claude-501/-Users-sarshanspal-eeg-meditation-research/1ad11732-dd04-4b5d-9a8d-b78f3650f2f7/scratchpad/a2_feature_profile.csv',index=False)
print(r.describe().T[['min','max']].to_string())
print('constant cols',r[r.constant].feature.tolist())
print('any |x|>1e3 after log',r.n_abs_gt1e3_log.sum(),' after scaler',r.n_abs_gt1e3_scaled.sum())
print(r.sort_values('max_abs_scaled',ascending=False).head(10).to_string())
print(r.sort_values('min').head(8)[['feature','min','max']].to_string())
print(r.sort_values('max',ascending=False).head(8)[['feature','min','max']].to_string())
# which feature families
import collections
print(collections.Counter(('asym' if f.startswith('asym') else 'ratio' if 'ratio' in f.lower() or '_over_' in f or '/' in f else 'IAF' if f.endswith('IAF') else 'band') for f in feats))
print(feats[:3],[f for f in feats if f.startswith('asym')][:3],[f for f in feats if not f.endswith('IAF') and not f.startswith('asym') and f.split('_')[-1] not in ('delta','theta','alpha','beta','gamma','low_alpha','high_alpha','lowbeta','highbeta')][:5])
