import sys, warnings, numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
z=np.load(sys.argv[1]+'/mat.npz'); X,y,s,ds=z['X'],z['y'],z['s'],z['ds']
def clf(): return Pipeline([('imp',SimpleImputer(strategy='median',keep_empty_features=True)),('vt',VarianceThreshold(0.0)),('sc',StandardScaler()),('clf',LogisticRegression(C=.1,max_iter=2000,random_state=42,solver="liblinear"))])
def cw(dsx,yx):
    import collections; c=collections.Counter(zip(dsx,yx)); w=np.array([1.0/c[k] for k in zip(dsx,yx)]); return w/w.mean()
print('numpy',np.__version__); nw=0; out=[]
for subj in ['braboszcz_sub-006','braboszcz_sub-007','brandmeyer_sub-001']:
    te=s==subj
    for scheme,tr in {'pooled':~te,'within':(~te)&(ds==ds[te][0]),'cross':ds!=ds[te][0]}.items():
        w=cw(ds[tr],y[tr])
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always'); m=clf().fit(X[tr],y[tr],sc__sample_weight=w,clf__sample_weight=w); dv=m.decision_function(X[te]); p=m.predict(X[te])
        nw+=len(rec); out.append((subj,scheme,round(balanced_accuracy_score(y[te],p),6),float(dv.sum())))
print('total FP warnings:',nw)
for o in out: print(o)
