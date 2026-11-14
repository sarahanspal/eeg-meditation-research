"""B3: Riemannian baseline (tangent space + logistic regression) vs the harmonized-feature logistic model, LOSO, matched 54-subject Braboszcz cohort.
Covariances: 14-channel per-epoch covariance (Ledoit-Wolf shrinkage) of the harmonized epochs (epoch_cache/ and epoch_cache_think1/), in microvolts.
Tangent space: reference = Riemannian mean of the TRAINING covariances in each fold. Classifier head matches combined_analysis.clf():
StandardScaler + LogisticRegression(C=0.1, liblinear), class-balanced sample weights in training. Folds: leave one subject out.
Tasks: MED_vs_IMW2 (label MED=1) and think2_vs_think1 (file-identity control, IMW2=1). Both models are scored on both tasks.
Output: results/harmonized/riemann_loso_subject.csv, riemann_vs_feature_paired.csv
Usage: venv/bin/python riemann_loso.py [--limit N] [--jobs J] [--out SUFFIX]
"""
import argparse, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score
import combined_analysis as ca
from combined_supplementary import bootstrap_ci, signflip
from braboszcz_classifier_cohort import selected_subjects

H = Path('results/harmonized'); TASKS = {'MED_vs_IMW2': ('MED', 'IMW2'), 'think2_vs_think1': ('IMW2', 'IMW1')}

def arrays(s):
    z = np.load(H/'epoch_cache'/f'{s}.npz'); out = {'MED': z['MED'], 'IMW2': z['IMW2']}
    p = H/'epoch_cache_think1'/f'{s}.npz'
    if p.exists(): out['IMW1'] = np.load(p)['IMW1']
    return out

def subject_covs(s):
    cov = Covariances(estimator='lwf'); return {k: cov.transform(v.astype(np.float64)*1e6) for k, v in arrays(s).items()}

def balanced_w(y):
    n = np.bincount(y); w = 1.0/n[y]; return w/w.mean()

def fit_predict(Xtr, ytr, Xte, tangent):
    steps = [TangentSpace(metric='riemann')] if tangent else []
    m = make_pipeline(*steps, StandardScaler(), LogisticRegression(C=.1, max_iter=2000, random_state=42, solver='liblinear'))
    w = balanced_w(ytr); m.fit(Xtr, ytr, standardscaler__sample_weight=w, logisticregression__sample_weight=w); return m.predict(Xte)

def feature_tables():
    T = pd.read_csv(H/'braboszcz_all_subjects_features.csv'); T = T[T.subject.isin(selected_subjects()) & T.condition.isin(['MED', 'IMW2', 'IMW1'])].copy()
    d, feats = ca.load_data(); feats = list(feats)
    logcols = [f for f in feats if not f.startswith('asym') and not f.endswith('IAF')]
    for f in logcols: T[f] = np.log(pd.to_numeric(T[f], errors='coerce').clip(lower=1e-12))
    T[feats] = T[feats].replace([np.inf, -np.inf], np.nan); return T, feats

def one_task(task, subs, C, F, feats, jobs):
    pos, neg = TASKS[task]; rows = []
    def fold(held):
        res = []
        tr = [s for s in subs if s != held and pos in C[s] and neg in C[s]]
        if held not in subs or pos not in C[held] or neg not in C[held] or not tr: return res
        def XY(ss): return (np.concatenate([np.concatenate([C[s][pos], C[s][neg]]) for s in ss]), np.concatenate([np.r_[np.ones(len(C[s][pos]), int), np.zeros(len(C[s][neg]), int)] for s in ss]))
        Xtr, ytr = XY(tr); Xte, yte = XY([held])
        res.append(dict(subject=held, task=task, model='riemann_tangent_logreg', n_test=len(yte), bal_acc=balanced_accuracy_score(yte, fit_predict(Xtr, ytr, Xte, True))))
        G = F[F.condition.isin([pos, neg])]; Gtr = G[G.subject.isin(tr)]; Gte = G[G.subject == held]
        from sklearn.impute import SimpleImputer; from sklearn.feature_selection import VarianceThreshold
        m = ca.clf(); ytr_f = (Gtr.condition == pos).astype(int).to_numpy(); w = balanced_w(ytr_f)
        m.fit(Gtr[feats].to_numpy(float), ytr_f, sc__sample_weight=w, clf__sample_weight=w)
        res.append(dict(subject=held, task=task, model='feature_logreg', n_test=len(Gte), bal_acc=balanced_accuracy_score((Gte.condition == pos).astype(int), m.predict(Gte[feats].to_numpy(float)))))
        return res
    out = Parallel(n_jobs=jobs)(delayed(fold)(h) for h in subs); return [r for rs in out for r in rs]

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--limit', type=int); ap.add_argument('--jobs', type=int, default=2); ap.add_argument('--out', default=''); a = ap.parse_args()
    subs = sorted(s for s in selected_subjects() if (H/'epoch_cache'/f'{s}.npz').exists()); subs = subs[:a.limit] if a.limit else subs
    t0 = time.time(); C = {s: subject_covs(s) for s in subs}; print(f'{len(subs)} subjects, covariances in {time.time()-t0:.0f}s; think1 available for {sum("IMW1" in C[s] for s in subs)}', flush=True)
    F, feats = feature_tables(); F = F[F.subject.isin(subs)]
    res = []
    for task in TASKS:
        t0 = time.time(); r = one_task(task, subs, C, F, feats, a.jobs); res += r; print(task, len(r)//2, 'subjects', f'{time.time()-t0:.0f}s', flush=True)
    R = pd.DataFrame(res); R.to_csv(H/f'riemann_loso_subject{a.out}.csv', index=False)
    rows = []
    for (task, model), g in R.groupby(['task', 'model']):
        v = g.bal_acc.to_numpy(); lo, hi = bootstrap_ci(v); rows.append(dict(task=task, model=model, N=len(v), mean=v.mean(), ci_low=lo, ci_high=hi, p_vs_chance=signflip(v)))
    S = pd.DataFrame(rows); print(S.round(4).to_string(index=False)); S.to_csv(H/f'riemann_loso_summary{a.out}.csv', index=False)
    P = []
    for task, g in R.groupby('task'):
        w = g.pivot(index='subject', columns='model', values='bal_acc').dropna(); dlt = (w.riemann_tangent_logreg-w.feature_logreg).to_numpy()
        if len(dlt) >= 3: lo, hi = bootstrap_ci(dlt); P.append(dict(task=task, N=len(dlt), mean_riemann_minus_feature=dlt.mean(), ci_low=lo, ci_high=hi, p_signflip=signflip(dlt+.5)))
    P = pd.DataFrame(P); print(P.round(4).to_string(index=False)); P.to_csv(H/f'riemann_vs_feature_paired{a.out}.csv', index=False)
