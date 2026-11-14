"""A1: CSP + LDA / logistic LOSO on the HARMONIZED Braboszcz MED vs IMW2 epochs (matched 54-subject cohort).
Epochs come from csp_harmonized_epochs.py (same preprocessing and epoching as the harmonized features).
CSP is fit inside each training fold only. Metric: balanced accuracy per held-out subject, then averaged.
Folds: LeaveOneGroupOut over sorted subject ids (same fold structure as csp_harness.py).
Usage: venv/bin/python csp_harmonized_loso.py [--limit N] [--jobs J] [--out PATH]
"""
import argparse, time
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from braboszcz_classifier_cohort import selected_subjects
from combined_supplementary import bootstrap_ci, signflip

CACHE = Path('results/harmonized/epoch_cache')

def load(limit=None):
    subs = sorted(s for s in selected_subjects() if (CACHE/f'{s}.npz').exists())
    if limit: subs = subs[:limit]
    X, y, g = [], [], []
    for s in subs:
        z = np.load(CACHE/f'{s}.npz')
        for lab, key in [(1, 'MED'), (0, 'IMW2')]:
            X.append(z[key]); y += [lab]*len(z[key]); g += [s]*len(z[key])
    return np.concatenate(X), np.array(y), np.array(g), subs

def one_fold(X, y, g, held):
    te = g == held; tr = ~te
    csp = CSP(n_components=6, reg='ledoit_wolf', log=True, norm_trace=False, rank='full')
    Ftr = csp.fit_transform(X[tr].astype(np.float64), y[tr]); Fte = csp.transform(X[te].astype(np.float64))
    rows = []
    for name, clf in [('LDA', LinearDiscriminantAnalysis()), ('logreg', LogisticRegression(max_iter=2000))]:
        pred = clf.fit(Ftr, y[tr]).predict(Fte)
        rows.append(dict(dataset='braboszcz', subject=held, classifier=name, n_test_MED=int((y[te] == 1).sum()),
                         n_test_IMW2=int((y[te] == 0).sum()), frac_pred_MED=float(pred.mean()),
                         bal_acc=balanced_accuracy_score(y[te], pred)))
    return rows

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--limit', type=int); ap.add_argument('--jobs', type=int, default=3)
    ap.add_argument('--out', default='results/harmonized/csp_harmonized_loso_subject.csv'); a = ap.parse_args()
    X, y, g, subs = load(a.limit); print(f'{len(subs)} subjects, {len(y)} epochs, X {X.shape}', flush=True)
    t0 = time.time()
    # Checkpoint every fold to a partial CSV so a killed run resumes instead of restarting.
    part = Path(a.out).with_suffix('.partial.csv')
    done = pd.read_csv(part) if part.exists() else pd.DataFrame()
    todo = [s for s in subs if done.empty or s not in set(done.subject)]
    print(f'{len(done.subject.unique()) if not done.empty else 0} folds already done, {len(todo)} to run', flush=True)
    for s in todo:
        rows = one_fold(X, y, g, s); done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
        done.to_csv(part, index=False)
        print(f'FOLD {s} LDA bal_acc={rows[0]["bal_acc"]:.3f} ({len(done.subject.unique())}/{len(subs)}, {(time.time()-t0)/60:.1f} min)', flush=True)
    df = done; print(df.to_string(index=False)); print(f'{(time.time()-t0)/60:.1f} min')
    if Path(a.out).exists() and not a.limit: raise SystemExit(f'{a.out} exists; not overwriting')
    df.to_csv(a.out, index=False)
    for c, gdf in df.groupby('classifier'):
        v = gdf.bal_acc.to_numpy(); lo, hi = bootstrap_ci(v)
        print(f'{c}: N={len(v)} mean={v.mean():.3f} 95%CI [{lo:.3f},{hi:.3f}] p_signflip(vs .5)={signflip(v):.4g} min={v.min():.3f} n<0.4={int((v<.4).sum())}')
    print('wrote', a.out)
