import re, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

def parse(f):
    if f.startswith("asym_"):  fam, ch, band = "asymmetry", f[5:].rsplit("_", 1)[0], f.rsplit("_", 1)[1]
    elif f.endswith("_IAF"):   fam, ch, band = "IAF", f[:-4], "alpha"
    elif "_ratio_" in f:       fam, ch, band = "ratio", f.split("_ratio_")[0], "/".join(f.split("_ratio_")[1].split("_"))
    else:                      fam, ch, band = "band power", *f.rsplit("_", 1)
    return dict(feature=f, family=fam, channel=ch, band=band)

def build_groups(feats):
    meta = pd.DataFrame([parse(f) for f in feats]); G = {}
    for b in sorted(meta[meta.family.isin(["band power", "asymmetry"])].band.unique()):
        G[f"band: {b}"] = np.where(meta.band.eq(b) & meta.family.isin(["band power", "asymmetry"]))[0]
    for fam in meta.family.unique(): G[f"family: {fam}"] = np.where(meta.family.eq(fam))[0]
    for ch in sorted({c for c in meta.channel if "-" not in c}):
        G[f"channel: {ch}"] = np.where(meta.channel.apply(lambda c: ch in c.split("-")))[0]
    return G, meta

def blocked_split(y, order, frac=0.7, buffer=2):
    tr, te = [], []
    for c in (0, 1):
        i = np.where(y == c)[0]; i = i[np.argsort(order[i])]; k = int(len(i) * frac)
        tr += list(i[:k]); te += list(i[k + buffer:])
    return np.array(tr), np.array(te)

def fit_score(Xtr, ytr, Xte, yte, cols):
    m = make_pipeline(VarianceThreshold(0.0), StandardScaler(), LinearSVC(C=1.0, dual=False, max_iter=5000, random_state=42))
    return balanced_accuracy_score(yte, m.fit(Xtr[:, cols], ytr).predict(Xte[:, cols]))

def _job(s, X, y, subj, order, G, do_loso, n_folds):
    from threadpoolctl import threadpool_limits
    rows = []
    with threadpool_limits(limits=1):
        m = subj == s; Xs, ys, os_ = X[m], y[m], order[m]; Xo, yo = X[~m], y[~m]
        splits = {"intra 10-fold": list(StratifiedKFold(n_folds).split(Xs, ys)),
                  "intra blocked (train early, test late)": [blocked_split(ys, os_)]}
        allc = np.arange(X.shape[1]); cfgs = [("all features", "all", allc)]
        for g, cols in G.items(): cfgs += [(g, "only", cols), (g, "drop", np.setdiff1d(allc, cols))]
        for g, kind, cols in cfgs:
            for reg, sp in splits.items():
                rows.append(dict(subject=s, group=g, kind=kind, regime=reg,
                                 bal_acc=np.mean([fit_score(Xs[a], ys[a], Xs[b], ys[b], cols) for a, b in sp])))
            if do_loso:
                rows.append(dict(subject=s, group=g, kind=kind, regime="cross-subject (LOSO)", bal_acc=fit_score(Xo, yo, Xs, ys, cols)))
    return rows

def run_attribution(df, feats, do_loso=False, n_folds=10, n_jobs=-2):
    G, meta = build_groups(feats); X = df[feats].values.astype(float)
    out = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(_job)(s, X, df.y.values, df.subject.values, df.epoch_idx.values, G, do_loso, n_folds)
        for s in sorted(df.subject.unique()))
    return pd.DataFrame([r for rows in out for r in rows]), meta

def summarize(res):
    full = res[res.group == "all features"].set_index(["subject", "regime"]).bal_acc
    r = res[res.group != "all features"].copy(); r["full"] = [full[(s, g)] for s, g in zip(r.subject, r.regime)]
    r["loss_vs_full"] = r.full - r.bal_acc
    return r.groupby(["regime", "kind", "group"]).agg(bal_acc=("bal_acc", "mean"), loss_vs_full=("loss_vs_full", "mean"), n=("subject", "nunique")).reset_index()

def univariate_map(df, feats):
    A = pd.DataFrame({s: {f: roc_auc_score(d.y, d[f]) - 0.5 for f in feats} for s, d in df.groupby("subject")})
    return pd.DataFrame({"within_strength": A.abs().median(1),
                         "direction_consistency": np.abs(np.sign(A).mean(1)),
                         "population_auc_shift": A.mean(1)}), A

def plot_attribution(summ, umap, meta):
    import matplotlib.pyplot as plt
    regs = list(summ.regime.unique()); fig, ax = plt.subplots(1, len(regs) + 1, figsize=(4.2 * (len(regs) + 1), 7))
    for a, reg in zip(ax, regs):
        d = summ[(summ.regime == reg) & (summ.kind == "only")].sort_values("bal_acc")
        a.barh(d.group, d.bal_acc, color="#0072B2"); a.axvline(.5, ls="--", c="gray"); a.set_xlim(.4, 1)
        a.set_title(f"Feature group alone\n{reg}", fontsize=9); a.set_xlabel("Balanced accuracy")
    u = umap.join(meta.set_index("feature")); a = ax[-1]
    for fam, c in zip(u.family.unique(), ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]):
        s = u[u.family == fam]; a.scatter(s.direction_consistency, s.within_strength, s=14, color=c, alpha=.7, label=fam)
    a.set_xlabel("Direction consistency across subjects"); a.set_ylabel("Within-subject strength |AUC-0.5|")
    a.legend(frameon=False, fontsize=8); a.set_title("Per-feature map", fontsize=9)
    plt.tight_layout(); plt.show()

from calibration import CFG, load_features
df, feats = load_features(CFG)
keep = pd.read_csv("braboszcz_intra_subject_accuracies.csv").subject.unique()
df = df[df.subject.isin(keep)].reset_index(drop=True)

res, meta = run_attribution(df, feats, do_loso=False)       # intra only: minutes
summ = summarize(res); umap, A = univariate_map(df, feats)
res.to_csv("attribution_intra.csv", index=False)
plot_attribution(summ, umap, meta)
summ[summ.kind == "only"].sort_values(["regime", "bal_acc"], ascending=[True, False])