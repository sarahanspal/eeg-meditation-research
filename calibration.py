import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import matplotlib as mpl
import time
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score

CFG = dict(
    FEATURES_CSV="braboszcz_all_subjects_features.csv", OUT_CSV="calibration_braboszcz.csv",
    POS_LABEL="MED", NEG_LABEL="IMW",        # Brandmeyer: "MED","MW".  Control: "IMW2","IMW1" (think2 vs think1)
    UNIT_COL="epoch_idx",                    # Brandmeyer: "probe_id"  (a probe's two epochs always travel together)
    BUFFER_UNITS=2,                          # Brandmeyer: 0
    ORDER_COL="epoch_idx",                   # Brandmeyer: "t_start"
    FRACTIONS=[0.0, 0.05, 0.10, 0.20, 0.30, 0.50],   # Brandmeyer has ~6-25 units/class: use [0, 0.25, 0.5]
    N_REPEATS=5, CAL_WEIGHT=20,
    CLASSIFIERS=["SVM-linear", "GaussianNB", "RandomForest"],
    MIN_UNITS_PER_CLASS=20,                  # Brandmeyer: ~6
    SUBJ_ZSCORE=False,                       # label-free per-subject z-scoring; embeds class imbalance, so keep off for the main result
    N_JOBS=-2, SEED=42)
META = {"subject", "style", "group", "condition", "epoch_idx", "probe_id", "sub_epoch", "t_start", "q1", "q2", "q3", "label"}

def load_features(cfg=CFG):
    df = pd.read_csv(cfg["FEATURES_CSV"])
    df = df[df.condition.isin([cfg["POS_LABEL"], cfg["NEG_LABEL"]])].copy()
    df = df.dropna(axis=1, how="all"); feats = [c for c in df.columns if c not in META]
    miss = df.groupby("subject")[feats].transform(lambda c: c.isna().any())    # same leak-safe NaN rule as your classifier
    df[feats] = df[feats].mask(miss)
    med = df[feats].median(); keep = med.dropna().index.tolist()
    df = df[[c for c in df.columns if c not in feats] + keep]; feats = keep
    df[feats] = df[feats].fillna(med[keep])
    if cfg["SUBJ_ZSCORE"]:
        g = df.groupby("subject")[feats]; df[feats] = (df[feats] - g.transform("mean")) / (g.transform("std") + 1e-12)
    df["y"] = (df.condition == cfg["POS_LABEL"]).astype(int)
    df["unit"] = df[cfg["UNIT_COL"]].values
    df = df.sort_values(["subject", cfg["ORDER_COL"]]).reset_index(drop=True)
    cnt = df.groupby(["subject", "y"]).unit.nunique().unstack(fill_value=0)
    bad = cnt[(cnt < cfg["MIN_UNITS_PER_CLASS"]).any(axis=1)].index.tolist()
    if bad: print(f"[load] dropping {len(bad)} subject(s) below {cfg['MIN_UNITS_PER_CLASS']} units/class: {bad}")
    df = df[~df.subject.isin(bad)].reset_index(drop=True)
    print(f"[load] {df.subject.nunique()} subjects, {len(df)} epochs, {len(feats)} features")
    return df, feats

def make_clf(name, seed):
    clf = {"SVM-linear": LinearSVC(C=1.0, dual=False, max_iter=5000, random_state=seed),
           "GaussianNB": GaussianNB(),
           "RandomForest": RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)}[name]
    return Pipeline([("vt", VarianceThreshold(0.0)), ("sc", StandardScaler()), ("clf", clf)])

def fit_eval(name, seed, Xtr, ytr, wtr, Xte, yte):
    pipe = make_clf(name, seed); pipe.fit(Xtr, ytr, clf__sample_weight=wtr); p = pipe.predict(Xte)
    return balanced_accuracy_score(yte, p), accuracy_score(yte, p)

def calibration_split(units_by_class, frac, rep, n_rep, buffer):
    """Per class: one contiguous block of units; units within `buffer` of the block are excluded from the test set."""
    cal, test = {}, {}
    for c, u in units_by_class.items():
        n = len(u); k = max(2, int(round(frac * n)))
        start = 0 if n_rep == 1 else int(round(rep / (n_rep - 1) * (n - k)))
        excl = np.arange(max(0, start - buffer), min(n, start + k + buffer))
        cal[c] = set(u[np.arange(start, start + k)]); test[c] = set(np.delete(u, excl))
    return cal, test

def _subject_job(s, X, y, subj, unit, cfg):
    from threadpoolctl import threadpool_limits
    rows = []
    with threadpool_limits(limits=1):
        m = subj == s; Xo, yo = X[~m], y[~m]; Xs, ys, us = X[m], y[m], unit[m]
        ubc = {c: np.unique(us[ys == c]) for c in (0, 1)}
        for name in cfg["CLASSIFIERS"]:
            for frac in cfg["FRACTIONS"]:
                for rep in ([0] if frac == 0 else range(cfg["N_REPEATS"])):
                    if frac == 0:
                        ba, acc = fit_eval(name, cfg["SEED"], Xo, yo, np.ones(len(yo)), Xs, ys)
                        rows.append(dict(subject=s, classifier=name, mode="pooled+cal", frac=0.0, rep=0, bal_acc=ba, acc=acc, n_cal=0, n_test=len(ys)))
                        continue
                    cal_u, test_u = calibration_split(ubc, frac, rep, cfg["N_REPEATS"], cfg["BUFFER_UNITS"])
                    # Unit identifiers can repeat across conditions (each run
                    # starts epoch_idx at zero), so select units within class.
                    cal_m = ((ys == 0) & np.isin(us, list(cal_u[0]))) | ((ys == 1) & np.isin(us, list(cal_u[1])))
                    te_m = ((ys == 0) & np.isin(us, list(test_u[0]))) | ((ys == 1) & np.isin(us, list(test_u[1])))
                    Xc, yc, Xt, yt = Xs[cal_m], ys[cal_m], Xs[te_m], ys[te_m]
                    if len(np.unique(yt)) < 2 or len(np.unique(yc)) < 2: continue
                    w = np.concatenate([np.ones(len(yo)), np.full(len(yc), float(cfg["CAL_WEIGHT"]))])
                    ba, acc = fit_eval(name, cfg["SEED"], np.vstack([Xo, Xc]), np.concatenate([yo, yc]), w, Xt, yt)
                    rows.append(dict(subject=s, classifier=name, mode="pooled+cal", frac=frac, rep=rep, bal_acc=ba, acc=acc, n_cal=len(yc), n_test=len(yt)))
                    ba, acc = fit_eval(name, cfg["SEED"], Xc, yc, np.ones(len(yc)), Xt, yt)          # calibration-only control
                    rows.append(dict(subject=s, classifier=name, mode="cal-only", frac=frac, rep=rep, bal_acc=ba, acc=acc, n_cal=len(yc), n_test=len(yt)))
    return rows

def run_calibration(df, feats, cfg=CFG, subjects=None):
    X = df[feats].values.astype(float); y = df.y.values; subj = df.subject.values; unit = df.unit.values
    subs = sorted(df.subject.unique()) if subjects is None else subjects
    if os.path.exists(cfg["OUT_CSV"]): os.remove(cfg["OUT_CSV"])
    out = []
    for rows in Parallel(n_jobs=cfg["N_JOBS"], backend="loky", verbose=5, return_as="generator")(
            delayed(_subject_job)(s, X, y, subj, unit, cfg) for s in subs):
        r = pd.DataFrame(rows); out.append(r)
        r.to_csv(cfg["OUT_CSV"], mode="a", header=not os.path.exists(cfg["OUT_CSV"]), index=False)   # crash-safe
    return pd.concat(out, ignore_index=True)

def summarize(res, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    per = res.groupby(["classifier", "mode", "frac", "subject"]).bal_acc.mean().reset_index()   # average repeats within subject first
    rows = []
    for (c, mo, f), d in per.groupby(["classifier", "mode", "frac"]):
        v = d.bal_acc.values; bs = rng.choice(v, (n_boot, len(v))).mean(1)
        rows.append(dict(classifier=c, mode=mo, frac=f, n_subj=len(v), mean=v.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
    return pd.DataFrame(rows), per

def main():
    # ---- RUN (Braboszcz, your 45 subjects) ----
    df, feats = load_features(CFG)
    keep = pd.read_csv("braboszcz_intra_subject_accuracies.csv").subject.unique()
    df = df[df.subject.isin(keep)].reset_index(drop=True)
    res = run_calibration(df, feats, CFG)
    summ, _ = summarize(res)

    fig, axes = plt.subplots(1, len(CFG["CLASSIFIERS"]), figsize=(5 * len(CFG["CLASSIFIERS"]), 4), sharey=True)
    for ax, classifier in zip(np.atleast_1d(axes), CFG["CLASSIFIERS"]):
        for mode, color in (("pooled+cal", "C0"), ("cal-only", "C3")):
            d = summ[(summ.classifier == classifier) & (summ["mode"] == mode)].sort_values("frac")
            ax.plot(d.frac * 100, d["mean"], "-o", color=color, label=mode)
            ax.fill_between(d.frac * 100, d.lo, d.hi, color=color, alpha=.2)
        ax.axhline(.5, ls="--", c="gray")
        ax.set_title(classifier)
        ax.set_xlabel("% of the held-out subject's data used")
        ax.legend()
    axes[0].set_ylabel("balanced accuracy (mean ± 95% CI over subjects)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
