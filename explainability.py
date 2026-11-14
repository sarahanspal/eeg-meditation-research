from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
try:
    import statsmodels.formula.api as smf
    from statsmodels.stats.multitest import multipletests
except ImportError:
    smf = None
    multipletests = None

from calibration import CFG, load_features

def subject_table(df, feats, summary_per=None, meta=None, top_k=20):
    X = df[feats].values.astype(float); y = df.y.values; subj = df.subject.values
    Z = (X - X.mean(0)) / (X.std(0) + 1e-12); subs = sorted(df.subject.unique()); D = {}
    for s in subs:                                    # per-subject Cohen's d (MED - non-MED) for every feature
        m = subj == s; a, b = Z[m & (y == 1)], Z[m & (y == 0)]
        D[s] = (a.mean(0) - b.mean(0)) / (np.sqrt((a.var(0, ddof=1) + b.var(0, ddof=1)) / 2) + 1e-12)
    Dm = pd.DataFrame(D).T; rows = []
    for s in subs:
        m = subj == s; others = Dm.drop(index=s).mean(0).values; ds = Dm.loc[s].values
        top = np.argsort(-np.abs(others))[:top_k]
        rows.append(dict(subject=s, n_epochs=int(m.sum()), minority_frac=min(y[m].mean(), 1 - y[m].mean()),
                         shift_rms=float(np.sqrt((Z[m].mean(0) ** 2).mean())),          # distance from the population
                         mean_abs_d=float(np.abs(ds).mean()),                           # within-subject separability
                         cos_align=float(ds @ others / (np.linalg.norm(ds) * np.linalg.norm(others) + 1e-12)),  # direction agreement with everyone else
                         sign_agree_topk=float((np.sign(ds[top]) == np.sign(others[top])).mean())))
    T = pd.DataFrame(rows).set_index("subject")
    if summary_per is not None:
        p = summary_per[summary_per["mode"] == "pooled+cal"].groupby(["subject", "frac"]).bal_acc.mean().unstack()
        T["loso_bal_acc"] = p[0.0]
        for f in (0.05, 0.10, 0.20):
            if f in p.columns: T[f"gain_{int(f*100)}"] = p[f] - p[0.0]
        T["curve_auc"] = p.apply(lambda r: np.trapz(r.values, r.index.values), axis=1)
    return T.join(meta, how="left") if meta is not None else T

def correlations(T, target, predictors):
    rows = []
    for c in predictors:
        x = T[[c, target]].dropna()
        if len(x) < 8 or x[c].nunique() < 3: continue
        r, p = spearmanr(x[c], x[target]); rows.append(dict(predictor=c, n=len(x), rho=r, p=p))
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["predictor", "n", "rho", "p", "p_fdr"])
    out = out.sort_values("p")
    if multipletests is not None:
        out["p_fdr"] = multipletests(out.p, method="fdr_bh")[1]
    else:
        # Benjamini-Hochberg adjusted p-values, avoiding a hard statsmodels dependency.
        ranked = out.p.values * len(out) / np.arange(1, len(out) + 1)
        out["p_fdr"] = np.minimum.accumulate(ranked[::-1])[::-1].clip(0, 1)
    return out.round(4)

def regression(T, target, predictors, factor=None):
    if smf is None:
        raise ImportError("Regression requires statsmodels; install it with `python -m pip install statsmodels`.")
    d = T[[target] + predictors + ([factor] if factor else [])].dropna().copy()
    for c in predictors: d[c] = (d[c] - d[c].mean()) / d[c].std()
    f = f"{target} ~ " + " + ".join(predictors) + (f" + C({factor})" if factor else "")
    return smf.ols(f, d).fit(cov_type="HC3")

def main():
    df, feats = load_features(CFG)
    per = pd.read_csv("calibration_braboszcz.csv")
    evaluated_subjects = set(per.subject.unique())
    df = df[df.subject.isin(evaluated_subjects)].reset_index(drop=True)

    # Use participant demographics when available; otherwise retain the style
    # already stored in the feature table and run the non-demographic analysis.
    participants = Path("participants.tsv")
    if participants.exists():
        p = pd.read_csv(participants, sep="\t").rename(
            columns={"participant_id": "subject", "group": "style"}
        )
        meta_cols = [c for c in ("style", "age", "gender", "years_of_practice", "sleep", "education") if c in p]
        meta = p.set_index("subject")[meta_cols]
    elif "style" in df.columns:
        meta = df.groupby("subject").style.first().to_frame()
    else:
        meta = None

    T = subject_table(df, feats, per, meta)
    predictors = ["n_epochs", "minority_frac", "shift_rms", "mean_abs_d", "cos_align", "sign_agree_topk"]
    predictors += [c for c in ("age", "years_of_practice") if c in T.columns]
    for target in ("loso_bal_acc", "gain_10", "curve_auc"):
        print(f"\n=== Spearman vs {target} ===\n", correlations(T, target, predictors))

    regression_predictors = [c for c in ("minority_frac", "shift_rms", "cos_align", "years_of_practice") if c in T.columns]
    factor = "style" if "style" in T.columns else None
    if regression_predictors and smf is not None:
        print(regression(T, "loso_bal_acc", regression_predictors, factor=factor).summary())
    elif regression_predictors:
        print("Skipping regression: install statsmodels to enable it.")
    T.to_csv("gapB_subject_table.csv")


if __name__ == "__main__":
    main()
