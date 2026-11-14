import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---- config ----
CLF_ORDER = ["GaussianNB", "kNN", "RandomForest", "SVM-linear", "SVM-RBF", "MLP"]
CLF_COLORS = {
    "GaussianNB":   "#e07a5f",
    "kNN":          "#81b29a",
    "RandomForest": "#3d5a80",
    "SVM-linear":   "#f2cc8f",
    "SVM-RBF":      "#6d597a",
    "MLP":          "#8ecae6",
}

def load_and_check(path):
    df = pd.read_csv(path)
    print(f"{path}: {df.shape}, cols = {df.columns.tolist()}")
    return df

# ---- load the six CSVs ----
intra_hyt = load_and_check("braboszcz_intra_subject_accuracies_HYT.csv")
intra_vip = load_and_check("braboszcz_intra_subject_accuracies_VIP.csv")
intra_isy = load_and_check("braboszcz_intra_subject_accuracies_ISY.csv")
inter_hyt = load_and_check("braboszcz_inter_subject_accuracies_HYT.csv")
inter_vip = load_and_check("braboszcz_inter_subject_accuracies_VIP.csv")
inter_isy = load_and_check("braboszcz_inter_subject_accuracies_ISY.csv")

def plot_panel(ax, df, title, clf_order=CLF_ORDER, chance=0.5, jitter=0.12, metric="accuracy"):
    """Dot-plot: per-subject accuracy scattered per classifier, mean±SD overlaid."""
    rng = np.random.default_rng(0)
    present = [c for c in clf_order if c in df["classifier"].unique()]
    xpos = np.arange(len(present))

    for i, clf in enumerate(present):
        vals = df.loc[df["classifier"] == clf, metric].dropna().values
        if len(vals) == 0:
            continue
        x_jitter = xpos[i] + rng.uniform(-jitter, jitter, size=len(vals))
        ax.scatter(x_jitter, vals, s=18, alpha=0.55, color=CLF_COLORS.get(clf, "gray"),
                   edgecolors="none", zorder=2)
        mean, sd = vals.mean(), vals.std()
        ax.errorbar(xpos[i], mean, yerr=sd, fmt="o", ms=7, mfc=CLF_COLORS.get(clf, "gray"),
                    mec="black", ecolor="black", elinewidth=1.2, capsize=4, zorder=3)

    ax.axhline(chance, color="gray", ls="--", lw=1, zorder=1)
    ax.set_xticks(xpos)
    ax.set_xticklabels(present, rotation=40, ha="right", fontsize=8)
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("Accuracy (%)" if ax.get_subplotspec().colspan.start == 0 else "")

fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)

plot_panel(axes[0, 0], intra_hyt, "HYT vs MW: Intra-subject")
plot_panel(axes[0, 1], intra_vip, "VIP vs MW: Intra-subject")
plot_panel(axes[0, 2], intra_isy, "ISY vs MW: Intra-subject")

plot_panel(axes[1, 0], inter_hyt, "HYT vs MW: Inter-subject (LOSO)")
plot_panel(axes[1, 1], inter_vip, "VIP vs MW: Inter-subject (LOSO)")
plot_panel(axes[1, 2], inter_isy, "ISY vs MW: Inter-subject (LOSO)")

fig.suptitle("Panel A : MED vs mind-wandering classification, by style", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("panelA_med_vs_mw.png", dpi=200, bbox_inches="tight")
plt.show()