import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# ---- 1. load raw features ----
df = pd.read_csv("braboszcz_all_subjects_features.csv")
meta_cols = ["subject", "style", "condition", "epoch_idx"]
feature_cols = [c for c in df.columns if c not in meta_cols]

# fill NaNs with global per-feature median (same as your classification pipeline)
df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

print(f"{df.shape[0]} epochs, {len(feature_cols)} features")
print(df.groupby(["style", "condition"]).size())

# ---- 2. helper: subsample, scale, run t-SNE, return 2D embedding + labels ----
def run_tsne(sub_df, label_col, max_per_class=600, perplexity=30, seed=0):
    """Subsamples per class (t-SNE gets slow/crowded above a few thousand points),
    scales features, runs t-SNE, returns (embedding, labels, kept_index)."""
    rng = np.random.default_rng(seed)
    parts = []
    for lab, grp in sub_df.groupby(label_col):
        if len(grp) > max_per_class:
            grp = grp.iloc[rng.choice(len(grp), max_per_class, replace=False)]
        parts.append(grp)
    sampled = pd.concat(parts)

    X = StandardScaler().fit_transform(sampled[feature_cols].values)
    emb = TSNE(n_components=2, perplexity=perplexity, init="pca",
               learning_rate="auto", random_state=seed).fit_transform(X)
    return emb, sampled[label_col].values, sampled.index

# ---- 3. plotting helper ----
def plot_tsne_panel(ax, emb, labels, title, palette=None, highlight_mask=None):
    classes = sorted(pd.unique(labels))
    if palette is None:
        cmap = plt.cm.get_cmap("tab10", len(classes))
        palette = {c: cmap(i) for i, c in enumerate(classes)}

    for c in classes:
        m = labels == c
        if highlight_mask is not None:
            base = m & ~highlight_mask
            hi = m & highlight_mask
            ax.scatter(emb[base, 0], emb[base, 1], s=6, alpha=0.15,
                       color=palette[c], label=None, zorder=1)
            ax.scatter(emb[hi, 0], emb[hi, 1], s=20, alpha=0.9,
                       color=palette[c], edgecolors="black", linewidths=0.3,
                       label=c, zorder=2)
        else:
            ax.scatter(emb[m, 0], emb[m, 1], s=6, alpha=0.35, color=palette[c], label=c)

    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=7, markerscale=2, loc="upper right", framealpha=0.6)

# ---- 4. build the comparisons (mirrors Figure 5: A binary style-vs-MW, B pairwise, C 3-class, D 4-class) ----
STYLES = ["HYT", "VIP", "ISY"]
comparisons = {}

# A: each style vs mind-wandering (IMW), pooled across all subjects of that style
for style in STYLES:
    sub = df[(df["style"] == style)].copy()
    sub["label"] = sub["condition"].map({"MED": style, "IMW": "MW"})
    comparisons[f"{style} vs MW"] = sub

# B: pairwise style comparisons, MED-only
med = df[df["condition"] == "MED"].copy()
med["label"] = med["style"]
for s1, s2 in [("HYT", "VIP"), ("HYT", "ISY"), ("VIP", "ISY")]:
    sub = med[med["style"].isin([s1, s2])]
    comparisons[f"{s1} vs {s2}"] = sub

# C: 3-class, MED-only
comparisons["HYT vs VIP vs ISY"] = med

# D: 4-class, MED (all styles) vs MW
four = df.copy()
four["label"] = np.where(four["condition"] == "IMW", "MW", four["style"])
comparisons["HYT vs VIP vs ISY vs MW"] = four

# ---- 5. run + plot all panels ----
n = len(comparisons)
ncols = 4
nrows = int(np.ceil(n / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
axes = axes.flatten()

for ax, (title, sub) in zip(axes, comparisons.items()):
    emb, labels, _ = run_tsne(sub, "label", max_per_class=600, perplexity=30, seed=0)
    plot_tsne_panel(ax, emb, labels, title)

for ax in axes[len(comparisons):]:
    ax.axis("off")

fig.suptitle("t-SNE of pooled EEG features across all subjects", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("figure5_tsne_pooled.png", dpi=200, bbox_inches="tight")
plt.show()