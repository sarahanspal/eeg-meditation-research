"""C2: three paper figures (PDF + PNG) in results/figures/. Numbers come only from harmonized/matched-cohort result files.
Palette = first three slots of the reference categorical palette (blue, orange, aqua; validated all-pairs in light mode by the skill's
documentation). Identity is also carried by direct text labels on the axes, never colour alone. Diverging scale: blue - neutral grey - orange.
Usage: venv/bin/python make_figures.py [fig1] [fig2] [fig3]   (default: all that have their inputs)
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from combined_supplementary import bootstrap_ci

R = Path('results'); H = R/'harmonized'; F = R/'figures'; F.mkdir(exist_ok=True)
BLUE, ORANGE, AQUA, INK, MUTED, GRID, MID = '#2a78d6', '#eb6834', '#1baf7a', '#0b0b0b', '#52514e', '#e4e3df', '#ecebe8'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.linewidth': .8, 'pdf.fonttype': 42, 'font.family': 'DejaVu Sans'})
def save(fig, name):
    for ext in ('pdf', 'png'): fig.savefig(F/f'{name}.{ext}', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig); print('saved', name)

def fig1():
    bc = pd.read_csv(R/'combined_blocked_controls.csv'); lo = pd.read_csv(R/'combined_loso_subject_results.csv'); lo = lo[(lo.dataset == 'braboszcz') & (lo.model == 'logistic')]
    cal = pd.read_csv(H/'calibration_decomposition_subject.csv'); cal = cal[cal.task == 'MED_vs_IMW2']; null = pd.read_csv(H/'calibration_null_adjusted_gain.csv').null_adjusted_gain
    rungs = [('Blocked MED vs IMW2 (within subject)', bc[bc.task == 'MED_vs_IMW'].bal_acc, BLUE), ('Think1 vs think2 (file control)', bc[bc.task == 'think2_vs_think1'].bal_acc, ORANGE),
             ('Think2 first vs second half (drift)', bc[bc.task == 'think2_first_vs_second_half'].bal_acc, ORANGE), ('LOSO within-dataset', lo[lo.scheme == 'within'].bal_acc, AQUA),
             ('LOSO pooled', lo[lo.scheme == 'pooled'].bal_acc, AQUA), ('LOSO cross-dataset', lo[lo.scheme == 'cross'].bal_acc, AQUA), ('Calibration, full update (~56 s)', cal.full_update, AQUA)]
    fig, ax = plt.subplots(figsize=(7.2, 3.9)); rng = np.random.default_rng(0)
    for i, (lab, v, c) in enumerate(rungs):
        v = v.to_numpy(); y = len(rungs)-1-i; m = v.mean(); l, h = bootstrap_ci(v)
        ax.scatter(v, y+rng.uniform(-.22, .22, len(v)), s=9, color=c, alpha=.35, linewidths=0, zorder=2)
        ax.plot([l, h], [y, y], color=INK, lw=2, solid_capstyle='round', zorder=3); ax.scatter([m], [y], s=40, color=c, edgecolor='white', linewidths=1.2, zorder=4)
        ax.text(1.02, y, f'{m:.3f} [{l:.3f}, {h:.3f}]  n={len(v)}', va='center', fontsize=8, color=MUTED, transform=ax.get_yaxis_transform())
    ax.axvline(.5, color=MUTED, ls='--', lw=1, zorder=1); ax.set_yticks(range(len(rungs))); ax.set_yticklabels([r[0] for r in rungs][::-1])
    ax.set_xlim(.1, 1.0); ax.set_xlabel('Balanced accuracy (per subject; mean with bootstrap 95% CI)'); ax.xaxis.grid(True, color=GRID, lw=.6); ax.set_axisbelow(True)
    ax.text(.5, -.75, 'chance', ha='center', fontsize=8, color=MUTED)
    ax.set_title('Meditation decoding after each control (Braboszcz, matched cohort)', fontsize=9, loc='left', color=INK); ax.set_ylim(-.9, len(rungs)-.4)
    fig.text(.01, -.02, f'Calibration null-adjusted gain (MED/IMW2 minus think2/think1): {null.mean():+.3f}', fontsize=8, color=MUTED)
    save(fig, 'fig1_accuracy_ladder')

def fig2():
    import mne
    S = pd.read_csv(R/'combined_feature_sign_consistency.csv'); th = S[S.feature.str.fullmatch(r'[A-Z]{1,2}\d_theta')].copy(); th['ch'] = th.feature.str.replace('_theta', '', regex=False)
    chs = [c for c in th.ch]; info = mne.create_info(chs, 256., 'eeg'); info.set_montage('standard_1020', on_missing='ignore')
    cmap = LinearSegmentedColormap.from_list('div', [BLUE, MID, ORANGE]); vmax = float(np.abs(th.median_standardized_difference).max())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={'width_ratios': [1, 1.5], 'wspace': .75})
    im, _ = mne.viz.plot_topomap(th.median_standardized_difference.to_numpy(), info, axes=a1, cmap=cmap, vlim=(-vmax, vmax), names=[c for c in chs], show=False, sensors=True, contours=0, extrapolate='head', sphere=None, size=1)
    cb = fig.colorbar(im, ax=a1, orientation='horizontal', fraction=.05, pad=.08); cb.set_label('Median d (MED - non-MED)', fontsize=8); cb.outline.set_linewidth(.5)
    a1.set_title('Theta effect by channel\n(interpolated between 14 sensors)', fontsize=9)
    top = S.sort_values('p_binomial').head(15).iloc[::-1]; sig = (top.p_fdr_bh < .05).to_numpy()
    col = [(ORANGE if f > .5 else BLUE) if s else 'white' for f, s in zip(top.frac_positive, sig)]; a2.barh(range(len(top)), top.frac_positive, height=.62, color=col, edgecolor=[c if c != 'white' else MUTED for c in col], linewidth=1)
    for i, (f, s) in enumerate(zip(top.frac_positive, sig)): a2.text(f+.01, i, f'{f:.2f}' + (' *' if s else ''), va='center', fontsize=8, color=INK)
    a2.axvline(.5, color=MUTED, ls='--', lw=1); a2.set_yticks(range(len(top))); a2.set_yticklabels(top.feature); a2.set_xlim(0, 1.0); a2.set_xlabel('Fraction of subjects with positive effect')
    a2.xaxis.grid(True, color=GRID, lw=.6); a2.set_axisbelow(True); n = int(top.n.max())
    a2.set_title(f'Top 15 features by sign test (n = {n} subjects)\nFilled + * = FDR < 0.05: orange higher in MED, blue lower; open = not significant', fontsize=9, loc='left')
    save(fig, 'fig2_theta_sign_consistency')

def fig3():
    S = pd.read_csv(H/'calibration_curve_summary.csv'); fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for task, lab, c, ls in [('MED_vs_IMW2', 'MED vs IMW2', BLUE, '-'), ('think2_vs_think1', 'think2 vs think1 (control)', ORANGE, '-')]:
        g = S[S.task == task].sort_values('calibration_seconds'); x = g.calibration_seconds
        ax.fill_between(x, g.ci_low, g.ci_high, color=c, alpha=.15, linewidth=0); ax.plot(x, g.mean_gain_full, color=c, lw=2, ls=ls, marker='o', ms=5, mfc=c, mec='white', mew=1)
        ax.text(x.iloc[-1]+4, g.mean_gain_full.iloc[-1], lab, color=INK, fontsize=8, va='center')
    ax.axhline(0, color=MUTED, ls='--', lw=1); ax.set_xlabel('Calibration data (seconds)'); ax.set_ylabel('Gain in balanced accuracy\n(full update minus population model)')
    ax.yaxis.grid(True, color=GRID, lw=.6); ax.set_axisbelow(True); ax.set_xlim(right=S.calibration_seconds.max()*1.5)
    ax.set_title('Calibration gain vs amount of calibration data (mean, bootstrap 95% CI band)', fontsize=9, loc='left'); save(fig, 'fig3_calibration')

if __name__ == '__main__':
    want = sys.argv[1:] or ['fig1', 'fig2', 'fig3']
    for k, fn, need in [('fig1', fig1, None), ('fig2', fig2, None), ('fig3', fig3, H/'calibration_curve_summary.csv')]:
        if k in want:
            if need is not None and not need.exists(): print(f'{k}: waiting for {need.name}'); continue
            fn()
