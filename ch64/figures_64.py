#paper figures for the 64-channel analysis
#
#fig1: theta topomaps: fraction of subjects with med > mind wandering, per cohort
#figs1: the same for all six bands (supplementary)
#fig2: learning curve: within-dataset accuracy vs number of training subjects
#fig3: calibration: population vs threshold-only vs full update, and the
#meditation-specific gain (needs calibration_64.py output)
#fig4: three-class confusion matrices (logistic), braboszcz only vs with brandmeyer
#figs2: the same confusion matrices for every model (supplementary)
#
#each figure is saved as pdf (vector, editable text) and png (300 dpi) in
#results/ch64/figures. a figure whose input file is missing is skipped, so this
#script can be re-run as more results come in.
#
#run from the repo root:
#.venv/bin/python ch64/figures_64.py
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.collections
import mne
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from common_64 import BANDS, OUT, bootstrap_ci

FIG_DIR = OUT / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

#colors (reference palette from the dataviz guide, light mode)
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'
BLUE = '#2a78d6'
ORANGE = '#eb6834'
AQUA = '#1baf7a'

#diverging blue to red with a neutral gray middle (0.5 = no consistent direction)
DIVERGING = LinearSegmentedColormap.from_list(
    'blue_gray_red', ['#104281', '#3987e5', '#9ec5f4', '#f0efec', '#f3a9a8', '#e34948', '#9e2b2b'])
#sequential blue for the confusion matrices
SEQUENTIAL = LinearSegmentedColormap.from_list(
    'blue_seq', ['#f4f8fd', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])

#a full-page width figure in most journals is about 7.2 inches
FULL_WIDTH = 7.2

MODEL_LABELS = {
    'logistic': 'Logistic regression', 'gaussian_nb': 'Gaussian NB', 'mlp': 'MLP',
    'random_forest': 'Random forest', 'svm_rbf': 'SVM (RBF)', 'svm_linear': 'SVM (linear)',
    'knn': 'kNN', 'dummy': 'Chance (uniform)',
}
COHORT_LABELS = {'braboszcz': 'Braboszcz', 'brandmeyer': 'Brandmeyer', 'pooled': 'Pooled'}
BAND_LABELS = {'delta': 'Delta 1-4 Hz', 'theta': 'Theta 4-8 Hz', 'alpha': 'Alpha 8-13 Hz',
               'beta': 'Beta 13-30 Hz', 'gamma1': 'Gamma 1', 'gamma2': 'Gamma 2'}


def set_style():
    #quiet, print-friendly defaults: thin recessive axes, no top/right spines, sans text
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
        'text.color': INK, 'axes.labelcolor': INK_2, 'axes.titlecolor': INK,
        'axes.edgecolor': AXIS, 'axes.linewidth': 0.8,
        'xtick.color': AXIS, 'ytick.color': AXIS,
        'xtick.labelcolor': INK_2, 'ytick.labelcolor': INK_2,
        'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': False, 'grid.color': GRID, 'grid.linewidth': 0.6,
        'legend.frameon': False, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'lines.solid_capstyle': 'round', 'lines.solid_joinstyle': 'round',
    })


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(FIG_DIR / f'{name}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {name}.pdf / .png')


def panel_letter(ax, letter, x=-0.12, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight='bold',
            color=INK, va='bottom', ha='left')


def y_gridlines(ax):
    #hairline horizontal grid behind the data
    ax.grid(axis='y', color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def chance_line(ax, y, label='chance', x_text=None, ha='left', va='bottom'):
    #solid muted reference line with a small label
    ax.axhline(y, color=MUTED, linewidth=0.8, zorder=1)
    if label:
        x0 = ax.get_xlim()[0] if x_text is None else x_text
        ax.text(x0, y, f' {label} ', color=MUTED, fontsize=6.5, va=va, ha=ha)


#fig 1 and s1: topomaps

def eeg_info(channels):
    #channel positions from the standard 10-20 montage
    info = mne.create_info(channels, sfreq=256., ch_types='eeg')
    info.set_montage('standard_1020')
    return info


def topomap(ax, info, values, mask, vlim):
    im, _ = mne.viz.plot_topomap(
        values, info, axes=ax, show=False, cmap=DIVERGING, vlim=vlim, contours=0,
        sensors=True, mask=mask, extrapolate='head', sphere='eeglab',
        mask_params=dict(marker='o', markerfacecolor='white', markeredgecolor=INK,
                         linewidth=0, markersize=3.6, markeredgewidth=0.8),
        outlines='head')
    #make the head outline and the non-significant sensor dots recessive.
    #sensors are a scatter (collection), significant ones are drawn as lines with markers
    for coll in ax.collections:
        if isinstance(coll, matplotlib.collections.PathCollection):
            coll.set_sizes([1.5])
            coll.set_facecolor(INK_2)
            coll.set_edgecolor('none')
    for line in ax.lines:
        if line.get_marker() in ('None', None, ''):
            line.set_color(MUTED)
            line.set_linewidth(0.8)
    return im


def band_values(signs, cohort, band, channels):
    rows = signs[(signs.cohort == cohort) & (signs.family == 'channel_x_band') & (signs.band == band)]
    rows = rows.set_index('channel').reindex(channels)
    return rows.frac_positive.to_numpy(), (rows.p_fdr_bh < 0.05).to_numpy(), int(rows.n.max())


def fig_topomaps(signs):
    channels = list(signs[signs.family == 'channel_x_band'].channel.drop_duplicates())
    info = eeg_info(channels)
    cohorts = ['braboszcz', 'brandmeyer', 'pooled']
    vlim = (0.15, 0.85)

    #main figure: theta only
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH * 0.78, 2.3))
    for ax, cohort, letter in zip(axes, cohorts, 'abc'):
        values, mask, n = band_values(signs, cohort, 'theta', channels)
        im = topomap(ax, info, values, mask, vlim)
        ax.set_title(f'{COHORT_LABELS[cohort]} (n = {n})', pad=6)
        panel_letter(ax, letter, x=-0.02, y=1.02)
    cbar = fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.06, pad=0.06,
                        aspect=40, shrink=0.6, ticks=[0.2, 0.35, 0.5, 0.65, 0.8])
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0, labelsize=7)
    cbar.set_label('Fraction of subjects with higher theta in meditation than mind wandering',
                   color=INK_2, fontsize=7.5)
    fig.text(0.5, -0.06, 'White circles: channels significant after Benjamini-Hochberg FDR '
             '(binomial sign test, 384 channel x band tests per cohort).',
             ha='center', color=MUTED, fontsize=6.5)
    save(fig, 'fig1_theta_topomap')

    #supplementary: all bands
    fig, axes = plt.subplots(len(cohorts), len(BANDS), figsize=(FULL_WIDTH, 3.9))
    for i, cohort in enumerate(cohorts):
        for j, band in enumerate(BANDS):
            values, mask, n = band_values(signs, cohort, band, channels)
            im = topomap(axes[i, j], info, values, mask, vlim)
            if i == 0:
                axes[i, j].set_title(BAND_LABELS[band], pad=6)
        axes[i, 0].text(-0.28, 0.5, f'{COHORT_LABELS[cohort]}\n(n = {n})', transform=axes[i, 0].transAxes,
                        ha='center', va='center', rotation=90, color=INK, fontsize=8)
    cbar = fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.04, pad=0.04,
                        aspect=50, shrink=0.5, ticks=[0.2, 0.35, 0.5, 0.65, 0.8])
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0, labelsize=7)
    cbar.set_label('Fraction of subjects with higher power in meditation than mind wandering '
                   '(white circles: FDR < 0.05)', color=INK_2, fontsize=7.5)
    save(fig, 'figS1_all_bands_topomap')


#fig 2: learning curve

def fig_learning_curve(lc, pop):
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    colors = {'braboszcz': BLUE, 'brandmeyer': ORANGE}
    #the full leave-one-subject-out model, trained on every other subject in the dataset
    full_n = {'braboszcz': 53, 'brandmeyer': 15}
    for dataset, color in colors.items():
        g = lc[lc.dataset == dataset].sort_values('n_train_subjects')
        x = g.n_train_subjects.to_numpy()
        ax.fill_between(x, g.ci_low, g.ci_high, color=color, alpha=0.12, linewidth=0)
        ax.plot(x, g['mean'], color=color, linewidth=2, label=COHORT_LABELS[dataset])
        ax.plot(x, g['mean'], 'o', color=color, markersize=4.5, markeredgecolor='white',
                markeredgewidth=1.2, zorder=3)
        #direct label just above (braboszcz) or below (brandmeyer) the line, clear of the markers
        if dataset == 'braboszcz':
            ax.text(24, g.loc[g.n_train_subjects == 24, 'mean'].iloc[0] + 0.009, COHORT_LABELS[dataset],
                    color=INK_2, fontsize=7, ha='center', va='bottom')
        else:
            ax.text(4.4, 0.49, COHORT_LABELS[dataset], color=INK_2, fontsize=7, ha='center', va='top')
        #full loso point, hollow, with its ci
        row = pop[(pop.contrast == 'MED_vs_MW') & (pop.dataset == dataset) &
                  (pop.scheme == 'within') & (pop.model == 'logistic')].iloc[0]
        ax.errorbar(full_n[dataset], row['mean'], yerr=[[row['mean'] - row.ci_low], [row.ci_high - row['mean']]],
                    fmt='o', color=color, markerfacecolor='white', markersize=5, markeredgewidth=1.4,
                    elinewidth=1, capsize=0, zorder=3)
    ax.set_xscale('log', base=2)
    ax.set_xticks([3, 6, 12, 15, 24, 36, 53])
    ax.set_xticklabels(['3', '6', '12', '15', '24', '36', '53'])
    ax.minorticks_off()
    ax.set_xlim(2.6, 90)
    ax.set_ylim(0.40, 0.62)
    ax.set_yticks([0.40, 0.45, 0.50, 0.55, 0.60])
    y_gridlines(ax)
    chance_line(ax, 0.5, x_text=88, ha='right', va='top')
    ax.set_xlabel('Training subjects (log scale)')
    ax.set_ylabel('Balanced accuracy, held-out subject')
    ax.set_title('Meditation vs mind wandering, within dataset', loc='left')
    #legend for the two marker types; the lines are labelled directly
    ax.plot([], [], 'o', color=MUTED, markersize=4.5, markeredgecolor='white', label='random subsets (5 draws)')
    ax.plot([], [], 'o', color=MUTED, markerfacecolor='white', markersize=5, markeredgewidth=1.4,
            label='all other subjects (LOSO)')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left', ncol=2, handlelength=1.4, columnspacing=1.0,
              bbox_to_anchor=(0, 1.0))
    fig.text(0.02, -0.08, 'Lines and shading: mean and bootstrap 95% CI over held-out subjects.',
             color=MUTED, fontsize=6.5)
    save(fig, 'fig2_learning_curve')


#fig 3: calibration

def per_subject_ci(reps, cols):
    #mean over the 5 placements per subject, then mean and bootstrap ci over subjects
    per = reps.groupby(['task', 'dataset', 'seconds_per_class', 'subject'])[cols].mean().reset_index()
    rows = []
    for keys, g in per.groupby(['task', 'dataset', 'seconds_per_class']):
        for c in cols:
            v = g[c].to_numpy()
            lo, hi = bootstrap_ci(v)
            rows.append(dict(zip(['task', 'dataset', 'seconds_per_class'], keys),
                             measure=c, mean=v.mean(), ci_low=lo, ci_high=hi, n=len(v)))
    return pd.DataFrame(rows)


def fig_calibration(reps, gain):
    methods = [('base', 'Population model', BLUE), ('bias_only', 'Threshold only', ORANGE),
               ('full_update', 'Full update (weight 20)', AQUA)]
    stats = per_subject_ci(reps, [m for m, _, _ in methods])
    panels = [('MED_vs_IMW2', 'braboszcz', 'Braboszcz', 'MED vs IMW2'),
              ('IMW2_vs_IMW1', 'braboszcz', 'Braboszcz', 'control: IMW2 vs IMW1'),
              ('MED_vs_IMW2', 'brandmeyer', 'Brandmeyer', 'MED vs MW')]
    #one y range for panels a-c (same measure), never cutting off a ci
    y_low = min(0.4, np.floor(stats.ci_low.min() * 10) / 10)
    ticks = [8, 16, 32, 64, 128]

    fig, axes = plt.subplots(1, 4, figsize=(FULL_WIDTH, 2.6),
                             gridspec_kw=dict(width_ratios=[1, 1, 1, 1], wspace=0.55))
    for ax, (task, dataset, name, contrast), letter in zip(axes[:3], panels, 'abc'):
        s = stats[(stats.task == task) & (stats.dataset == dataset)]
        for measure, label, color in methods:
            g = s[s.measure == measure].sort_values('seconds_per_class')
            if g.empty:
                continue
            x = g.seconds_per_class.to_numpy()
            ax.fill_between(x, g.ci_low, g.ci_high, color=color, alpha=0.12, linewidth=0)
            ax.plot(x, g['mean'], color=color, linewidth=2, label=label)
            ax.plot(x, g['mean'], 'o', color=color, markersize=4, markeredgecolor='white',
                    markeredgewidth=1.1, zorder=3)
        n = int(s.n.max()) if len(s) else 0
        ax.set_title(f'{name} (n = {n})\n{contrast}', loc='left', fontsize=7.5)
        ax.set_xscale('log', base=2)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.minorticks_off()
        ax.set_xlim(6.5, 160)
        ax.set_ylim(y_low, 1.0)
        y_gridlines(ax)
        #the lines often sit on chance here, so the label goes in the footnote
        chance_line(ax, 0.5, label=None)
        panel_letter(ax, letter, x=-0.22, y=1.13)
    axes[0].set_ylabel('Balanced accuracy, held-out epochs')

    #panel d: meditation-specific gain
    ax = axes[3]
    for g_name, label, color in [('gain_full', 'Full update', AQUA), ('gain_bias', 'Threshold only', ORANGE)]:
        g = gain[gain.gain == g_name].sort_values('seconds_per_class')
        if g.empty:
            continue
        x = g.seconds_per_class.to_numpy()
        ax.fill_between(x, g.ci_low, g.ci_high, color=color, alpha=0.12, linewidth=0)
        ax.plot(x, g.mean_difference, color=color, linewidth=2, label=label)
        ax.plot(x, g.mean_difference, 'o', color=color, markersize=4, markeredgecolor='white',
                markeredgewidth=1.1, zorder=3)
    ax.set_xscale('log', base=2)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(6.5, 160)
    y_gridlines(ax)
    chance_line(ax, 0.0, label=None)
    ax.set_ylabel('Meditation gain minus control gain')
    n = int(gain.n_subjects.max()) if len(gain) else 0
    ax.set_title(f'Braboszcz (n = {n})\nmeditation-specific gain', loc='left', fontsize=7.5)
    panel_letter(ax, 'd', x=-0.3, y=1.13)
    ax.legend(loc='upper left', handlelength=1.4)
    fig.text(0.5, -0.02, 'Calibration data per class (s, log scale)', ha='center', color=INK_2, fontsize=8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.4, 1.12), handlelength=1.6)
    fig.text(0.02, -0.1, 'Mean over subjects of the mean over 5 calibration placements; shading: '
             'bootstrap 95% CI over subjects. Gray line: chance (a-c) or no difference (d). '
             'Gains are relative to the population model.',
             color=MUTED, fontsize=6.5)
    save(fig, 'fig3_calibration')


#fig 4 and s2: confusion matrices

def confusion_panel(ax, cm, row_names, col_names, title):
    #row-normalized fill, count and percent written in each cell
    counts = cm.to_numpy(float)
    props = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    ax.pcolormesh(props, cmap=SEQUENTIAL, vmin=0, vmax=1, edgecolors='white', linewidth=2)
    for i in range(props.shape[0]):
        for j in range(props.shape[1]):
            dark = props[i, j] > 0.55
            ax.text(j + 0.5, i + 0.5, f'{int(counts[i, j])}\n{props[i, j]:.0%}', ha='center', va='center',
                    fontsize=7, color='white' if dark else INK, linespacing=1.15)
    ax.set_xticks(np.arange(len(col_names)) + 0.5)
    ax.set_xticklabels(col_names)
    ax.set_yticks(np.arange(len(row_names)) + 0.5)
    ax.set_yticklabels(row_names)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.tick_params(length=0)
    for side in ('left', 'bottom'):
        ax.spines[side].set_visible(False)
    ax.set_title(title, fontsize=7.5)


def confusion_matrix_for(confusion, analysis, model, styles=('HYT', 'VIP', 'ISY')):
    g = confusion[(confusion.analysis == analysis) & (confusion.model == model)]
    return g.pivot(index='true', columns='predicted', values='n_subjects').reindex(index=styles, columns=styles)


def fig_confusion(confusion, summary):
    styles = ['HYT', 'VIP', 'ISY']
    runs = [('three_class_v1_braboszcz', ['HYT', 'VIP', 'ISY'], 'Braboszcz only'),
            ('three_class_v2_with_brandmeyer', ['HYT +\nBrandmeyer', 'VIP', 'ISY'], 'Brandmeyer added to Himalayan')]

    def subtitle(analysis, model):
        r = summary[(summary.analysis == analysis) & (summary.model == model)].iloc[0]
        return f'macro bal. acc. {r.macro_bal_acc:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}], n = {r.n_subjects}'

    #main figure: logistic regression
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH * 0.62, 2.3), gridspec_kw=dict(wspace=0.55))
    for ax, (analysis, rows, title), letter in zip(axes, runs, 'ab'):
        cm = confusion_matrix_for(confusion, analysis, 'logistic', styles)
        confusion_panel(ax, cm, rows, styles, f'{title}\n{subtitle(analysis, "logistic")}')
        ax.set_xlabel('Predicted style')
        panel_letter(ax, letter, x=-0.32, y=1.12)
    axes[0].set_ylabel('True style')
    fig.text(0.02, 0.0, 'Logistic regression, leave-one-subject-out; one count per subject (majority vote over '
             'their epochs). Cell shading: share of the row. Chance = 0.33.', color=MUTED, fontsize=6.5)
    save(fig, 'fig4_confusion_logistic')

    #supplementary: every model
    models = [m for m in MODEL_LABELS if m in set(confusion.model)]
    fig, axes = plt.subplots(2, len(models), figsize=(FULL_WIDTH * 1.35, 3.6),
                             gridspec_kw=dict(wspace=0.35, hspace=0.45))
    for i, (analysis, rows, title) in enumerate(runs):
        for j, model in enumerate(models):
            cm = confusion_matrix_for(confusion, analysis, model, styles)
            r = summary[(summary.analysis == analysis) & (summary.model == model)].iloc[0]
            confusion_panel(axes[i, j], cm, rows if j == 0 else [''] * 3, styles,
                            f'{MODEL_LABELS[model]}\n{r.macro_bal_acc:.2f}')
        axes[i, 0].set_ylabel(title, fontsize=7.5)
    fig.text(0.5, -0.02, 'Rows: true style; columns: predicted style (majority vote). Number under each '
             'model: macro balanced accuracy (chance = 0.33).', ha='center', color=MUTED, fontsize=6.5)
    save(fig, 'figS2_confusion_all_models')


def main():
    set_style()
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    signs_file = OUT / 'theta_sign_consistency.csv'
    if signs_file.exists():
        fig_topomaps(pd.read_csv(signs_file))

    lc_file = OUT / 'population_learning_curve_summary.csv'
    if lc_file.exists():
        fig_learning_curve(pd.read_csv(lc_file), pd.read_csv(OUT / 'population_summary.csv'))

    reps_file = OUT / 'calibration_subject_reps.csv'
    if reps_file.exists():
        fig_calibration(pd.read_csv(reps_file), pd.read_csv(OUT / 'calibration_meditation_specific_gain.csv'))
    else:
        print('calibration results not there yet, skipped fig3')

    conf_file = OUT / 'style_confusion_matrices.csv'
    if conf_file.exists():
        fig_confusion(pd.read_csv(conf_file), pd.read_csv(OUT / 'style_summary.csv'))
    print(f'figures in {FIG_DIR}')


if __name__ == '__main__':
    main()
