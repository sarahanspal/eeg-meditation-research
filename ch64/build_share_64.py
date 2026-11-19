#build the _final share folder (nothing old is overwritten)
#
#writes to results/ch64/share/:
#paper_numbers_ch64_final.csv: long format, one row per claim x channel_set (14ch / 64ch).
#  the 14ch rows are copied unchanged from paper_numbers_ch64.csv (which holds the
#  values from results/combined_*.csv); the 64ch rows are read from the _final files.
#headline_changes_final.csv: every headline number that moved by more than 0.01
#  against results/ch64/backup_before_final/
#summary_final/, per_subject_final/, figures_final/ (150 dpi pngs)
#
#run from the repo root, after finalize_64.py and figures_64.py --suffix _final:
#.venv/bin/python ch64/build_share_64.py
import shutil

import numpy as np
import pandas as pd
from PIL import Image

from common_64 import OUT

SHARE = OUT / 'share'
BACKUP = OUT / 'backup_before_final'
CAL_SECONDS = 32   #seconds per class used for the single-amount calibration claims (64 s in total)
NAN = np.nan


def rd(name):
    return pd.read_csv(OUT / name)


def vals(n=NAN, mean=NAN, lo=NAN, hi=NAN, p=NAN, pf=NAN, src='', note=''):
    return dict(N=n, mean=mean, ci_low=lo, ci_high=hi, p=p, p_fdr=pf, source_file=src, note=note)


def pick(df, **where):
    #the one row that matches, or None
    m = df
    for k, v in where.items():
        m = m[m[k] == v]
    return m.iloc[0] if len(m) else None


def from_acc(r, src, note=''):
    #a summary row with n_subjects / mean / ci / p_vs_chance / p_fdr_bh
    if r is None:
        return vals(src=src, note='not available')
    return vals(int(r.n_subjects), r['mean'], r.ci_low, r.ci_high, r.p_vs_chance,
                r.get('p_fdr_bh', NAN), src, note)


def row64(claim, old):
    #the 64-channel values for one claim, from the _final files
    f = lambda name: f'results/ch64/{name}'
    c = claim

    if c.startswith('LOSO_control'):
        r = pick(rd('population_summary_final.csv'), contrast='IMW2_vs_IMW1', model='logistic')
        return from_acc(r, f('population_summary_final.csv'))
    if c.startswith('LOSO_'):
        _, ds, scheme = c.split('_')
        r = pick(rd('population_summary_final.csv'), contrast='MED_vs_MW', dataset=ds, scheme=scheme, model='logistic')
        return from_acc(r, f('population_summary_final.csv'))
    if c.startswith('N_combined_loso'):
        ds = c.split('_')[-1]
        r = pick(rd('population_summary_final.csv'), contrast='MED_vs_MW', dataset=ds, scheme='within', model='logistic')
        return vals(int(r.n_subjects), r.n_subjects, src=f('population_summary_final.csv'))
    if c.startswith('learning_curve'):
        ds, n = c.split('_')[2:]
        r = pick(rd('population_learning_curve_summary_final.csv'), dataset=ds, n_train_subjects=int(n))
        return from_acc(r, f('population_learning_curve_summary_final.csv'))
    if c.startswith('dataset_identity_probe'):
        i = 1 if 'himalayan' in c else 0
        r = rd('style_dataset_probe_med.csv').iloc[i]
        return vals(int(r.n_subjects), r.value, src=f('style_dataset_probe_med.csv'), note='MED epochs only, version-2 subjects')
    if c.startswith('sign_'):
        ch, band = c.split('_')[1:3]
        what = 'fracpos' if c.endswith('fracpos') else 'median_d'
        r = pick(rd('theta_sign_consistency.csv'), cohort='pooled', feature=f'{ch}_{band}')
        mean = r.frac_positive if what == 'fracpos' else r.median_standardized_difference
        return vals(int(r.n), mean, p=r.p_binomial, pf=r.p_fdr_bh, src=f('theta_sign_consistency.csv'),
                    note='64-ch FDR family = 64 channels x bands')
    if c.startswith('theta_vs_accuracy'):
        parts = c.split('_')   #theta_vs_accuracy_frontal_theta_braboszcz
        r = pick(rd('population_theta_vs_accuracy_final.csv'), measure=f'{parts[3]}_{parts[4]}', cohort=parts[5])
        return vals(int(r.n_subjects), r.spearman_rho, p=r.p, src=f('population_theta_vs_accuracy_final.csv'))
    if c.startswith('theta_ctr_vs_meditators'):
        r = pick(rd('theta_ctr_vs_meditators.csv'), measure='frontal_theta' if c.endswith('frontal') else 'midline_theta')
        return vals(f'{r.n_a}+{r.n_b}', r.mean_difference, p=r.p_permutation, src=f('theta_ctr_vs_meditators.csv'),
                    note=f'rank-biserial = {r.rank_biserial:.3f}, Mann-Whitney p = {r.p_mannwhitney:.3f}')
    if c.startswith('recording_order'):
        r = pick(rd('theta_recording_order_tests.csv'), measure='frontal_theta' if 'frontal' in c else 'midline_theta')
        return vals(f'{r.n_a}+{r.n_b}', r.mean_difference, p=r.p_permutation, src=f('theta_recording_order_tests.csv'),
                    note=f'Mann-Whitney p = {r.p_mannwhitney:.3f}')
    if c.startswith('theta_'):
        _, ds, ch, band, _ = c.split('_')
        r = pick(rd('theta_named_channels.csv'), cohort=ds, feature=f'{ch}_{band}')
        return vals(int(r.n), r.frac_positive, p=r.p_binomial, pf=r.p_fdr_bh, src=f('theta_named_channels.csv'),
                    note=f'median d = {r.median_standardized_difference:.3f}')
    if c.startswith('trait_'):
        t = rd('theta_trait_like_for_like.csv')
        r = pick(t, source='64-channel', subjects='all', outcome='d14_frontal')
        return vals(f'{r.n_meditators}+{r.n_ctr}', r.cohens_d, r.d_ci_low, r.d_ci_high, r.p_mannwhitney,
                    src=f('theta_trait_like_for_like.csv'),
                    note='cohen d of the 14-channel outcome (mean of F3/F4/F7/F8) on 64-channel data, meditators minus CTR; p = Mann-Whitney')
    if c.startswith('clf_friedman'):
        _, _, contrast, ds = c.split('_', 3)[0], None, None, None
        key = c[len('clf_friedman_'):]
        contrast, ds = key.rsplit('_', 1)
        r = pick(rd('classifier_comparison_friedman_final.csv'), contrast=contrast, dataset=ds)
        return vals(int(r.n_subjects), r.chi2, p=r.p, src=f('classifier_comparison_friedman_final.csv'), note=r.mean_ranks)
    if c.startswith('clf_'):
        key = c[len('clf_'):]
        for contrast in ('IMW2_vs_IMW1', 'MED_vs_MW'):
            if key.startswith(contrast + '_'):
                ds, model = key[len(contrast) + 1:].split('_', 1)
        r = pick(rd('classifier_comparison_summary_final.csv'), contrast=contrast, dataset=ds, model=model)
        return from_acc(r, f('classifier_comparison_summary_final.csv'))
    if c.startswith('style_transfer'):
        r = pick(rd('style_transfer_brandmeyer_final.csv'), model='logistic')
        return vals(int(r.n_brandmeyer_subjects), r.share_himalayan, p=r.p_binomial_vs_third,
                    src=f('style_transfer_brandmeyer_final.csv'), note='binomial vs 1/3')
    if c.startswith('style_'):
        analysis = c[len('style_'):-len('_logistic')]
        r = pick(rd('style_summary_final.csv'), analysis=analysis, model='logistic')
        return vals(int(r.n_subjects), r.macro_bal_acc, r.ci_low, r.ci_high, r.p_vs_chance, r.p_fdr_bh,
                    f('style_summary_final.csv'), 'macro balanced accuracy over subjects')
    if c.startswith('calibration_null_adjusted_gain'):
        g = rd('calibration_meditation_specific_gain_final.csv')
        r = pick(g, seconds_per_class=CAL_SECONDS, gain='gain_full')
        return vals(int(r.n_subjects), r.mean_difference, r.ci_low, r.ci_high, r.p_signflip, r.p_fdr_bh,
                    f('calibration_meditation_specific_gain_final.csv'),
                    f'{CAL_SECONDS} s per class; other amounts in the source file')
    if c.startswith('calibration_'):
        task = 'MED_vs_IMW2' if 'MED_vs_IMW2' in c else 'IMW2_vs_IMW1'
        measure = c.split(task.lower().replace('med_vs_imw2', 'x'), 1)[-1] if False else None
        for m in ('gain_full', 'full_update', 'bias_only', 'base'):
            if c.endswith(m):
                measure = m
        if 'think2_vs_think1' in c:
            task = 'IMW2_vs_IMW1'
        s = pick(rd('calibration_summary_final.csv'), task=task, dataset='braboszcz', seconds_per_class=CAL_SECONDS)
        if measure == 'gain_full':
            return vals(int(s.n_subjects), s.gain_full_mean, s.gain_full_ci_low, s.gain_full_ci_high, s.gain_full_p,
                        src=f('calibration_summary_final.csv'), note=f'{CAL_SECONDS} s per class')
        return vals(int(s.n_subjects), s[measure], src=f('calibration_summary_final.csv'),
                    note=f'{CAL_SECONDS} s per class; no per-measure ci in the summary')
    if c.startswith('aperiodic_'):
        f_path = OUT / 'aperiodic_sign_consistency.csv'
        if not f_path.exists():
            return vals(src=f('aperiodic_sign_consistency.csv'), note='aperiodic run not finished')
        r = pick(pd.read_csv(f_path), feature=c[len('aperiodic_'):])
        return vals(int(r.n), r.frac_positive, p=r.p_binomial, pf=r.p_fdr_bh, src=f('aperiodic_sign_consistency.csv'),
                    note=f'median d = {r.median_standardized_difference:.3f}')
    return vals(note='no 64-ch number')


def long_table():
    wide = pd.read_csv(SHARE / 'paper_numbers_ch64.csv')
    rows = []
    for _, o in wide.iterrows():
        base = dict(claim_id=o.claim_id, analysis=o.analysis, dataset=o.dataset, metric=o.metric)
        old_note = o.note if isinstance(o.note, str) and '14-ch' in o.note else ''
        rows.append({**base, 'channel_set': '14ch',
                     **vals(o.N_14ch, o.mean_14ch, o.ci_low_14ch, o.ci_high_14ch, o.p_14ch, o.p_fdr_14ch,
                            o.source_file_14ch if isinstance(o.source_file_14ch, str) else '', old_note)})
        try:
            new = row64(o.claim_id, o)
        except Exception as e:
            new = vals(note=f'lookup failed: {e}')
        rows.append({**base, 'channel_set': '64ch', **new})
    cols = ['claim_id', 'channel_set', 'analysis', 'dataset', 'metric', 'N', 'mean', 'ci_low', 'ci_high',
            'p', 'p_fdr', 'source_file', 'note']
    out = pd.DataFrame(rows)[cols]

    #extra rows that only exist in the final run
    extra = []
    g = rd('calibration_meditation_specific_gain_final.csv')
    for _, r in g[g.gain == 'gain_full'].iterrows():
        extra.append(dict(claim_id=f'calibration_null_adjusted_gain_{int(r.seconds_per_class)}s', channel_set='64ch',
                          analysis='calibration null-adjusted gain', dataset='braboszcz',
                          metric='gain(MED/IMW2) - gain(IMW2/IMW1), full update',
                          **vals(int(r.n_subjects), r.mean_difference, r.ci_low, r.ci_high, r.p_signflip, r.p_fdr_bh,
                                 'results/ch64/calibration_meditation_specific_gain_final.csv',
                                 f'{int(r.seconds_per_class)} s per class')))
    d = rd('classifier_real_minus_control_final.csv')
    for _, r in d[d.model != 'dummy'].iterrows():
        extra.append(dict(claim_id=f'clf_real_minus_control_{r.model}', channel_set='64ch',
                          analysis='classifier real minus control (MED_vs_MW - IMW2_vs_IMW1)', dataset='braboszcz',
                          metric='mean paired difference in balanced accuracy',
                          **vals(int(r.n_subjects), r.mean_difference, r.ci_low, r.ci_high, r.p_signflip, r.p_fdr_bh,
                                 'results/ch64/classifier_real_minus_control_final.csv')))
    return pd.concat([out, pd.DataFrame(extra)[cols]], ignore_index=True)


def headline_changes():
    #compare the old and the final summaries, report moves above 0.01
    pairs = [('population_summary', 'population_summary_final', ['contrast', 'dataset', 'scheme', 'model'], 'mean'),
             ('classifier_comparison_summary', 'classifier_comparison_summary_final', ['contrast', 'dataset', 'model'], 'mean'),
             ('style_summary', 'style_summary_final', ['analysis', 'model'], 'macro_bal_acc'),
             ('population_learning_curve_summary', 'population_learning_curve_summary_final', ['dataset', 'n_train_subjects'], 'mean'),
             ('theta_named_channels', 'theta_named_channels', ['cohort', 'feature'], 'frac_positive'),
             ('theta_ctr_vs_meditators', 'theta_ctr_vs_meditators', ['measure'], 'mean_difference')]
    rows = []
    for old_name, new_name, keys, col in pairs:
        old = pd.read_csv(BACKUP / f'{old_name}.csv')
        new = rd(f'{new_name}.csv')
        m = old.merge(new, on=keys, suffixes=('_old', '_new'))
        for _, r in m.iterrows():
            if abs(r[f'{col}_new'] - r[f'{col}_old']) > 0.01:
                rows.append(dict(file=new_name, key=' / '.join(str(r[k]) for k in keys), old=r[f'{col}_old'],
                                 new=r[f'{col}_new'], change=r[f'{col}_new'] - r[f'{col}_old']))
    return pd.DataFrame(rows)


def copy_files():
    groups = {
        'summary_final': ['population_summary_final', 'population_learning_curve_summary_final',
                          'population_theta_vs_accuracy_final', 'classifier_comparison_summary_final',
                          'classifier_comparison_friedman_final', 'classifier_real_minus_control_final',
                          'style_summary_final', 'style_confusion_matrices_final', 'style_transfer_brandmeyer_final',
                          'style_v2_accuracy_by_source_final', 'calibration_summary_final',
                          'calibration_meditation_specific_gain_final', 'cohort_final_ns', 'cohort_64',
                          'theta_named_channels', 'theta_sign_consistency', 'theta_ctr_vs_meditators',
                          'theta_recording_order_tests', 'theta_recording_order_groups',
                          'theta_trait_like_for_like', 'aperiodic_sign_consistency', 'aperiodic_alignment_check'],
        'per_subject_final': ['population_subject_results_final', 'population_learning_curve_subject_final',
                              'classifier_comparison_subject_final', 'style_subject_predictions_final',
                              'calibration_subject_reps_final', 'theta_shift_subject'],
    }
    for folder, names in groups.items():
        (SHARE / folder).mkdir(exist_ok=True)
        for n in names:
            if (OUT / f'{n}.csv').exists():
                shutil.copy(OUT / f'{n}.csv', SHARE / folder / f'{n}.csv')
            else:
                print(f'  missing, not copied: {n}.csv')
    (SHARE / 'figures_final').mkdir(exist_ok=True)
    for png in sorted((OUT / 'figures').glob('*_final.png')):
        im = Image.open(png)
        im.resize((im.width // 2, im.height // 2), Image.LANCZOS).save(SHARE / 'figures_final' / png.name, dpi=(150, 150))


def main():
    table = long_table()
    table.to_csv(SHARE / 'paper_numbers_ch64_final.csv', index=False)
    bad = table[table.note.astype(str).str.startswith(('lookup failed', 'not available'))]
    print(f'paper_numbers_ch64_final.csv: {len(table)} rows, {len(bad)} lookups failed')
    print(bad[['claim_id', 'note']].to_string(index=False))
    changes = headline_changes()
    changes.to_csv(SHARE / 'headline_changes_final.csv', index=False)
    print(f'\nheadline numbers that moved by more than 0.01: {len(changes)}')
    print(changes.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    copy_files()
    print('\ncopied files to', SHARE)


if __name__ == '__main__':
    main()
