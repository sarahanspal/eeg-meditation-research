#script 1: per-subject theta shift on 64 channels
#
#for every subject: standardized difference (med minus mind wandering) for every
#channel x band, plus a frontal mean (F3, F4, Fz, F7, F8) and a midline mean
#(Fz, FCz, Cz). mind wandering is imw2 for braboszcz and mw for brandmeyer.
#
#then:
#sign consistency per channel x band (binomial test, bh-fdr), per dataset and pooled
#ctr vs matched meditators on the frontal theta shift (mann-whitney + permutation)
#recording order: med-first vs imw-first in the matched cohort
#
#the per-subject table (theta_shift_subject.csv) is read by population_64.py.
#
#run from the repo root:
#.venv/bin/python ch64/theta_shift_64.py
#.venv/bin/python ch64/theta_shift_64.py --n-brab 3 --n-brand 2 --n-ctr 3   (test run)
import argparse

import numpy as np
import pandas as pd
from scipy.stats import binomtest, mannwhitneyu

from common_64 import (BANDS, FRONTAL, MIDLINE, MW_LABEL, add_fdr, add_subset_args,
                       band_columns, load_from_args, output_dir, print_table,
                       standardized_differences)

N_PERMUTATIONS = 10000
SEED = 42

#the two summary measures used in the group comparisons
REGION_MEASURES = ['frontal_theta', 'midline_theta']
NAMED_ROWS = ['Fz_theta', 'FCz_theta', 'Cz_theta', 'frontal_theta', 'midline_theta']


def add_region_means(d):
    #per epoch mean of the log relative power over the frontal and midline channels
    region_cols = []
    for band in BANDS:
        d[f'frontal_{band}'] = d[[f'{ch}_{band}' for ch in FRONTAL]].mean(axis=1)
        d[f'midline_{band}'] = d[[f'{ch}_{band}' for ch in MIDLINE]].mean(axis=1)
        region_cols += [f'frontal_{band}', f'midline_{band}']
    return d, region_cols


def subject_shifts(d, cols):
    #one row per subject: standardized med minus mind wandering difference per column
    rows = []
    for subj, g in d.groupby('subj'):
        dataset = g.dataset.iloc[0]
        med = g[g.condition == 'MED']
        mw = g[g.condition == MW_LABEL[dataset]]
        info = dict(subj=subj, subject=g.subject.iloc[0], dataset=dataset,
                    style=g['style'].iloc[0], order=g['order'].iloc[0] if 'order' in g else None,
                    age=g.age.iloc[0], n_med=len(med), n_mw=len(mw))
        diffs = standardized_differences(med, mw, cols)
        rows.append({**info, **diffs.to_dict()})
    return pd.DataFrame(rows)


def sign_consistency(shifts, cols, cohort):
    #fraction of subjects with a positive shift, binomial test against 0.5
    rows = []
    for c in cols:
        v = shifts[c].dropna().to_numpy()
        if len(v) == 0:
            continue
        n_pos = int((v > 0).sum())
        rows.append(dict(cohort=cohort, feature=c, channel=c.split('_')[0], band=c.split('_')[1],
                         n=len(v), frac_positive=n_pos / len(v),
                         median_standardized_difference=float(np.median(v)),
                         p_binomial=binomtest(n_pos, len(v), 0.5).pvalue))
    return pd.DataFrame(rows)


def permutation_test(a, b, n_perm=N_PERMUTATIONS, seed=SEED):
    #two-sided permutation test on the difference in means, shuffling group labels
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    observed = a.mean() - b.mean()
    both = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(both)
        diff = both[:len(a)].mean() - both[len(a):].mean()
        if abs(diff) >= abs(observed):
            count += 1
    return observed, (count + 1) / (n_perm + 1)


def compare_groups(a, b, name_a, name_b, measure):
    #mann-whitney u, rank-biserial effect size and permutation p for two groups of subjects
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    b = np.asarray(b, float)
    b = b[np.isfinite(b)]
    row = dict(measure=measure, group_a=name_a, group_b=name_b, n_a=len(a), n_b=len(b),
               median_a=np.median(a) if len(a) else np.nan,
               median_b=np.median(b) if len(b) else np.nan)
    if len(a) == 0 or len(b) == 0:
        return row
    u, p_mw = mannwhitneyu(a, b, alternative='two-sided')
    mean_diff, p_perm = permutation_test(a, b)
    row.update(U=u, rank_biserial=2 * u / (len(a) * len(b)) - 1, p_mannwhitney=p_mw,
               mean_difference=mean_diff, p_permutation=p_perm)
    return row


def group_sign_row(values, group, measure):
    #fraction positive, median and binomial p for one group of subjects
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return dict(measure=measure, group=group, n=0)
    n_pos = int((v > 0).sum())
    return dict(measure=measure, group=group, n=len(v), frac_positive=n_pos / len(v),
                median=float(np.median(v)), p_binomial=binomtest(n_pos, len(v), 0.5).pvalue)


def main():
    parser = argparse.ArgumentParser()
    add_subset_args(parser)
    args = parser.parse_args()
    out = output_dir(args)

    d, feats = load_from_args(args, include_ctr=True)
    #take the channel x band columns before adding the region means,
    #so the regions stay out of the 384-test fdr family
    band_cols = band_columns(d)
    d, region_cols = add_region_means(d)

    #per-subject shift table, everyone including ctr
    shifts = subject_shifts(d, band_cols + region_cols)
    shifts.to_csv(out / 'theta_shift_subject.csv', index=False)
    print(f'subjects: {len(shifts)}  ({shifts.groupby(["dataset", "style"]).size().to_dict()})')

    #sign consistency, meditators only (ctr are not part of these cohorts)
    meditators = shifts[shifts['style'] != 'CTR']
    cohorts = {
        'braboszcz': meditators[meditators.dataset == 'braboszcz'],
        'brandmeyer': meditators[meditators.dataset == 'brandmeyer'],
        'pooled': meditators,
    }
    tables = []
    for name, s in cohorts.items():
        #fdr over the 384 channel x band tests, and separately over the 12 region tests
        chan = add_fdr(sign_consistency(s, band_cols, name), 'p_binomial')
        chan['family'] = 'channel_x_band'
        reg = add_fdr(sign_consistency(s, region_cols, name), 'p_binomial')
        reg['family'] = 'region'
        tables += [chan, reg]
    signs = pd.concat(tables, ignore_index=True)
    signs.to_csv(out / 'theta_sign_consistency.csv', index=False)

    named = signs[signs.feature.isin(NAMED_ROWS)].copy()
    named['feature'] = pd.Categorical(named.feature, NAMED_ROWS)
    named = named.sort_values(['cohort', 'feature'])
    named.to_csv(out / 'theta_named_channels.csv', index=False)
    print_table('theta shift at Fz, FCz, Cz and the two regions', named.drop(columns=['channel', 'band']))

    top = signs[signs.family == 'channel_x_band'].sort_values('p_binomial')
    print_table('strongest channel x band effects (pooled)', top[top.cohort == 'pooled'].head(10))

    #ctr vs matched braboszcz meditators
    brab = shifts[shifts.dataset == 'braboszcz']
    rows = []
    for m in REGION_MEASURES:
        rows.append(compare_groups(brab.loc[brab['style'] != 'CTR', m], brab.loc[brab['style'] == 'CTR', m],
                                   'matched_meditators', 'CTR', m))
    ctr = pd.DataFrame(rows)
    ctr.to_csv(out / 'theta_ctr_vs_meditators.csv', index=False)
    print_table('ctr vs matched meditators', ctr)

    #recording order, matched cohort only
    matched = brab[brab['style'] != 'CTR']
    rows, tests = [], []
    for m in REGION_MEASURES:
        for order in ['MED-first', 'IMW-first']:
            rows.append(group_sign_row(matched.loc[matched.order == order, m], order, m))
        tests.append(compare_groups(matched.loc[matched.order == 'MED-first', m],
                                    matched.loc[matched.order == 'IMW-first', m],
                                    'MED-first', 'IMW-first', m))
    order_groups = pd.DataFrame(rows)
    order_tests = pd.DataFrame(tests)
    order_groups.to_csv(out / 'theta_recording_order_groups.csv', index=False)
    order_tests.to_csv(out / 'theta_recording_order_tests.csv', index=False)
    print_table('recording order: per group', order_groups)
    print_table('recording order: med-first vs imw-first', order_tests)
    print(f'\nwrote results to {out}')


if __name__ == '__main__':
    main()
