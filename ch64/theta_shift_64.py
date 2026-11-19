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
#trait like-for-like: the 14-channel trait outcome (cohen d of the per-subject
#  med - imw2 standardized difference, mean over F3/F4/F7/F8 per channel, matched
#  meditators minus ctr) on the 64-channel data, on all subjects and on the subjects
#  that are in both analyses, next to the 14-channel values on those same subjects
#
#writes cohort_64.csv: every subject, epochs per condition, and which analyses it is in.
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

from common_64 import (BANDS, FRONTAL, MIDLINE, MW_LABEL, ROOT, add_fdr, add_subset_args,
                       band_columns, load_from_args, output_dir, print_cohort, print_table,
                       standardized_differences)

N_PERMUTATIONS = 10000
N_BOOT = 5000
SEED = 42

#the 14-channel trait analysis (trait_ctr_vs_experienced.py): its frontal channels
#and its per-subject values, used for the like-for-like comparison
TRAIT_14_CHANNELS = ['F3', 'F4', 'F7', 'F8']
TRAIT_14_FILE = ROOT / 'results' / 'harmonized' / 'trait_ctr_vs_experienced_subject_values.csv'

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


def cohen_d(a, b):
    #same formula as trait_ctr_vs_experienced.cohen_d (pooled sd, ddof 1)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / sp


def trait_row(a, b, source, subjects, outcome):
    #matched meditators (a) minus ctr (b): cohen d with bootstrap ci, plus mann-whitney
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    b = np.asarray(b, float)
    b = b[np.isfinite(b)]
    row = dict(source=source, subjects=subjects, outcome=outcome, n_meditators=len(a), n_ctr=len(b))
    if len(a) < 5 or len(b) < 5:
        return row
    rng = np.random.default_rng(SEED)
    boot = [cohen_d(rng.choice(a, len(a)), rng.choice(b, len(b))) for _ in range(N_BOOT)]
    u, p = mannwhitneyu(a, b, alternative='two-sided')
    row.update(mean_meditators=a.mean(), mean_ctr=b.mean(), mean_difference=a.mean() - b.mean(),
               cohens_d=cohen_d(a, b), d_ci_low=np.percentile(boot, 2.5),
               d_ci_high=np.percentile(boot, 97.5),
               rank_biserial=2 * u / (len(a) * len(b)) - 1, p_mannwhitney=p)
    return row


def trait_like_for_like(shifts):
    #the 14-channel outcome on 64-channel data: mean over F3/F4/F7/F8 of the
    #per-channel standardized med - imw2 difference (theta)
    brab = shifts[shifts.dataset == 'braboszcz'].copy()
    brab['d14_frontal'] = brab[[f'{c}_theta' for c in TRAIT_14_CHANNELS]].mean(axis=1, skipna=False)
    brab['arm'] = np.where(brab['style'] == 'CTR', 'CTR', 'experienced')

    def rows_for(t, source, subjects, outcomes):
        return [trait_row(t.loc[t.arm == 'experienced', o], t.loc[t.arm == 'CTR', o],
                          source, subjects, o) for o in outcomes]

    #64-channel, all subjects, both outcomes (the first one is the old 64-channel number)
    rows = rows_for(brab, '64-channel', 'all', ['frontal_theta', 'd14_frontal'])
    if not TRAIT_14_FILE.exists():
        print(f'\n{TRAIT_14_FILE} not found, like-for-like rows against the 14-channel values skipped')
        return pd.DataFrame(rows)

    old = pd.read_csv(TRAIT_14_FILE)
    old = old[old.d_MED_minus_IMW2.notna()]
    new = brab[brab.d14_frontal.notna()]
    common = set(old.subject) & set(new.subject)
    #same arm in both tables, otherwise the subject is not comparable
    arm_old = dict(zip(old.subject, old.arm))
    common = {s for s in common if arm_old[s] == new.set_index('subject').arm[s]}
    rows += rows_for(old.rename(columns={'d_MED_minus_IMW2': 'd14_frontal'}), '14-channel', 'all',
                     ['d14_frontal'])
    rows += rows_for(old[old.subject.isin(common)].rename(columns={'d_MED_minus_IMW2': 'd14_frontal'}),
                     '14-channel', 'common', ['d14_frontal'])
    rows += rows_for(new[new.subject.isin(common)], '64-channel', 'common', ['d14_frontal'])
    only_old = sorted(set(old.subject) - common)
    only_new = sorted(set(new.subject) - common)
    print(f'\ntrait like-for-like: {len(common)} subjects in both, only 14-channel: {only_old}, '
          f'only 64-channel: {only_new}')
    return pd.DataFrame(rows)


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

    #who is in which analysis, same rule in every script
    cohort = print_cohort(d)
    cohort.to_csv(out / 'cohort_64.csv', index=False)

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
    missing_order = matched[matched.order.isna()].subject.tolist()
    if missing_order:
        print(f'  no recording order for: {missing_order}')

    #trait: 14-channel outcome on 64-channel data, like for like
    trait = trait_like_for_like(shifts)
    trait.to_csv(out / 'theta_trait_like_for_like.csv', index=False)
    print_table('trait like-for-like (matched meditators minus ctr)', trait)
    print(f'\nwrote results to {out}')


if __name__ == '__main__':
    main()
