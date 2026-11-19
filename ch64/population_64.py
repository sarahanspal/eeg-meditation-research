#script 2: population (leave-one-subject-out) classifiers on 64 channels
#
#logistic regression with the same clf() settings as combined_analysis.py:
#med vs mind wandering: within-dataset, pooled (equal weight per dataset x state)
#and cross-dataset, using combined_analysis.evaluate()
#imw2 vs imw1 label control: braboszcz, within-dataset only
#learning curve: 3, 6, 12, 24, 36 training subjects, 5 random draws each
#spearman: per-subject frontal / midline theta shift vs per-subject accuracy
#
#needs theta_shift_subject.csv from theta_shift_64.py for the spearman part.
#
#run from the repo root:
#.venv/bin/python ch64/population_64.py
#.venv/bin/python ch64/population_64.py --n-brab 3 --n-brand 2   (test run)
import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

from common_64 import (STYLES, add_fdr, add_subset_args, cell_weights, clf,
                       imw2_vs_imw1, load_from_args, med_vs_mw, output_dir, print_table,
                       subject_summary)
from combined_analysis import evaluate

TRAIN_SIZES = (3, 6, 12, 24, 36)
N_DRAWS = 5
SEED = 42


def summarize(res):
    #one row per contrast x dataset x scheme x model, subject is the unit
    rows = []
    for keys, g in res.groupby(['contrast', 'dataset', 'scheme', 'model']):
        per_subject = g.groupby('subject').bal_acc.mean()
        rows.append({**dict(zip(['contrast', 'dataset', 'scheme', 'model'], keys)),
                     **subject_summary(per_subject)})
    summary = pd.DataFrame(rows)
    #fdr only over the logistic rows, the dummy is just a sanity check
    logistic = summary.model == 'logistic'
    summary.loc[logistic, 'p_fdr_bh'] = add_fdr(summary[logistic], 'p_vs_chance').p_fdr_bh
    return summary


def learning_curve(d, feats, sizes=TRAIN_SIZES, draws=N_DRAWS, seed=SEED):
    #same design as combined_supplementary.learning_curve, on the 64-channel data:
    #train on n random subjects from the held-out subject's own dataset
    #(braboszcz draws an equal number per style), test on the held-out subject
    X = d[feats].to_numpy(float)
    y = d.y.to_numpy()
    s = d.subj.to_numpy()
    info = d[['subj', 'dataset', 'style']].drop_duplicates('subj').set_index('subj')
    rng = np.random.default_rng(seed)
    rows = []

    for held in info.index:
        te = s == held
        if np.unique(y[te]).size < 2:
            continue
        dataset = info.loc[held, 'dataset']
        pool = [x for x in info.index if x != held and info.loc[x, 'dataset'] == dataset]

        for n in sizes:
            if n > len(pool):
                continue
            if dataset == 'braboszcz':
                #n has to split evenly over the three styles
                if n % 3:
                    continue
                by_style = {st: [x for x in pool if info.loc[x, 'style'] == st] for st in STYLES}
                if any(len(v) < n // 3 for v in by_style.values()):
                    continue

            for draw in range(draws):
                if dataset == 'braboszcz':
                    chosen = np.concatenate([rng.choice(by_style[st], n // 3, replace=False)
                                             for st in STYLES])
                else:
                    chosen = rng.choice(pool, n, replace=False)
                tr = np.isin(s, chosen)
                if np.unique(y[tr]).size < 2:
                    continue
                w = cell_weights(d.iloc[np.flatnonzero(tr)])
                model = clf().fit(X[tr], y[tr], sc__sample_weight=w, clf__sample_weight=w)
                rows.append(dict(subject=held, dataset=dataset, n_train_subjects=n, draw=draw,
                                 bal_acc=balanced_accuracy_score(y[te], model.predict(X[te]))))
    return pd.DataFrame(rows)


def learning_curve_summary(lc):
    if len(lc) == 0:
        return pd.DataFrame()
    #average the draws first so each subject counts once per training size
    per = lc.groupby(['dataset', 'n_train_subjects', 'subject']).bal_acc.mean().reset_index()
    rows = []
    for (dataset, n), g in per.groupby(['dataset', 'n_train_subjects']):
        rows.append(dict(dataset=dataset, n_train_subjects=n, **subject_summary(g.bal_acc)))
    return pd.DataFrame(rows)


def theta_vs_accuracy(res, shifts):
    #spearman between the per-subject theta shift and the within-dataset logistic accuracy
    acc = res[(res.contrast == 'MED_vs_MW') & (res.scheme == 'within') & (res.model == 'logistic')]
    acc = acc.groupby(['subject', 'dataset']).bal_acc.mean().reset_index()
    merged = acc.merge(shifts, left_on='subject', right_on='subj', suffixes=('', '_shift'))
    rows = []
    for measure in ['frontal_theta', 'midline_theta']:
        for name, g in [('braboszcz', merged[merged.dataset == 'braboszcz']),
                        ('brandmeyer', merged[merged.dataset == 'brandmeyer']),
                        ('pooled', merged)]:
            g = g[[measure, 'bal_acc']].dropna()
            row = dict(measure=measure, cohort=name, n_subjects=len(g))
            if len(g) >= 3:
                rho, p = spearmanr(g[measure], g.bal_acc)
                row.update(spearman_rho=rho, p=p)
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    add_subset_args(parser)
    args = parser.parse_args()
    out = output_dir(args)
    threadpool_limits(limits=1)

    d, feats = load_from_args(args)
    main_task = med_vs_mw(d)
    control = imw2_vs_imw1(d)
    print(f'features: {len(feats)}  subjects: {main_task.subj.nunique()}  epochs: {len(main_task)}')

    #med vs mind wandering: within, pooled and cross
    res_main = evaluate(main_task, feats)
    res_main['contrast'] = 'MED_vs_MW'
    #label control: one dataset, so "pooled" is the same fit as "within" and
    #"cross" has no training data, keep only within
    res_ctrl = evaluate(control, feats)
    res_ctrl = res_ctrl[res_ctrl.scheme == 'within']
    res_ctrl['contrast'] = 'IMW2_vs_IMW1'
    res = pd.concat([res_main, res_ctrl], ignore_index=True)
    res.to_csv(out / 'population_subject_results.csv', index=False)

    summary = summarize(res)
    summary.to_csv(out / 'population_summary.csv', index=False)
    print_table('loso summary (balanced accuracy, subject = unit)', summary)

    #learning curve on med vs mind wandering
    lc = learning_curve(main_task, feats)
    lc.to_csv(out / 'population_learning_curve_subject.csv', index=False)
    lc_summary = learning_curve_summary(lc)
    lc_summary.to_csv(out / 'population_learning_curve_summary.csv', index=False)
    print_table('learning curve', lc_summary)

    #theta shift vs accuracy
    shift_file = out / 'theta_shift_subject.csv'
    if shift_file.exists():
        corr = theta_vs_accuracy(res, pd.read_csv(shift_file))
        corr.to_csv(out / 'population_theta_vs_accuracy.csv', index=False)
        print_table('spearman: theta shift vs within-dataset accuracy', corr)
    else:
        print(f'\n{shift_file} not found, run theta_shift_64.py first for the spearman part')
    print(f'\nwrote results to {out}')


if __name__ == '__main__':
    main()
