#final step: apply the common cohort rule to the finished per-subject results
#
#no model is refit here. the per-subject files of scripts 2, 3, 4 and 6 are read, every
#subject that does not pass the cohort rule for that analysis is dropped from the
#evaluation (they stayed in training), and the summary statistics are recomputed.
#cohort rule: >= 5 epochs in every condition the analysis uses (cohort_64.csv,
#written by theta_shift_64.py from the feature csvs).
#
#the dummy classifier is set to its expected accuracy (0.5, or 1/3 for three-class)
#and its p value to 1.
#
#writes files with the suffix _final next to the old ones, nothing is overwritten:
#population_summary_final.csv, population_subject_results_final.csv,
#population_learning_curve_*_final.csv, population_theta_vs_accuracy_final.csv
#classifier_comparison_*_final.csv, classifier_real_minus_control_final.csv
#style_summary_final.csv, style_subject_predictions_final.csv,
#style_confusion_matrices_final.csv, style_transfer_brandmeyer_final.csv,
#style_v2_accuracy_by_source_final.csv
#calibration_*_final.csv (if calibration_64.py has finished)
#cohort_final_ns.csv: n subjects per analysis and dataset
#
#run from the repo root:
#.venv/bin/python ch64/finalize_64.py
#.venv/bin/python ch64/finalize_64.py --dir results/ch64/smoke   (test run)
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from calibration_64 import meditation_specific_gain, summarize as calibration_summarize
from classifier_comparison_64 import friedman, summarize as comparison_summarize
from common_64 import OUT, STYLES, add_fdr, print_table, subject_summary
from population_64 import learning_curve_summary, summarize as population_summarize
from population_64 import theta_vs_accuracy
from style_classification_64 import confusion_tables, summarize as style_summarize

#column names in cohort_64.csv
MED = 'MED_vs_MW (classification, theta shift, calibration, aperiodic)'
CONTROL = 'IMW2_vs_IMW1 (label control, braboszcz)'
STYLE_MED = 'style / meditators vs ctr on MED epochs'
STYLE_IMW2 = 'meditators vs ctr on IMW2 epochs'


def passing(cohort, column):
    return set(cohort.loc[cohort[column], 'subj'])


def keep(df, subject_col, ok):
    #keep the rows whose subject passes, report how many subjects were dropped
    before = df[subject_col].nunique()
    out = df[df[subject_col].isin(ok)].copy()
    return out, before - out[subject_col].nunique()


def fix_dummy_p(summary, pcol='p_vs_chance'):
    #the dummy has no signal by construction
    summary = summary.copy()
    summary.loc[summary.model == 'dummy', pcol] = 1.0
    return summary


def finalize_population(out, cohort, log):
    res = pd.read_csv(out / 'population_subject_results.csv')
    parts = []
    for contrast, column in [('MED_vs_MW', MED), ('IMW2_vs_IMW1', CONTROL)]:
        r, dropped = keep(res[res.contrast == contrast], 'subject', passing(cohort, column))
        log.append(f'population {contrast}: dropped {dropped} subjects')
        parts.append(r)
    res = pd.concat(parts, ignore_index=True)
    #the dummy predicts one class, so its balanced accuracy is 0.5 already
    res.loc[res.model == 'dummy', 'bal_acc'] = 0.5
    res.to_csv(out / 'population_subject_results_final.csv', index=False)
    summary = fix_dummy_p(population_summarize(res))
    summary.to_csv(out / 'population_summary_final.csv', index=False)
    print_table('population summary (final)', summary)

    #learning curve: same subject filter on the evaluated subjects
    lc = pd.read_csv(out / 'population_learning_curve_subject.csv')
    lc, dropped = keep(lc, 'subject', passing(cohort, MED))
    log.append(f'learning curve: dropped {dropped} subjects')
    lc.to_csv(out / 'population_learning_curve_subject_final.csv', index=False)
    learning_curve_summary(lc).to_csv(out / 'population_learning_curve_summary_final.csv', index=False)

    shifts = pd.read_csv(out / 'theta_shift_subject.csv')
    theta_vs_accuracy(res, shifts).to_csv(out / 'population_theta_vs_accuracy_final.csv', index=False)


def real_minus_control(res):
    #per model: braboszcz subjects in both contrasts, med vs mw minus imw2 vs imw1
    wide = res[res.dataset == 'braboszcz'].pivot_table(index=['subject', 'model'], columns='contrast',
                                                       values='bal_acc').dropna().reset_index()
    rows = []
    for model, g in wide.groupby('model'):
        delta = g['MED_vs_MW'] - g['IMW2_vs_IMW1']
        s = subject_summary(delta, chance=0.0)
        rows.append(dict(model=model, n_subjects=s['n_subjects'], mean_med_vs_mw=g['MED_vs_MW'].mean(),
                         mean_control=g['IMW2_vs_IMW1'].mean(), mean_difference=s['mean'],
                         ci_low=s['ci_low'], ci_high=s['ci_high'], p_signflip=s['p_vs_chance']))
    table = pd.DataFrame(rows)
    #fdr across the real models, the dummy is not a model
    real = table.model != 'dummy'
    table['p_fdr_bh'] = np.nan
    table.loc[real, 'p_fdr_bh'] = add_fdr(table[real], 'p_signflip').p_fdr_bh
    table.loc[~real, 'p_signflip'] = 1.0
    return table


def finalize_classifiers(out, cohort, log):
    res = pd.read_csv(out / 'classifier_comparison_subject.csv')
    parts = []
    for contrast, column in [('MED_vs_MW', MED), ('IMW2_vs_IMW1', CONTROL)]:
        r, dropped = keep(res[res.contrast == contrast], 'subject', passing(cohort, column))
        log.append(f'classifier comparison {contrast}: dropped {dropped} subjects')
        parts.append(r)
    res = pd.concat(parts, ignore_index=True)
    res.to_csv(out / 'classifier_comparison_subject_final.csv', index=False)

    summary = comparison_summarize(res)
    summary.to_csv(out / 'classifier_comparison_summary_final.csv', index=False)
    print_table('classifier comparison summary (final)', summary)
    fr = friedman(res)
    fr.to_csv(out / 'classifier_comparison_friedman_final.csv', index=False)
    print_table('friedman (final)', fr)

    diff = real_minus_control(res)
    diff.to_csv(out / 'classifier_real_minus_control_final.csv', index=False)
    print_table('real minus control (final)', diff)


def finalize_style(out, cohort, log):
    pred = pd.read_csv(out / 'style_subject_predictions.csv')
    parts = []
    for analysis, g in pred.groupby('analysis'):
        column = STYLE_IMW2 if analysis.startswith('meditators_vs_ctr_IMW2') else STYLE_MED
        g, dropped = keep(g, 'subject', passing(cohort, column))
        log.append(f'style {analysis}: dropped {dropped} subjects')
        parts.append(g)
    pred = pd.concat(parts, ignore_index=True)

    chance_of = {a: (1 / 3 if a.startswith('three_class') else 0.5) for a in pred.analysis.unique()}
    #dummy = expected accuracy, exactly 1 / k
    is_dummy = pred.model == 'dummy'
    pred.loc[is_dummy, 'frac_correct'] = pred.loc[is_dummy, 'analysis'].map(chance_of)
    pred.to_csv(out / 'style_subject_predictions_final.csv', index=False)

    transfer_name = 'transfer_braboszcz_to_brandmeyer'
    loso = pred[pred.analysis != transfer_name]
    summary = style_summarize(loso, {a: c for a, c in chance_of.items() if a != transfer_name})
    summary.loc[summary.model == 'dummy', ['p_vs_chance', 'p_fdr_bh']] = 1.0
    summary.to_csv(out / 'style_summary_final.csv', index=False)
    print_table('style summary (final)', summary)

    #confusion matrices from the filtered predictions
    confusion = pd.concat([confusion_tables(loso, STYLES, a) for a in
                           ['three_class_v1_braboszcz', 'three_class_v2_with_brandmeyer']],
                          ignore_index=True)
    confusion.to_csv(out / 'style_confusion_matrices_final.csv', index=False)

    #transfer test, recomputed from the filtered votes
    rows = []
    for model, g in pred[pred.analysis == transfer_name].groupby('model'):
        votes = g.vote
        n, n_him = len(g), int((votes == 'HYT').sum())
        rows.append(dict(model=model, n_brandmeyer_subjects=n, n_called_himalayan=n_him,
                         share_himalayan=n_him / n, n_called_VIP=int((votes == 'VIP').sum()),
                         n_called_ISY=int((votes == 'ISY').sum()),
                         p_binomial_vs_third=binomtest(n_him, n, 1 / 3).pvalue))
    pd.DataFrame(rows).to_csv(out / 'style_transfer_brandmeyer_final.csv', index=False)

    #version 2 accuracy by source
    v2 = loso[loso.analysis == 'three_class_v2_with_brandmeyer']
    if len(v2):
        source = v2.assign(source=np.where(v2.dataset == 'brandmeyer', 'HYT (brandmeyer)',
                                           v2.true + ' (braboszcz)'))
        source = source.groupby(['model', 'source']).frac_correct.agg(['mean', 'count']).reset_index()
        source.to_csv(out / 'style_v2_accuracy_by_source_final.csv', index=False)


def finalize_calibration(out, cohort, log):
    reps_file = out / 'calibration_subject_reps.csv'
    if not reps_file.exists():
        log.append('calibration: calibration_subject_reps.csv not there, skipped')
        return
    reps = pd.read_csv(reps_file)
    #the 5 x 5 design: (calibration amount, placement) rows per subject. the rule
    #(>= 6 units per class, k + 2 units available per class, amounts under 2 units
    #skipped) is applied inside calibration_64.run_subject, so a subject / amount that
    #fails it has no rows. here only the common cohort rule is added.
    parts = []
    for task, column in [('MED_vs_IMW2', MED), ('IMW2_vs_IMW1', CONTROL)]:
        r, dropped = keep(reps[reps.task == task], 'subject', passing(cohort, column))
        log.append(f'calibration {task}: dropped {dropped} subjects')
        parts.append(r)
    reps = pd.concat(parts, ignore_index=True)

    #brandmeyer 8 s is one probe (under 2 units), it must not be in the file
    bad = reps[(reps.dataset == 'brandmeyer') & (reps.seconds_per_class == 8)]
    assert len(bad) == 0, 'brandmeyer 8 s rows should have been skipped'
    reps.to_csv(out / 'calibration_subject_reps_final.csv', index=False)

    summary = calibration_summarize(reps)
    summary.to_csv(out / 'calibration_summary_final.csv', index=False)
    print_table('calibration summary (final)', summary)

    #null-adjusted gain per amount, braboszcz subjects that are in both runs
    gain = meditation_specific_gain(reps)
    secs = reps.groupby('seconds_per_class').calibration_seconds.mean().rename('mean_calibration_seconds')
    gain = gain.merge(secs, left_on='seconds_per_class', right_index=True, how='left')
    gain['p_fdr_bh'] = np.nan
    for g in gain.gain.unique():
        rows = gain.gain == g
        gain.loc[rows, 'p_fdr_bh'] = add_fdr(gain[rows], 'p_signflip').p_fdr_bh
    gain.to_csv(out / 'calibration_meditation_specific_gain_final.csv', index=False)
    print_table('null-adjusted calibration gain (final)', gain)


def cohort_ns(cohort, out):
    rows = []
    for column in [MED, CONTROL, STYLE_MED, STYLE_IMW2]:
        t = cohort[cohort[column]]
        rows.append(dict(analysis=column, braboszcz=int((t.dataset == 'braboszcz').sum()),
                         brandmeyer=int((t.dataset == 'brandmeyer').sum()),
                         braboszcz_ctr=int(((t.dataset == 'braboszcz') & (t['style'] == 'CTR')).sum())))
    table = pd.DataFrame(rows)
    table.to_csv(out / 'cohort_final_ns.csv', index=False)
    print_table('subjects passing the cohort rule', table)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default=str(OUT), help='results folder (default results/ch64)')
    args = parser.parse_args()
    out = Path(args.dir)
    cohort = pd.read_csv(out / 'cohort_64.csv')
    log = []

    cohort_ns(cohort, out)
    finalize_population(out, cohort, log)
    finalize_classifiers(out, cohort, log)
    finalize_style(out, cohort, log)
    finalize_calibration(out, cohort, log)

    print('\nsubjects dropped by the cohort rule')
    for line in log:
        print(' ', line)
    print(f'\nwrote _final files to {out}')


if __name__ == '__main__':
    main()
