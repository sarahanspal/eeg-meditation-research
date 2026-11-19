#script 4: calibration decomposition on 64 channels
#
#for each held-out subject, compare three classifiers on the same test epochs:
#population: trained on every other subject (clf(), cell_weights)
#threshold only: population model, only the decision threshold moved using the
#subject's own calibration epochs
#full update: refit with the calibration epochs added, weight 20
#
#same design as calibration_decomposition.py, but the calibration amount is set
#in seconds per class instead of a fraction, at 5 amounts x 5 placements.
#calibration blocks and guard gaps come from calibration_decomposition.calibration_split
#and combined_analysis.estimate_buffer_epochs.
#
#tasks: med vs mind wandering (population pooled over both datasets) and the
#imw2 vs imw1 label control (braboszcz only). the meditation-specific gain is the
#per-subject full-update gain on med vs imw2 minus the gain on imw2 vs imw1.
#
#run from the repo root:
#.venv/bin/python ch64/calibration_64.py --jobs 4
#.venv/bin/python ch64/calibration_64.py --n-brab 3 --n-brand 2   (test run)
import argparse
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

from common_64 import (add_subset_args, cell_weights, clf, imw2_vs_imw1, load_from_args,
                       med_vs_mw, output_dir, print_table, quiet_blas_warnings, subject_summary)
from calibration_decomposition import calibration_split
from combined_analysis import estimate_buffer_epochs

#calibration amounts, seconds of labelled data per class
CAL_SECONDS_PER_CLASS = (8, 16, 32, 64, 128)
N_PLACEMENTS = 5
CAL_WEIGHT = 20
EPOCH_SECONDS = 4


def calibration_units(g):
    #ordered calibration units per class for one subject.
    #braboszcz: one unit = one 4 s epoch. brandmeyer: one unit = one probe in one
    #session (two 4 s epochs), so epochs before the same probe stay together.
    units = {}
    if g.dataset.iloc[0] == 'brandmeyer':
        unit_col, unit_seconds = 'cal_unit', 2 * EPOCH_SECONDS
        for c in (0, 1):
            z = g[g.y == c].sort_values(['session', 't_start'])
            units[c] = z.cal_unit.drop_duplicates().to_numpy()
    else:
        unit_col, unit_seconds = 'epoch_idx', EPOCH_SECONDS
        for c in (0, 1):
            units[c] = np.sort(g.loc[g.y == c, 'epoch_idx'].unique())
    return unit_col, unit_seconds, units


def split_by_count(units, k, rep, n_rep, buffer):
    #calibration_split() takes a fraction, so call it per class with k / n,
    #which gives exactly k units for that class (it still enforces at least 2)
    cal, test = {}, {}
    for c, u in units.items():
        cal_c, test_c = calibration_split({c: u}, k / len(u), rep, n_rep, buffer)
        cal[c], test[c] = cal_c[c], test_c[c]
    return cal, test


def run_subject(subj, g, X, y, s, ds, feats, task, seconds=CAL_SECONDS_PER_CLASS,
                n_rep=N_PLACEMENTS, cal_weight=CAL_WEIGHT):
    #all calibration amounts and placements for one held-out subject (runs in a joblib worker).
    #X, y, s, ds are the full arrays for every subject. they are the same objects for
    #every job, so joblib writes them to temp disk only once; the training rows are
    #picked here inside the worker.
    quiet_blas_warnings()
    other = s != subj
    X_other, y_other, ds_other = X[other], y[other], ds[other]
    g = g.sort_values(['condition', 'epoch_idx']).reset_index(drop=True)
    unit_col, unit_seconds, units = calibration_units(g)
    if min(len(u) for u in units.values()) < 6:
        return []

    #guard gap from the autocorrelation, as in calibration_decomposition.py.
    #braboszcz rows carry empty session / t_start columns after the concat, drop
    #them so estimate_buffer_epochs orders by epoch_idx
    if g.dataset.iloc[0] == 'braboszcz':
        buffer = estimate_buffer_epochs(g.drop(columns=['session', 't_start'], errors='ignore'), feats)
    else:
        #a probe is two epochs, so halve the guard (same as the 14-channel code)
        buffer = max(1, int(np.ceil(estimate_buffer_epochs(g, feats) / 2)))

    #population model, equal weight per dataset x state (same as the 14-channel code)
    w_other = cell_weights(pd.DataFrame({'dataset': ds_other, 'y': y_other}))
    pop = clf().fit(X_other, y_other, sc__sample_weight=w_other, clf__sample_weight=w_other)
    Xt = g[feats].to_numpy(float)
    yt = g.y.to_numpy()
    u = g[unit_col].to_numpy()
    rows = []

    for sec in seconds:
        k = int(round(sec / unit_seconds))
        #calibration_split always takes at least 2 units, so a smaller amount would
        #just repeat the 2-unit result (8 s is one brandmeyer probe), skip it
        if k < 2:
            continue
        #need room for the block plus at least two test units in every class
        if any(k + 2 > len(v) for v in units.values()):
            continue
        for rep in range(n_rep):
            cal, test = split_by_count(units, k, rep, n_rep, buffer)
            cm = (np.isin(u, list(cal[0])) & (yt == 0)) | (np.isin(u, list(cal[1])) & (yt == 1))
            tm = (np.isin(u, list(test[0])) & (yt == 0)) | (np.isin(u, list(test[1])) & (yt == 1))
            if np.unique(yt[cm]).size < 2 or np.unique(yt[tm]).size < 2:
                continue
            Xc, yc, Xte, yte = Xt[cm], yt[cm], Xt[tm], yt[tm]

            #population model as is
            base = balanced_accuracy_score(yte, pop.predict(Xte))

            #threshold only: best threshold on the calibration scores
            sc = pop.decision_function(Xc)
            candidates = np.r_[-np.inf, np.unique(sc), np.inf]
            scores = [balanced_accuracy_score(yc, sc > t) for t in candidates]
            thr = candidates[int(np.argmax(scores))]
            bias = balanced_accuracy_score(yte, pop.decision_function(Xte) > thr)

            #full update: refit with the calibration epochs added at weight 20.
            #weights are recomputed on the new training set, then the calibration
            #rows (the last ones) are multiplied by the calibration weight
            X_train = np.vstack([X_other, Xc])
            y_train = np.r_[y_other, yc]
            ds_train = np.r_[ds_other, np.full(len(yc), g.dataset.iloc[0])]
            w = cell_weights(pd.DataFrame({'dataset': ds_train, 'y': y_train}))
            w[-len(yc):] *= cal_weight
            full = clf().fit(X_train, y_train, sc__sample_weight=w, clf__sample_weight=w)
            updated = balanced_accuracy_score(yte, full.predict(Xte))

            rows.append(dict(subject=subj, dataset=g.dataset.iloc[0], task=task,
                             seconds_per_class=sec, rep=rep, n_cal=int(cm.sum()),
                             calibration_seconds=int(cm.sum()) * EPOCH_SECONDS,
                             n_test=int(tm.sum()), buffer_units=buffer,
                             base=base, bias_only=bias, full_update=updated,
                             gain_bias=bias - base, gain_full=updated - base))
    return rows


def run_task(d, feats, task, jobs=1):
    #hold out each subject in turn, population model = all other subjects in d
    X = d[feats].to_numpy(float)
    y = d.y.to_numpy()
    s = d.subj.to_numpy()
    ds = d.dataset.to_numpy()
    jobs_list = []
    for subj in np.unique(s):
        other = s != subj
        if np.unique(y[other]).size < 2:
            continue
        #pass the full arrays, not X[other]: a sliced copy per subject would be written
        #to temp disk once per subject (about 50 mb each) and can fill the disk
        jobs_list.append(delayed(run_subject)(subj, d[~other], X, y, s, ds, feats, task))
    results = Parallel(n_jobs=jobs)(jobs_list)
    return pd.DataFrame([r for rs in results for r in rs])


def summarize(res):
    rows = []
    for (task, dataset, sec), g in res.groupby(['task', 'dataset', 'seconds_per_class']):
        per = g.groupby('subject')[['base', 'bias_only', 'full_update', 'gain_bias', 'gain_full',
                                    'calibration_seconds']].mean()
        row = dict(task=task, dataset=dataset, seconds_per_class=sec,
                   mean_calibration_seconds=per.calibration_seconds.mean(), n_subjects=len(per),
                   base=per.base.mean(), bias_only=per.bias_only.mean(), full_update=per.full_update.mean())
        for gain in ['gain_bias', 'gain_full']:
            s = subject_summary(per[gain], chance=0.0)
            row.update({f'{gain}_mean': s['mean'], f'{gain}_ci_low': s['ci_low'],
                        f'{gain}_ci_high': s['ci_high'], f'{gain}_p': s['p_vs_chance']})
        rows.append(row)
    return pd.DataFrame(rows)


def meditation_specific_gain(res):
    #per braboszcz subject: gain on med vs imw2 minus gain on imw2 vs imw1, per amount
    per = res[res.dataset == 'braboszcz'].groupby(['task', 'seconds_per_class', 'subject'])[
        ['gain_bias', 'gain_full']].mean().reset_index()
    rows = []
    for sec, g in per.groupby('seconds_per_class'):
        wide = g.pivot(index='subject', columns='task', values=['gain_bias', 'gain_full']).dropna()
        for gain in ['gain_bias', 'gain_full']:
            if wide.empty or 'MED_vs_IMW2' not in wide[gain]:
                continue
            delta = wide[gain]['MED_vs_IMW2'] - wide[gain]['IMW2_vs_IMW1']
            rows.append(dict(seconds_per_class=sec, gain=gain,
                             med_gain=wide[gain]['MED_vs_IMW2'].mean(),
                             control_gain=wide[gain]['IMW2_vs_IMW1'].mean(),
                             **subject_summary(delta, chance=0.0)))
    out = pd.DataFrame(rows)
    return out.rename(columns={'mean': 'mean_difference', 'p_vs_chance': 'p_signflip'})


def main():
    parser = argparse.ArgumentParser()
    add_subset_args(parser)
    parser.add_argument('--jobs', type=int, default=1, help='parallel subjects')
    args = parser.parse_args()
    out = output_dir(args)
    threadpool_limits(limits=1)

    d, feats = load_from_args(args)
    parts = []
    #braboszcz med vs imw2 is called MED_vs_IMW2 so it pairs with the control by name;
    #brandmeyer rows in the same task are med vs mw
    for task, data in [('MED_vs_IMW2', med_vs_mw(d)), ('IMW2_vs_IMW1', imw2_vs_imw1(d))]:
        start = time.time()
        r = run_task(data.reset_index(drop=True), feats, task, args.jobs)
        parts.append(r)
        print(f'{task}: {r.subject.nunique() if len(r) else 0} subjects, {time.time() - start:.0f}s', flush=True)
    res = pd.concat(parts, ignore_index=True)
    res.to_csv(out / 'calibration_subject_reps.csv', index=False)

    summary = summarize(res)
    summary.to_csv(out / 'calibration_summary.csv', index=False)
    print_table('calibration summary (per-subject means over 5 placements)', summary)

    gain = meditation_specific_gain(res)
    gain.to_csv(out / 'calibration_meditation_specific_gain.csv', index=False)
    print_table('meditation-specific gain (MED_vs_IMW2 minus IMW2_vs_IMW1, braboszcz)', gain)
    print(f'\nwrote results to {out}')


if __name__ == '__main__':
    main()
