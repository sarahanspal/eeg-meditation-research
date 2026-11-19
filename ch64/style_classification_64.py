#script 6: who is meditating, and in which style? (leave-one-subject-out, subject = unit)
#
#all models from classifier_comparison_64.py with the same fixed settings and the
#same clf() preprocessing, fit on training folds only.
#
#a) matched 54 braboszcz meditators vs ctr, on med epochs and on imw2 epochs,
#equal total weight per group. also the mean age per group, and a rerun with
#age regressed out of every feature (regression fit on the training fold only).
#b) style vs style on med epochs, braboszcz only: hyt vs vip, hyt vs isy, vip vs isy.
#c) three-class hyt vs vip vs isy on med epochs, chance 1/3.
#version 1: himalayan = braboszcz hyt only.
#version 2: himalayan = braboszcz hyt + all brandmeyer subjects, equal total
#weight per class, reported next to a dataset-identity probe on med epochs.
#transfer: train on braboszcz only, majority vote for each brandmeyer subject,
#share called himalayan vs 1/3 (binomial test).
#
#a held-out subject belongs to one class, so the per-subject score is the share of
#their epochs predicted correctly. the summary is the macro (class-balanced) mean
#of those scores, with a class-stratified bootstrap ci and a sign-flip p vs chance.
#majority-vote ties go to the first class in the class list.
#
#every subset keeps only subjects with >= MIN_EPOCHS epochs in the condition it uses
#(common_64.keep_min_epochs), the same rule as the other scripts.
#
#the dummy (uniform random guess) is scored by its expected accuracy, the mean
#predicted probability of the true class, which is exactly 1/k. scoring its random
#hard guesses instead adds noise (sd ~0.045 per subject) that is shared between
#analyses with the same subjects, and gave baselines a bit below chance.
#its majority vote still comes from the random guesses.
#
#run from the repo root:
#.venv/bin/python ch64/style_classification_64.py --jobs 4
#.venv/bin/python ch64/style_classification_64.py --n-brab 3 --n-brand 2 --n-ctr 3   (test run)
#.venv/bin/python ch64/style_classification_64.py --models dummy --update   (re-run one model only)
import argparse
import time
import zlib

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import binomtest
from sklearn.dummy import DummyClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from threadpoolctl import threadpool_limits

from classifier_comparison_64 import MODEL_NAMES, fit_weighted, make_models
from combined_analysis import dataset_probe
from common_64 import (STYLES, add_fdr, add_subset_args, clf, keep_min_epochs, load_from_args,
                       output_dir, print_cohort, print_table, quiet_blas_warnings)

N_BOOT = 5000
N_FLIP = 20000
SEED = 42


#models

def make_model(name, seed=SEED):
    #the models from classifier_comparison_64.py, except the dummy. here every
    #held-out subject has one class and the classes get equal total weight, so
    #"most frequent" is an exact tie decided by rounding, and it flips against the
    #held-out subject's class (scores below chance). a uniform random guess is the
    #honest chance baseline for this design.
    #the dummy needs a different seed for every held-out subject: with one fixed
    #seed every subject gets the same guess sequence, the guesses are no longer
    #independent and the sign-flip p value is wrong.
    if name == 'dummy':
        return clf().set_params(clf=DummyClassifier(strategy='uniform', random_state=seed))
    return make_models([name])[name]


#weights and age

def equal_class_weights(y):
    #every class gets the same total weight, whatever its number of epochs
    counts = np.bincount(y)
    w = 1.0 / counts[y]
    return w / w.mean()


def residualize_age(Xtr, age_tr, Xte, age_te):
    #regress age out of every feature, fit on the training fold only.
    #missing values are filled with the training median first (the same thing the
    #imputer in clf() would do), so the regression never sees a nan.
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xtr = np.where(np.isnan(Xtr), med, Xtr)
    Xte = np.where(np.isnan(Xte), med, Xte)
    #subjects with no age get the training median age
    age_med = np.nanmedian(age_tr)
    age_tr = np.where(np.isnan(age_tr), age_med, age_tr)
    age_te = np.where(np.isnan(age_te), age_med, age_te)
    A_tr = np.column_stack([np.ones(len(age_tr)), age_tr])
    A_te = np.column_stack([np.ones(len(age_te)), age_te])
    beta = np.linalg.lstsq(A_tr, Xtr, rcond=None)[0]
    return Xtr - A_tr @ beta, Xte - A_te @ beta


#one fold

def run_fold(name, X, y, age, tr, te, n_classes, age_resid, seed=SEED):
    #fit one model on the training subjects, predict the held-out subject's epochs
    quiet_blas_warnings()
    Xtr, Xte = X[tr], X[te]
    if age_resid:
        Xtr, Xte = residualize_age(Xtr, age[tr], Xte, age[te])
    model = make_model(name, seed)
    fit_weighted(model, Xtr, y[tr], equal_class_weights(y[tr]))
    pred = model.predict(Xte)
    counts = np.bincount(pred, minlength=n_classes)
    true = y[te][0]
    if name == 'dummy':
        #expected accuracy of the random guess (exactly 1/k), see the header
        frac = float(np.mean(model.predict_proba(Xte)[:, true]))
    else:
        frac = float(np.mean(pred == true))
    return dict(model=name, n_epochs=int(te.sum()), frac_correct=frac,
                vote=int(np.argmax(counts)), counts=counts)


def run_loso(d, feats, classes, analysis, models, jobs=1, age_resid=False):
    #leave one subject out. d needs a 'label' column holding one of classes.
    d = d.reset_index(drop=True)
    X = d[feats].to_numpy(float)
    y = d.label.map({c: i for i, c in enumerate(classes)}).to_numpy()
    s = d.subj.to_numpy()
    age = d.age.to_numpy(float)
    subjects = d.drop_duplicates('subj').set_index('subj')

    tasks = []
    #dummy seed: different for every held-out subject and every analysis
    base_seed = SEED + zlib.crc32(analysis.encode()) % 1_000_000
    for i, subj in enumerate(subjects.index):
        te = s == subj
        tr = ~te
        #every class has to be in the training fold
        if np.unique(y[tr]).size < len(classes):
            continue
        for name in models:
            #seed differs per held-out subject (only the dummy uses it)
            tasks.append((subj, delayed(run_fold)(name, X, y, age, tr, te, len(classes),
                                                  age_resid, base_seed + i)))
    if not tasks:
        print(f'  {analysis}: not enough subjects per class for leave-one-subject-out, skipped')
        return pd.DataFrame()

    results = Parallel(n_jobs=jobs)(t for _, t in tasks)
    rows = []
    for (subj, _), r in zip(tasks, results):
        counts = r.pop('counts')
        row = dict(analysis=analysis, subject=subj, dataset=subjects.loc[subj, 'dataset'],
                   true=subjects.loc[subj, 'label'], **r)
        row['vote'] = classes[row['vote']]
        row['vote_correct'] = row['vote'] == row['true']
        for c, n in zip(classes, counts):
            row[f'share_{c}'] = n / r['n_epochs']
        rows.append(row)
    return pd.DataFrame(rows)


#subject-level statistics

def macro_mean(scores, labels):
    #mean of the per-class means, so every class counts the same
    return np.mean([scores[labels == c].mean() for c in np.unique(labels)])


def stratified_bootstrap_ci(scores, labels, n=N_BOOT, seed=SEED):
    #resample subjects with replacement inside each class, recompute the macro mean
    rng = np.random.default_rng(seed)
    groups = [scores[labels == c] for c in np.unique(labels)]
    boots = np.empty(n)
    for i in range(n):
        boots[i] = np.mean([rng.choice(g, len(g), replace=True).mean() for g in groups])
    return np.percentile(boots, [2.5, 97.5])


def macro_signflip(scores, labels, chance, n=N_FLIP, seed=SEED):
    #sign-flip test like combined_supplementary.signflip, but on the macro mean
    dev = scores - chance
    observed = macro_mean(dev, labels)
    rng = np.random.default_rng(seed)
    flips = rng.choice([-1, 1], size=(n, len(dev))) * dev
    #macro mean of every flipped row
    null = np.mean([flips[:, labels == c].mean(axis=1) for c in np.unique(labels)], axis=0)
    #small tolerance: a deviation that is zero up to rounding (e.g. 1/3) must give p = 1
    return (1 + np.sum(np.abs(null) >= abs(observed) - 1e-12)) / (n + 1)


def summarize(res, chance_of):
    rows = []
    for (analysis, model), g in res.groupby(['analysis', 'model']):
        scores = g.frac_correct.to_numpy()
        labels = g.true.to_numpy()
        chance = chance_of[analysis]
        ci = stratified_bootstrap_ci(scores, labels)
        rows.append(dict(analysis=analysis, model=model, n_subjects=len(g),
                         n_per_class=', '.join(f'{c}={n}' for c, n in g.true.value_counts().sort_index().items()),
                         chance=chance, macro_bal_acc=macro_mean(scores, labels), ci_low=ci[0], ci_high=ci[1],
                         p_vs_chance=macro_signflip(scores, labels, chance),
                         vote_bal_acc=balanced_accuracy_score(labels, g.vote)))
    return add_fdr(pd.DataFrame(rows), 'p_vs_chance')


def confusion_tables(res, classes, analysis):
    #subject-level confusion matrix (majority vote) for each model, long format
    rows = []
    for model, g in res[res.analysis == analysis].groupby('model'):
        cm = confusion_matrix(g.true, g.vote, labels=classes)
        for i, t in enumerate(classes):
            for j, p in enumerate(classes):
                rows.append(dict(analysis=analysis, model=model, true=t, predicted=p, n_subjects=int(cm[i, j])))
    return pd.DataFrame(rows)


#the analyses

def age_table(d):
    #mean age per group, one value per subject
    subj = d.drop_duplicates('subj')
    rows = []
    for group, g in subj.groupby('label'):
        rows.append(dict(group=group, n_subjects=len(g), mean_age=g.age.mean(), sd_age=g.age.std(),
                         min_age=g.age.min(), max_age=g.age.max(), n_missing_age=int(g.age.isna().sum())))
    return pd.DataFrame(rows)


def transfer_test(train, test, feats, models):
    #train on braboszcz only (three styles), majority vote for every brandmeyer subject
    X = train[feats].to_numpy(float)
    y = train.label.map({c: i for i, c in enumerate(STYLES)}).to_numpy()
    rows, preds = [], []
    for name in models:
        model = make_model(name)
        fit_weighted(model, X, y, equal_class_weights(y))
        votes = []
        for i, (subj, g) in enumerate(test.groupby('subj')):
            #new random guesses for every brandmeyer subject (dummy only)
            if name == 'dummy':
                model.set_params(clf__random_state=SEED + i)
            counts = np.bincount(model.predict(g[feats].to_numpy(float)), minlength=len(STYLES))
            vote = STYLES[int(np.argmax(counts))]
            votes.append(vote)
            preds.append(dict(analysis='transfer_braboszcz_to_brandmeyer', model=name, subject=subj,
                              n_epochs=len(g), vote=vote, **{f'share_{c}': n / len(g) for c, n in zip(STYLES, counts)}))
        n_him = votes.count('HYT')
        rows.append(dict(model=name, n_brandmeyer_subjects=len(votes), n_called_himalayan=n_him,
                         share_himalayan=n_him / len(votes) if votes else np.nan,
                         n_called_VIP=votes.count('VIP'), n_called_ISY=votes.count('ISY'),
                         p_binomial_vs_third=binomtest(n_him, len(votes), 1 / 3).pvalue if votes else np.nan))
    return pd.DataFrame(rows), pd.DataFrame(preds)


def main():
    parser = argparse.ArgumentParser()
    add_subset_args(parser)
    parser.add_argument('--jobs', type=int, default=1, help='parallel folds')
    parser.add_argument('--models', nargs='+', default=MODEL_NAMES, choices=MODEL_NAMES)
    parser.add_argument('--update', action='store_true',
                        help='only re-run --models and swap their rows into the existing result '
                             'files, keeping the other models (summary and fdr are recomputed over all)')
    args = parser.parse_args()
    out = output_dir(args)
    threadpool_limits(limits=1)

    d, feats = load_from_args(args, include_ctr=True)
    print_cohort(d, ['style / meditators vs ctr on MED epochs', 'meditators vs ctr on IMW2 epochs'])
    brab = d[d.dataset == 'braboszcz']
    brand_med = keep_min_epochs(d[(d.dataset == 'brandmeyer') & (d.condition == 'MED')], ['MED'])
    med_epochs = keep_min_epochs(brab[(brab.condition == 'MED') & brab.matched], ['MED'])

    results, chance_of, extra = [], {}, {}

    #a) meditators vs ctr
    groups = ['CTR', 'meditator']
    ages = None
    for cond in ['MED', 'IMW2']:
        a = keep_min_epochs(brab[brab.condition == cond], [cond])
        a['label'] = np.where(a['style'] == 'CTR', 'CTR', 'meditator')
        if ages is None:
            ages = age_table(a)
        for age_resid in [False, True]:
            name = f'meditators_vs_ctr_{cond}' + ('_age_regressed' if age_resid else '')
            start = time.time()
            results.append(run_loso(a, feats, groups, name, args.models, args.jobs, age_resid))
            chance_of[name] = 0.5
            print(f'{name}: {time.time() - start:.0f}s', flush=True)
    ages.to_csv(out / 'style_age_by_group.csv', index=False)
    print_table('age per group (braboszcz)', ages)

    #b) pairwise styles on med epochs
    for s1, s2 in [('HYT', 'VIP'), ('HYT', 'ISY'), ('VIP', 'ISY')]:
        b = med_epochs[med_epochs['style'].isin([s1, s2])].copy()
        b['label'] = b['style']
        name = f'{s1}_vs_{s2}'
        results.append(run_loso(b, feats, [s1, s2], name, args.models, args.jobs))
        chance_of[name] = 0.5
        print(f'{name} done', flush=True)

    #c) three-class, version 1: braboszcz only
    v1 = med_epochs.copy()
    v1['label'] = v1['style']
    results.append(run_loso(v1, feats, STYLES, 'three_class_v1_braboszcz', args.models, args.jobs))
    chance_of['three_class_v1_braboszcz'] = 1 / 3
    print('three_class_v1 done', flush=True)

    #c) three-class, version 2: brandmeyer subjects join the himalayan class
    brand_med['label'] = 'HYT'
    v2 = pd.concat([v1, brand_med], ignore_index=True)
    results.append(run_loso(v2, feats, STYLES, 'three_class_v2_with_brandmeyer', args.models, args.jobs))
    chance_of['three_class_v2_with_brandmeyer'] = 1 / 3
    print('three_class_v2 done', flush=True)

    #dataset-identity probe on med epochs: all of version 2, and the himalayan class only
    probes = []
    for label, subset in [('all version 2 subjects', v2),
                          ('himalayan class only (braboszcz HYT vs brandmeyer)', v2[v2.label == 'HYT'])]:
        try:
            probes.append(dataset_probe(subset, feats).assign(epochs='MED', subjects=label))
        except ValueError as e:
            #happens in tiny test runs when a fold has only one dataset
            print(f'  dataset probe ({label}) skipped: {e}')
    probe = pd.concat(probes, ignore_index=True) if probes else pd.DataFrame()
    probe.to_csv(out / 'style_dataset_probe_med.csv', index=False)

    #c) transfer test
    transfer, transfer_preds = transfer_test(v1, brand_med, feats, args.models)

    #save everything
    res = pd.concat([r for r in results if len(r)], ignore_index=True)
    res = pd.concat([res, transfer_preds], ignore_index=True)

    if args.update:
        #keep the old rows of every model we did not re-run
        old = pd.read_csv(out / 'style_subject_predictions.csv')
        res = pd.concat([old[~old.model.isin(args.models)], res], ignore_index=True)
        old_transfer = pd.read_csv(out / 'style_transfer_brandmeyer.csv')
        transfer = pd.concat([old_transfer[~old_transfer.model.isin(args.models)], transfer],
                             ignore_index=True)
        print(f'updated rows for {args.models}, kept the other models', flush=True)

    transfer.to_csv(out / 'style_transfer_brandmeyer.csv', index=False)
    res.to_csv(out / 'style_subject_predictions.csv', index=False)
    loso = res[res.analysis != 'transfer_braboszcz_to_brandmeyer']
    summary = summarize(loso, chance_of)
    summary.to_csv(out / 'style_summary.csv', index=False)

    confusion = pd.concat([confusion_tables(loso, STYLES, a) for a in
                           ['three_class_v1_braboszcz', 'three_class_v2_with_brandmeyer']], ignore_index=True)
    confusion.to_csv(out / 'style_confusion_matrices.csv', index=False)

    #version 2 split by where the himalayan subjects come from
    v2_res = loso[loso.analysis == 'three_class_v2_with_brandmeyer']
    if len(v2_res):
        source = v2_res.assign(source=np.where(v2_res.dataset == 'brandmeyer', 'HYT (brandmeyer)',
                                               v2_res.true + ' (braboszcz)'))
        source = source.groupby(['model', 'source']).frac_correct.agg(['mean', 'count']).reset_index()
        source.to_csv(out / 'style_v2_accuracy_by_source.csv', index=False)
        print_table('version 2: per-subject accuracy by source', source)

    print_table('summary (macro balanced accuracy over subjects)', summary)
    print_table('dataset-identity probe on med epochs (logistic, 5-fold grouped by subject)', probe)
    print_table('transfer: braboszcz model on brandmeyer med epochs', transfer)
    for analysis in ['three_class_v1_braboszcz', 'three_class_v2_with_brandmeyer']:
        cm = confusion[(confusion.analysis == analysis) & (confusion.model == 'logistic')]
        if len(cm):
            print(f'\nconfusion matrix, {analysis}, logistic (rows = true, cols = majority vote)')
            print(cm.pivot(index='true', columns='predicted', values='n_subjects').reindex(index=STYLES, columns=STYLES))
    print(f'\nwrote results to {out}')


if __name__ == '__main__':
    main()
