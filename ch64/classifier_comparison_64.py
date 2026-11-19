#script 3: compare classifier families with within-dataset leave-one-subject-out
#
#models: logistic regression (clf() from combined_analysis.py), gaussian naive
#bayes, mlp, random forest, svm-rbf, linear svm, knn and a majority-class dummy.
#every model sits behind the same preprocessing as clf() (median imputation,
#zero-variance filter, z-scoring), fit on the training folds only.
#hyperparameters are fixed below and never tuned on test data.
#
#contrasts: med vs mind wandering (braboszcz 54, brandmeyer), imw2 vs imw1 (braboszcz)
#summary: mean, bootstrap 95% ci and sign-flip p per model and contrast,
#plus a friedman test across models.
#
#make_models() and fit_weighted() are also used by style_classification_64.py.
#
#run from the repo root:
#.venv/bin/python ch64/classifier_comparison_64.py --jobs 4
#.venv/bin/python ch64/classifier_comparison_64.py --n-brab 3 --n-brand 2   (test run)
import argparse
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import friedmanchisquare
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.utils.validation import has_fit_parameter
from threadpoolctl import threadpool_limits

from common_64 import (add_fdr, add_subset_args, cell_weights, clf, imw2_vs_imw1,
                       load_from_args, med_vs_mw, output_dir, print_table, quiet_blas_warnings,
                       subject_summary)

SEED = 42

#fixed settings for every model, picked before running anything.
#logistic is exactly clf() from combined_analysis.py.
MODEL_SETTINGS = {
    'logistic': (None, 'C=0.1, solver=liblinear, max_iter=2000 (clf() from combined_analysis.py)'),
    'gaussian_nb': (lambda: GaussianNB(var_smoothing=1e-9),
                    'var_smoothing=1e-9 (sklearn default)'),
    'mlp': (lambda: MLPClassifier(hidden_layer_sizes=(64,), activation='relu', solver='adam',
                                  alpha=1e-3, learning_rate_init=1e-3, max_iter=500,
                                  early_stopping=True, validation_fraction=0.1,
                                  n_iter_no_change=10, random_state=SEED),
            'one hidden layer of 64 relu units, adam, alpha (l2)=1e-3, lr=1e-3, max_iter=500, '
            'early stopping on 10% of the training fold, patience 10'),
    'random_forest': (lambda: RandomForestClassifier(n_estimators=200, max_features='sqrt',
                                                     min_samples_leaf=5, n_jobs=1, random_state=SEED),
                      'n_estimators=200, max_features=sqrt, min_samples_leaf=5'),
    'svm_rbf': (lambda: SVC(kernel='rbf', C=1.0, gamma='scale', cache_size=1000, random_state=SEED),
                'kernel=rbf, C=1.0, gamma=scale (1 / (n_features * var(X)))'),
    'svm_linear': (lambda: LinearSVC(C=0.1, dual=False, max_iter=5000, random_state=SEED),
                   'C=0.1 (same as logistic), dual=False, squared hinge, l2, max_iter=5000'),
    'knn': (lambda: KNeighborsClassifier(n_neighbors=15, weights='uniform', metric='euclidean'),
            'k=15, uniform weights, euclidean distance'),
    'dummy': (lambda: DummyClassifier(strategy='most_frequent'),
              'most frequent class (weighted) in the training fold'),
}
MODEL_NAMES = list(MODEL_SETTINGS)


def make_models(names=MODEL_NAMES):
    #every model gets the clf() preprocessing, only the last step changes
    models = {}
    for name in names:
        build, _ = MODEL_SETTINGS[name]
        models[name] = clf() if build is None else clf().set_params(clf=build())
    return models


def fit_weighted(pipe, X, y, w, seed=SEED):
    #fit with sample weights. mlp and knn have no sample_weight argument, so for
    #those we resample the training rows with probability proportional to the weight,
    #which gives the same balance on average.
    if has_fit_parameter(pipe.named_steps['clf'], 'sample_weight'):
        return pipe.fit(X, y, sc__sample_weight=w, clf__sample_weight=w)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=len(y), replace=True, p=w / w.sum())
    return pipe.fit(X[idx], y[idx])


def hyperparameter_table():
    rows = []
    for name, (build, settings) in MODEL_SETTINGS.items():
        est = clf().named_steps['clf'] if build is None else build()
        weighting = ('sample_weight' if has_fit_parameter(est, 'sample_weight')
                     else 'weighted resampling of training rows')
        rows.append(dict(model=name, estimator=type(est).__name__, settings=settings,
                         class_balance=weighting))
    table = pd.DataFrame(rows)
    note = ('all models: SimpleImputer(median) -> VarianceThreshold(0) -> StandardScaler -> model, '
            'fit on the training folds only; no hyperparameter tuning')
    return table, note


def run_fold(name, X, y, tr, te, w):
    #one held-out subject, one model (this runs inside a joblib worker)
    quiet_blas_warnings()
    model = make_models([name])[name]
    start = time.time()
    fit_weighted(model, X[tr], y[tr], w)
    pred = model.predict(X[te])
    return dict(model=name, n_test=int(te.sum()), bal_acc=balanced_accuracy_score(y[te], pred),
                fit_seconds=time.time() - start)


def within_loso(d, feats, names, jobs=1):
    #leave one subject out inside each dataset, every model
    rows = []
    for dataset, dd in d.groupby('dataset'):
        dd = dd.reset_index(drop=True)
        X = dd[feats].to_numpy(float)
        y = dd.y.to_numpy()
        s = dd.subj.to_numpy()
        tasks = []
        for subj in np.unique(s):
            te = s == subj
            tr = ~te
            #need both classes in the test subject and in the training fold
            if np.unique(y[te]).size < 2 or np.unique(y[tr]).size < 2:
                continue
            w = cell_weights(dd[tr])
            for name in names:
                tasks.append((subj, delayed(run_fold)(name, X, y, tr, te, w)))
        results = Parallel(n_jobs=jobs)(t for _, t in tasks)
        for (subj, _), r in zip(tasks, results):
            rows.append(dict(subject=subj, dataset=dataset, **r))
    return pd.DataFrame(rows)


def summarize(res):
    rows = []
    for keys, g in res.groupby(['contrast', 'dataset', 'model']):
        rows.append({**dict(zip(['contrast', 'dataset', 'model'], keys)),
                     **subject_summary(g.groupby('subject').bal_acc.mean()),
                     'median_fit_seconds': g.fit_seconds.median()})
    return add_fdr(pd.DataFrame(rows), 'p_vs_chance')


def friedman(res):
    #friedman test across models on the subjects that have every model
    rows = []
    for (contrast, dataset), g in res.groupby(['contrast', 'dataset']):
        wide = g.pivot_table(index='subject', columns='model', values='bal_acc').dropna()
        row = dict(contrast=contrast, dataset=dataset, n_subjects=len(wide), n_models=wide.shape[1])
        if wide.shape[1] >= 3 and len(wide) >= 2:
            stat, p = friedmanchisquare(*[wide[c] for c in wide.columns])
            row.update(chi2=stat, p=p)
            #mean rank of each model (1 = best) to show which way it goes
            ranks = wide.rank(axis=1, ascending=False).mean()
            row['mean_ranks'] = ', '.join(f'{m}={r:.2f}' for m, r in ranks.sort_values().items())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    add_subset_args(parser)
    parser.add_argument('--jobs', type=int, default=1, help='parallel folds')
    parser.add_argument('--models', nargs='+', default=MODEL_NAMES, choices=MODEL_NAMES)
    args = parser.parse_args()
    out = output_dir(args)
    threadpool_limits(limits=1)

    #hyperparameter table for the paper
    table, note = hyperparameter_table()
    table.to_csv(out / 'classifier_hyperparameters.csv', index=False)
    with open(out / 'classifier_hyperparameters.md', 'w') as f:
        f.write('| model | estimator | settings | class balance |\n|---|---|---|---|\n')
        for r in table.itertuples():
            f.write(f'| {r.model} | {r.estimator} | {r.settings} | {r.class_balance} |\n')
        f.write(f'\n{note}\n')
    print_table('hyperparameters', table)
    print(note)

    d, feats = load_from_args(args)
    parts = []
    for contrast, task in [('MED_vs_MW', med_vs_mw(d)), ('IMW2_vs_IMW1', imw2_vs_imw1(d))]:
        start = time.time()
        r = within_loso(task, feats, args.models, args.jobs)
        r['contrast'] = contrast
        parts.append(r)
        print(f'{contrast}: {r.subject.nunique()} subjects, {time.time() - start:.0f}s', flush=True)
    res = pd.concat(parts, ignore_index=True)
    res.to_csv(out / 'classifier_comparison_subject.csv', index=False)

    summary = summarize(res)
    summary.to_csv(out / 'classifier_comparison_summary.csv', index=False)
    print_table('summary (balanced accuracy, subject = unit)', summary)

    fr = friedman(res)
    fr.to_csv(out / 'classifier_comparison_friedman.csv', index=False)
    print_table('friedman across models', fr)
    print(f'\nwrote results to {out}')


if __name__ == '__main__':
    main()
