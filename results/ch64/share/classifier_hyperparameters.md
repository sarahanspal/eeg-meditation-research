| model | estimator | settings | class balance |
|---|---|---|---|
| logistic | LogisticRegression | C=0.1, solver=liblinear, max_iter=2000 (clf() from combined_analysis.py) | sample_weight |
| gaussian_nb | GaussianNB | var_smoothing=1e-9 (sklearn default) | sample_weight |
| mlp | MLPClassifier | one hidden layer of 64 relu units, adam, alpha (l2)=1e-3, lr=1e-3, max_iter=500, early stopping on 10% of the training fold, patience 10 | weighted resampling of training rows |
| random_forest | RandomForestClassifier | n_estimators=200, max_features=sqrt, min_samples_leaf=5 | sample_weight |
| svm_rbf | SVC | kernel=rbf, C=1.0, gamma=scale (1 / (n_features * var(X))) | sample_weight |
| svm_linear | LinearSVC | C=0.1 (same as logistic), dual=False, squared hinge, l2, max_iter=5000 | sample_weight |
| knn | KNeighborsClassifier | k=15, uniform weights, euclidean distance | weighted resampling of training rows |
| dummy | DummyClassifier | most frequent class (weighted) in the training fold | sample_weight |

all models: SimpleImputer(median) -> VarianceThreshold(0) -> StandardScaler -> model, fit on the training folds only; no hyperparameter tuning
