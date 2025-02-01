# compare_classifiers 
[![Documentation Status](https://readthedocs.org/projects/compare-classifiers-524/badge/?version=latest)](https://compare-classifiers-524.readthedocs.io/en/latest/?badge=latest)
![Repo Status](https://img.shields.io/badge/repo%20status-Active-brightgreen)
![Python Versions](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
![CI/CD](https://github.com/UBC-MDS/compare_classifiers/actions/workflows/ci-cd.yml/badge.svg)
[![codecov](https://codecov.io/gh/UBC-MDS/compare_classifiers/graph/badge.svg?token=Divjf41jU3)](https://codecov.io/gh/UBC-MDS/compare_classifiers)

Compare metrics such as f1 score and confusion matrices for your machine learning models and through voting or stacking them, then predict on test data with your choice of voting or stacking!

This package is helpful when you are deciding whether to use a single Classifier or combine multiple well-performing Classifiers through an ensemble using Voting or Stacking to yield a more stable and trustworthy classification result. Each of the four functions serves a unique purpose:

`confusion_matrices`: provides confusion matrices side-by-side for all Classifiers to compare their performances.

`compare_f1`: provides a Pandas data frame, each row listing model fit time, and training and test scores for each Classifier.

`ensemble_compare_f1`: provides a Pandas data frame containing model fit time, training and test scores for both Voting and Stacking ensembles, with each ensemble in its own row.

`ensemble_predict`: provides classification predictions via Voting or Stacking multiple Classifiers.

Before using `ensemble_predict` on test or unseen data, we recommend that you run each of the three other functions on training data to examine how Classifiers perform individually on their own, and the ensemble performances of Voting against Stacking to make a well-informed decision. Sometimes, an individual Classifier could generate a better controlled machine learning environment if its performance rivals that of an ensemble.

## Contributors

Ke Gao: kegao1995@gmail.com
Bryan Lee
Susannah Sun
Wangkai Zhu

## Installation

```bash
$ pip install compare_classifiers
```

## Usage

`compare_classifiers` can be used to show confusion matrices and f1 scores for individual estimators, as well as f1 score for voting or stacking the estimators,
as follows:

```python
from compare_classifiers.confusion_matrices import confusion_matrices
from compare_classifiers.compare_f1 import compare_f1
from compare_classifiers.ensemble_compare_f1 import ensemble_compare_f1

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

estimators = [
    ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ('svr', make_pipeline(StandardScaler(),
                          LinearSVC(random_state=42)))
]

# show confusion matrices for estimators:
confusion_matrices(estimators, X_train, X_test, y_train, y_test)

# show fit time and f1 scores of estimators' cross validation results:
compare_f1(estimators, X_train, y_train) 

# show cross validation fit time and f1 scores by voting and stacking the estimators:
ensemble_compare_f1(estimators, X_train, y_train) 
```

At last, you can decide to predict on test data through voting or stacking the estimators:

```python
from compare_classifiers.ensemble_predict import ensemble_predict

# predict class labels for unseen data through voting results of estimators:
ensemble_predict(estimators, X_train, y_train, ensemble_method, unseen_data, 'voting') 
```

## Similar Packages

We are not aware of similar packages existing. Though there are available functions to present metrics for a single model and a single ensemble, we have not found functions that compare and display metrics and results for multiple models or ensembles all at once. Neither is there a function that predicts based on dynamic input of ensemble method.

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`compare_classifiers` was created by Bryan Lee. It is licensed under the terms of the MIT license.

## Credits

`compare_classifiers` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
