# compare_classifiers

Compare metrics such as f1 score and confusion matrices for your machine learning models and through voting or stacking them, then predict on test data with your choice of voting or stacking.

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
confusion_matrices(estimators, X, y)

# show fit time and f1 scores of estimators' cross validation results:
compare_f1(estimators, X, y) 

# show cross validation fit time and f1 scores by voting and stacking the estimators:
ensemble_compare_f1(estimators, X, y) 
```

At last, you can decide to predict on test data through voting or stacking the estimators:

```python
from compare_classifiers.ensemble_predict import ensemble_predict

# predict class labels for unseen data through voting results of estimators:
ensemble_predict(estimators, X, y, ensemble_method, unseen_data, 'voting') 
```

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`compare_classifiers` was created by Bryan Lee. It is licensed under the terms of the MIT license.

## Credits

`compare_classifiers` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
