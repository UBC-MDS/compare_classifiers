from compare_classifiers.compare_f1 import compare_f1

import pandas as pd

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import pytest


@pytest.fixture
def synthetic_data():
    # Generate synthetic classification dataset
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    return X, y

@pytest.fixture
def estimators():
    # Define a list of estimators
    return [
        ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
        ('svm', make_pipeline(StandardScaler(), LinearSVC(random_state=42)))
    ]

def test_compare_f1(synthetic_data, estimators):
    X, y = synthetic_data

    # Test the function with valid input (estimators and dataset)
    result = compare_f1(estimators, X, y)

    # Check that result is a pandas DataFrame
    assert isinstance(result, pd.DataFrame)

    # Check that the DataFrame has the correct columns
    assert set(result.columns) == {'model', 'fit_time', 'test_f1_score', 'train_f1_score'}

    # Check that each row corresponds to an estimator
    assert result.shape[0] == len(estimators)

    # Ensure that all rows have non-null values for Fit Time, Test Score, and Train Score
    for index, row in result.iterrows():
        assert row['model'] in ['rf', 'svm']
        assert row['fit_time'] is not None
        assert row['test_f1_score'] is not None
        assert row['train_f1_score'] is not None
        assert 0 <= row['test_f1_score'] <= 1  # Verify the range of test_f1_score
        assert 0 <= row['train_f1_score'] <= 1  # Verify the range of train_f1_score