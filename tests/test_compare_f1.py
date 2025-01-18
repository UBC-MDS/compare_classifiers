import sys
import os
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/compare_classifiers')))
from compare_f1 import compare_f1
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError


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

def test_compare_f1_basic(synthetic_data, estimators):
    X, y = synthetic_data

    # Test the function with valid input (estimators and dataset)
    result = compare_f1(estimators, X, y)

    # Check that result is a pandas DataFrame
    assert isinstance(result, pd.DataFrame)

    # Check that the DataFrame has the correct columns
    assert set(result.columns) == {'Estimator', 'Fit Time', 'Test Score (F1)', 'Train Score (F1)'}

    # Check that each row corresponds to an estimator
    assert len(result) == len(estimators)

    # Ensure that all rows have non-null values for Fit Time, Test Score, and Train Score
    for index, row in result.iterrows():
        assert row['Estimator'] in ['rf', 'svm']
        assert row['Fit Time'] is not None
        assert row['Test Score (F1)'] is not None
        assert row['Train Score (F1)'] is not None

def test_compare_f1_with_invalid_estimator(synthetic_data):
    X, y = synthetic_data

    # Test with an invalid estimator that raises an error
    invalid_estimators = [
        ('invalid', None)  # Invalid estimator (None)
    ]

    result = compare_f1(invalid_estimators, X, y)

    # Check that the result has None for this invalid estimator
    assert len(result) == 1
    assert result['Estimator'][0] == 'invalid'
    assert result['Fit Time'][0] is None
    assert result['Test Score (F1)'][0] is None
    assert result['Train Score (F1)'][0] is None

def test_compare_f1_with_no_estimators(synthetic_data):
    X, y = synthetic_data

    # Test with an empty estimator list
    result = compare_f1([], X, y)

    # Check that the result is an empty DataFrame
    assert result.empty

def test_compare_f1_with_empty_data(synthetic_data):
    X, y = synthetic_data

    # Test with empty dataset (X or y)
    result = compare_f1([('rf', RandomForestClassifier(n_estimators=10, random_state=42))], [], y)

    # The function should handle empty X gracefully
    assert result is not None
    assert len(result) == 1
    assert result['Estimator'][0] == 'rf'
    assert result['Fit Time'][0] is None
    assert result['Test Score (F1)'][0] is None
    assert result['Train Score (F1)'][0] is None

def test_compare_f1_with_unfitted_estimator(synthetic_data):
    X, y = synthetic_data

    # Test with an estimator that does not support fitting in the usual way (e.g., LinearSVC without scaling)
    unfitted_estimators = [
        ('svm_unfitted', LinearSVC(random_state=42))  # Not using a pipeline with StandardScaler
    ]
    
    result = compare_f1(unfitted_estimators, X, y)
    
    # The result should be calculated even if it's not fitted correctly yet
    assert len(result) == 1
    assert result['Estimator'][0] == 'svm_unfitted'
    assert result['Fit Time'][0] is not None
    assert result['Test Score (F1)'][0] is not None
    assert result['Train Score (F1)'][0] is not None

def test_compare_f1_with_no_data():
    # Test with no data (empty input)
    result = compare_f1([], [], [])

    # The result should be an empty DataFrame
    assert result.empty
