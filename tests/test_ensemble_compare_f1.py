# %%
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from tests.test_data import models

from compare_classifiers.ensemble_compare_f1 import ensemble_compare_f1

import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Get test data
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create test estimators
model_dict = models()
knn5_and_mnb = model_dict['knn5_and_mnb']
two_pipes = model_dict['two_pipes']


def test_ensemble_compare_f1_ind():
    """Returns data frame with fit time, test score and train score for voting and stacking ensembles for individual estimators."""
    result = ensemble_compare_f1(knn5_and_mnb, X_train, y_train)
    # Check that result is a pandas DataFrame
    assert isinstance(result, pd.DataFrame)
    
    # Check that the DataFrame has the correct columns
    assert set(result.columns) == {'method', 'fit_time', 'test_f1_score', 'train_f1_score'}

    # Check that each row corresponds to an estimator
    assert result.shape[0] == 2

    # Ensure that all rows have non-null values for fit time, test score, and train score
    for index, row in result.iterrows():
        assert row['method'] in ['voting', 'stacking']
        assert row['fit_time'] is not None
        assert row['test_f1_score'] is not None
        assert row['train_f1_score'] is not None
        assert 0 <= row['test_f1_score'] <= 1  # Verify the range of test_f1_score
        assert 0 <= row['train_f1_score'] <= 1  # Verify the range of train_f1_score


def test_ensemble_compare_f1_pipe():
    """Returns data frame with fit time, test score and train score for voting and stacking ensembles for pipeline estimators."""
    result = ensemble_compare_f1(two_pipes, X_train, y_train)
    # Check that result is a pandas DataFrame
    assert isinstance(result, pd.DataFrame)
    
    # Check that the DataFrame has the correct columns
    assert set(result.columns) == {'method', 'fit_time', 'test_f1_score', 'train_f1_score'}

    # Check that each row corresponds to an estimator
    assert result.shape[0] == 2

    # Ensure that all rows have non-null values for fit time, test score, and train score
    for index, row in result.iterrows():
        assert row['method'] in ['voting', 'stacking']
        assert row['fit_time'] is not None
        assert row['test_f1_score'] is not None
        assert row['train_f1_score'] is not None
        assert 0 <= row['test_f1_score'] <= 1  # Verify the range of test_f1_score
        assert 0 <= row['train_f1_score'] <= 1  # Verify the range of train_f1_score