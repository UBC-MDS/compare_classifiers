from compare_classifiers.ensemble_predict import ensemble_predict

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from tests.test_data import test_data, models

import pytest

import numpy as np

from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier

# Create test data

data_dict = test_data()
X_train = data_dict['X_train']
X_train_ss = data_dict['X_train_ss']
X_test_ss = data_dict['X_test_ss']
X_test_rs = data_dict['X_test_rs']
y_train = data_dict['y_train']
y_test = data_dict['y_test']

model_dict = models()
knn5 = model_dict['knn5']
knn5_and_mnb = [
    ('knn5', knn5),
    ('mnp', model_dict['mnp'])
]
two_pipes = [
    ('pipe_rf', model_dict['pipe_rf']),
    ('pipe_svm', model_dict['pipe_svm'])
]
multi_ind = [
    ('logreg', model_dict['logreg']),
    ('gb', model_dict['gb']),
    ('svm', model_dict['svm']),
    ('rf', model_dict['rf']),
    ('knn5', knn5)
]
multi_pipe = [
    ('pipe_svm', model_dict['pipe_svm']),
    ('pipe_rf', model_dict['pipe_rf']),
    ('pipe_knn5', model_dict['pipe_knn5']),
    ('pipe_gb', model_dict['pipe_gb']),
    ('pipe_mnp', model_dict['pipe_mnp'])
]
rfr = model_dict['rfr']
pipe_regressor = model_dict['pipe_regressor']

VOTING = 'voting'
STACKING = 'stacking'


# Error handling test cases

def test_ensemble_method_incorrect_type():
    """Raises error when ensemble_method is not a string."""
    with pytest.raises(TypeError):
        ensemble_predict(knn5_and_mnb, X_train, y_train, 1, X_test_ss)
    with pytest.raises(TypeError):
        ensemble_predict(knn5_and_mnb, X_train, y_train, True, X_test_ss)
    with pytest.raises(TypeError):
        ensemble_predict(knn5_and_mnb, X_train, y_train, {}, X_test_ss)
    with pytest.raises(TypeError):
        ensemble_predict(knn5_and_mnb, X_train, y_train, (), X_test_ss)
    with pytest.raises(TypeError):
        ensemble_predict(knn5_and_mnb, X_train, y_train, X_train, X_test_ss)
    with pytest.raises(TypeError):
        ensemble_predict(knn5_and_mnb, X_train, y_train, y_train, X_test_ss)

def test_ensemble_method_incorrect_value():
    """Raises error when ensemble_method is a string of neither 'voting' nor 'stacking'."""
    with pytest.raises(ValueError):
        ensemble_predict(knn5_and_mnb, X_train, y_train, '', X_test_ss)


# Success test cases

def test_individual_voting_success():
    """When estimators is a list of individual Classifiers, returns the same numpy array as sklearn's VotingClassifier when ensemble_method is 'voting'."""
    vc = VotingClassifier(knn5_and_mnb)
    vc = vc.fit(X_train_ss, y_train)
    predictions = vc.predict(X_test_ss)
    assert(np.all(ensemble_predict(knn5_and_mnb, X_train_ss, y_train, VOTING, X_test_ss) == predictions))

def test_individual_stacking_success():
    """When estimators is a list of individual Classifiers, returns the same numpy array as sklearn's StackingClassifier when ensemble_method is 'stacking'."""
    sc = StackingClassifier(knn5_and_mnb)
    sc = sc.fit(X_train_ss, y_train)
    predictions = sc.predict(X_test_ss)
    assert(np.all(ensemble_predict(knn5_and_mnb, X_train_ss, y_train, STACKING, X_test_ss) == predictions))

def test_pipeline_voting_success():
    """When estimators is a list of pipelines, returns the same numpy array as sklearn's VotingClassifier when ensemble_method is 'voting'."""
    vc = VotingClassifier(two_pipes)
    vc = vc.fit(X_train, y_train)
    predictions = vc.predict(X_test_rs)
    assert(np.all(ensemble_predict(two_pipes, X_train, y_train, VOTING, X_test_rs) == predictions))

def test_pipeline_stacking_success():
    """When estimators is a list of pipelines, returns the same numpy array as sklearn's StackingClassifier when ensemble_method is 'stacking'."""
    sc = StackingClassifier(two_pipes)
    sc = sc.fit(X_train, y_train)
    predictions = sc.predict(X_test_rs)
    assert(np.all(ensemble_predict(two_pipes, X_train, y_train, STACKING, X_test_rs) == predictions))

def test_multi_individual_voting_success():
    """When estimators is a list of more than 2 individual Classifiers, returns the same numpy array as sklearn's VotingClassifier when ensemble_method is 'voting'."""
    vc = VotingClassifier(multi_ind)
    vc = vc.fit(X_train_ss, y_train)
    predictions = vc.predict(X_test_ss)
    assert(np.all(ensemble_predict(multi_ind, X_train_ss, y_train, VOTING, X_test_ss) == predictions))

def test_multi_individual_stacking_success():
    """When estimators is a list of more than 2 individual Classifiers, returns the same numpy array as sklearn's StackingClassifier when ensemble_method is 'stacking'."""
    sc = StackingClassifier(multi_ind)
    sc = sc.fit(X_train_ss, y_train)
    predictions = sc.predict(X_test_ss)
    assert(np.all(ensemble_predict(multi_ind, X_train_ss, y_train, STACKING, X_test_ss) == predictions))

def test_multi_pipeline_voting_success():
    """When estimators is a list of more than 2 pipelines, returns the same numpy array as sklearn's VotingClassifier when ensemble_method is 'voting'."""
    vc = VotingClassifier(multi_pipe)
    vc = vc.fit(X_train, y_train)
    predictions = vc.predict(X_test_rs)
    assert(np.all(ensemble_predict(multi_pipe, X_train, y_train, VOTING, X_test_rs) == predictions))

def test_multi_pipeline_stacking_success():
    """When estimators is a list of more than 2 pipelines, returns the same numpy array as sklearn's StackingClassifier when ensemble_method is 'stacking'."""
    sc = StackingClassifier(multi_pipe)
    sc = sc.fit(X_train, y_train)
    predictions = sc.predict(X_test_rs)
    assert(np.all(ensemble_predict(multi_pipe, X_train, y_train, STACKING, X_test_rs) == predictions))