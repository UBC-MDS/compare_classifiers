from compare_classifiers.error_handling.check_valid_estimators import check_valid_estimators

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from tests.test_data import models

import pytest

model_dict = models()
knn5 = model_dict['knn5']
rfr = model_dict['rfr']
pipe_regressor = model_dict['pipe_regressor']

FIRST = 'first'

def test_estimators_incorrect_type():
    """Raises error when estimators is not a list of (name, estimator) tuples where name is a string and estimator is a sklearn individual or pipeline Classifier."""
    with pytest.raises(TypeError):
        check_valid_estimators('', FIRST)
    with pytest.raises(TypeError):
        check_valid_estimators(1, FIRST)
    with pytest.raises(TypeError):
        check_valid_estimators(True, FIRST)
    with pytest.raises(TypeError):
        check_valid_estimators({}, FIRST)
    with pytest.raises(TypeError):
        check_valid_estimators(('knn5', knn5), FIRST)
    with pytest.raises(TypeError):
        check_valid_estimators([('', )], FIRST)
    with pytest.raises(TypeError):
        check_valid_estimators([('knn5', knn5), ('knn5', knn5, '')], FIRST)
    with pytest.raises(TypeError):
        check_valid_estimators([('knn5', knn5), (knn5, knn5)], FIRST)
    with pytest.raises(TypeError):
        check_valid_estimators([('knn5', knn5), ('', '')], FIRST)
    with pytest.raises(TypeError):
        check_valid_estimators([('knn5', knn5), ('rfr', rfr)], FIRST)
    with pytest.raises(TypeError):
        check_valid_estimators([('knn5', knn5), pipe_regressor], FIRST)

def test_estimators_incorrect_length():
    """Raises error when estimators is a list containing 0 or 1 item."""
    with pytest.raises(ValueError):
        check_valid_estimators([], FIRST)
    with pytest.raises(ValueError):
        check_valid_estimators([('knn5', knn5)], FIRST)