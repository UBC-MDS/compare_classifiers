from compare_classifiers.error_handling.check_valid_y import check_valid_y

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from tests.test_data import models

import pytest
import pandas as pd, numpy as np

model_dict = models()
knn5 = model_dict['knn5']
rfr = model_dict['rfr']

FIRST = 'first'

def test_y_incorrect_type():
    """Raises error when y is not a pandas series."""
    with pytest.raises(TypeError):
        check_valid_y('', FIRST)
    with pytest.raises(TypeError):
        check_valid_y(1, FIRST)
    with pytest.raises(TypeError):
        check_valid_y(True, FIRST)
    with pytest.raises(TypeError):
        check_valid_y({}, FIRST)
    with pytest.raises(TypeError):
        check_valid_y((), FIRST)
    with pytest.raises(TypeError):
        check_valid_y(pd.DataFrame({}), FIRST)

def test_y_empty():
    """Raises error when y is empty"""
    with pytest.raises(ValueError):
        check_valid_y(pd.Series(), FIRST)
    with pytest.raises(ValueError):
        check_valid_y(np.empty(0), FIRST)

def test_y_ndarray():
    """No error raised when y is an ndarray (this test case is added as it will not be covered in the success test cases of calling functions)"""
    check_valid_y(np.zeros(5), FIRST)