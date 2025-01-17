from compare_classifiers.error_handling.check_valid_X import check_valid_X

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
empty_df = pd.DataFrame({'A' : []})

def test_X_incorrect_type():
    """Raises error when X is not a pandas data frame."""
    with pytest.raises(TypeError):
        check_valid_X('', FIRST)
    with pytest.raises(TypeError):
        check_valid_X(1, FIRST)
    with pytest.raises(TypeError):
        check_valid_X(True, FIRST)
    with pytest.raises(TypeError):
        check_valid_X({}, FIRST)
    with pytest.raises(TypeError):
        check_valid_X((), FIRST)
    with pytest.raises(TypeError):
        check_valid_X(pd.Series(), FIRST)

def test_X_empty():
    """Raises error when X is empty"""
    with pytest.raises(ValueError):
        check_valid_X(empty_df, FIRST)
    with pytest.raises(ValueError):
        check_valid_X(np.empty(0), FIRST)