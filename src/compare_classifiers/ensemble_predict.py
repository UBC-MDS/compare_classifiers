import pandas as pd, numpy as np

import sklearn
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.base import is_classifier

ESTIMATOR_ERROR = 'first parameter has to be a list of (name, estimator) tuples where name is a string and estimator is a sklearn Classifier or pipeline'
METHOD_ERROR = 'fourth parameter has to be a string of two possible values: "voting" and "stacking"'

def ensemble_predict(estimators, X_train, y_train, ensemble_method, test_data):
    """predict class for test data with provided estimators and whether predicting through Voting or Stacking

    Parameters
    ----------
    estimators : list of tuples
        Individual estimators to be processed through the voting or stacking classifying ensemble. Each tuple contains a string: label of estimator, and a model: the estimator.

    X_train : Pandas data frame or Numpy array
        Data frame containing training data along with n features or ndarray with no feature names.
        
    y_train : Pandas series
        Target class labels for data in X_train.

    ensemble_method : str
        Whether prediction is made through voting or stacking. Possible values are: 'voting' or 'stacking'.
        
    test_data : Pandas data frame
        Data to make predictions on.

    Returns
    -------
    Numpy array
        Predicted class labels for test_data.

    Examples
    --------
    >>> estimators = [
    ...     ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ...     ('svm', make_pipeline(StandardScaler(), LinearSVC(random_state=42)))
    ... ]
    >>> ensemble_predict(estimators, X, y, unseen_data, 'voting')
    """

    # Check if estimators is a list
    if not isinstance(estimators, list):
        raise TypeError(ESTIMATOR_ERROR)
    # Iterate through each element in the list
    for item in estimators:
        # Check if the item is a tuple with exactly two elements
        if not (isinstance(item, tuple) and len(item) == 2):
            raise TypeError(ESTIMATOR_ERROR)
        # Check if the first element is a string
        if not isinstance(item[0], str):
            raise TypeError(ESTIMATOR_ERROR)
        # Check if the second element is an instance of an sklearn classifier
        if not (is_classifier(item[1]) or isinstance(item[1], sklearn.pipeline.Pipeline)):
            raise TypeError(ESTIMATOR_ERROR)
    
    # Check if X_train is Pandas data frame or Numpy array
    if not (isinstance(X_train, pd.DataFrame) or isinstance(X_train, np.ndarray)):
        raise TypeError('second parameter has to be a Pandas data frame or Numpy array containing training data')   
    # Check if X_train contains data
    empty_df = isinstance(X_train, pd.DataFrame) and X_train.empty
    empty_ndarr = isinstance(X_train, np.ndarray) and X_train.size == 0
    if (empty_df or empty_ndarr):
        raise ValueError('second parameter seems to be an empty Pandas data frame or Numpy array. Please ensure data is present.') 
    
    # Check if y_train is Pandas series
    if not isinstance(y_train, pd.Series):
        raise TypeError('third parameter has to be a Pandas series')
    # Check if y_train contains data
    if y_train.empty:
        raise ValueError('third parameter seems to be an empty Pandas series. Please ensure your series contains data.') 
    
    # Check if ensemble_method is string
    if not isinstance(ensemble_method, str):
        raise TypeError(METHOD_ERROR)
    # Check if ensemble_method is either 'voting' or 'stacking'
    if (not ensemble_method == 'voting' and not ensemble_method == 'stacking'):
        raise ValueError(METHOD_ERROR)
    
    # Check if test_data is a Pandas data frame
    if not (isinstance(test_data, pd.DataFrame) or isinstance(test_data, np.ndarray)):
        raise TypeError('fifth parameter has to be a Pandas data frame or Numpy array containing training data')
    # Check if test_data contains data
    empty_df = isinstance(test_data, pd.DataFrame) and test_data.empty
    empty_ndarr = isinstance(test_data, np.ndarray) and test_data.size == 0
    if (empty_df or empty_ndarr):
        raise ValueError('fifth parameter seems to be an empty Pandas data frame or Numpy array. Please ensure data is present.') 

    # Return predictions if voting    
    if ensemble_method == 'voting':
        ev = VotingClassifier(estimators)
        ev = ev.fit(X_train, y_train)
        return ev.predict(test_data)
    
    # Return predictions if stacking
    if ensemble_method == 'stacking':
        sc = StackingClassifier(estimators)
        sc = sc.fit(X_train, y_train)
        return sc.predict(test_data)