def ensemble_predict(estimators, X_train, y_train, ensemble_method, test_data):
    """predict class for test data with provided estimators and whether predicting through Voting or Stacking

    Parameters
    ----------
    estimators : list of tuples
        Individual estimators to be processed through the voting or stacking classifying ensemble. Each tuple contains a string: label of estimator, and a model: the estimator.

    X_train : Pandas data frame
        Training data without classes.

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
    >>> ensemble_predict(estimators, X, y, unseen_data, 'voting')
    """    
    return None