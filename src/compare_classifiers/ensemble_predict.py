from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier

def ensemble_predict(estimators, X_train, y_train, ensemble_method, test_data):
    """predict class for test data with provided estimators and whether predicting through Voting or Stacking

    Parameters
    ----------
    estimators : list of tuples
        Individual estimators to be processed through the voting or stacking classifying ensemble. Each tuple contains a string: label of estimator, and a model: the estimator.

    X_train : Pandas data frame
        Data frame containing training data along with n features.
        
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
    if ensemble_method == 'voting':
        ev = VotingClassifier(estimators)
        ev = ev.fit(X_train, y_train)
        return ev.predict(test_data)
    if ensemble_method == 'stacking':
        sc = StackingClassifier(estimators)
        sc = sc.fit(X_train, y_train)
        return sc.predict(test_data)