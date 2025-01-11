def confusion_matrices(estimators, X_train, y_train):
    """
    Display confusion matrices for multiple estimators on a dataset.

    Parameters:
    -----------
    estimators : list of tuples
        A list of (name, estimator) tuples, each containing a string: name/label of estimator, and a model: the estimator, which implements
        the scikit-learn API (`fit`, `predict`, etc.).

    X_train : Pandas data frame
        Data frame containing training data along with n features.
        
    y_train : Pandas series
        Target class labels for data in X_train.
    
    Returns:
    --------
    None
        Displays confusion matrices for each estimator using the provided training data.

    Example:
    --------
    >>> estimators = [
    ...     ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ...     ('svm', make_pipeline(StandardScaler(), LinearSVC(random_state=42)))
    ... ]
    >>> confusion_matrices(estimators, X, y)
    """
    pass