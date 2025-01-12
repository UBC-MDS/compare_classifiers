def compare_f1(estimators, X, y):
    """
    Show cross validation results, including fit time and f1 scores for each estimator.

    Parameters
    ----------
    estimators : list of tuples
        A list of (name, estimator) tuples, consisting of individual estimators to be processed through the voting or stacking classifying ensemble. Each tuple contains a string: name/label of estimator, and a model: the estimator, which implements
        the scikit-learn API (`fit`, `predict`, etc.).
    
    X_train : Pandas data frame
        Data frame containing training data along with n features.
        
    y_train : Pandas series
        Target class labels for data in X_train.

    Returns:
    --------
    Pandas data frame
        A data frame showing cross validation results on training data, with 3 columns: fit_time, test_score, train_score and 1 rows for each estimator.

    Example:
    -------- 
    >>> estimators = [
    ...     ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ...     ('svm', make_pipeline(StandardScaler(), LinearSVC(random_state=42)))
    ... ]
    >>> compare_f1(estimators, X, y)
    """
    pass