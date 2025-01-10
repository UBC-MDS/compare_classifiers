def ensemble_compare_f1(estimators, X_train, y_train, method='stacking'):
    """
    Show cross validation fit time and f1 scores of a classifier by stacking or voting the estimators.

    Parameters:
    estimators (list): List of (str, estimator) tuples.
    X_train (array-like): Feature matrix for training.
    y_train (array-like): Target vector for training.
    method (str): Ensemble method, 'stacking' or 'voting'. Default is 'stacking'.

    Returns:
    dict: Dictionary containing fit times and f1 scores.
    """
    # ...existing code...

# Example usage:
# estimators = [('lr', LogisticRegression()), ('rf', RandomForestClassifier())]
# X_train = ... # feature matrix for training
# y_train = ... # target vector for training
# result = ensemble_compare_f1(estimators, X_train, y_train, method='stacking')
# print(result)