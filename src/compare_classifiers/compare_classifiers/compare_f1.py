def compare_f1(estimators, X, y):
    """
    Evaluates the performance and timing of a scikit-learn pipeline. 
    For each model in the list, the following metrics are calculated:
    - F1 Score
    - Accuracy
    - Precision
    - Recall

    Parameters
    ----------
    - estimators : list or pandas series
        A scikit-learn pipeline containing one or more models or transformers.
    
    - X : Pandas Data frame 
        Feature matrix for training and testing.

    - y : list or pandas series 
        Target vector for training and testing.

    Returns:
    --------
    - pandas DataFrame 
        A DataFrame containing performance metrics (F1 Score, Accuracy, 
        Precision, Recall).

    Example:
    -------- 
    >>> models = [('lr', LogisticRegression()), ('rf', RandomForestClassifier())]
    ... # X_train = ... # feature matrix for training
    ... # y_train = ... # target vector for training
    >>> compare_f1(pipeline, X_train, y_train)
    """

# Example usage:
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # generating training and testing split
# estimators = [('lr', LogisticRegression()), ('rf', RandomForestClassifier())] # generating list of models to score
# result = compare_f1(estimators, X_train, y_train)
# print(result)