def scoring_df(pipeline, X, y):
    """
    Evaluates the performance and timing of a scikit-learn pipeline. For each model
    in the pipeline, the following metrics are calculated:
    - F1 Score
    - Accuracy
    - Precision
    - Recall

    Parameters
    ----------
    - pipeline (Pipeline): A scikit-learn pipeline containing one or more models or transformers.
    - X (array-like or DataFrame): Feature matrix for training and testing.
    - y (array-like or DataFrame): Target vector for training and testing.

    Returns:
    - pd.DataFrame: A DataFrame containing performance metrics (F1 Score, Accuracy, Precision, Recall).
    >>> pipeline = Pipeline([
    >>>     ('scaler', StandardScaler()), 
    >>>     ('svc', SVC(kernel='linear', random_state=42)),
    >>>     ('random_forest',RandomForestRandomForestClassifier(n_estimators=100))
    >>> ])
    >>> scoring_df(pipeline, X_train, y_train)
    """