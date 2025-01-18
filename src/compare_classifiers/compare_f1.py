import pandas as pd
from sklearn.model_selection import cross_val_score
import time

def compare_f1(estimators, X, y):
    """
    Show cross validation results, including fit time and f1 scores for each estimator.

    Parameters
    ----------
    estimators : list of tuples
        A list of (name, estimator) tuples, consisting of individual estimators to be processed through the voting or stacking classifying ensemble. Each tuple contains a string: name/label of estimator, and a model: the estimator, which implements
        the scikit-learn API (`fit`, `predict`, etc.).
    
    X_train : Pandas data frame or Numpy array
        Data frame containing training data along with n features or ndarray with no feature names.
        
    y_train : Pandas series or Numpy array
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

    results = []

    for name, estimator in estimators:
        try:
            start_time = time.time()
            
            cv_results = cross_val_score(estimator, X, y, scoring='f1', cv=5)
            fit_time = time.time() - start_time
            
            test_score = cv_results.mean()
            
            estimator.fit(X, y)
            train_score = cross_val_score(estimator, X, y, scoring='f1', cv=5).mean()
            
            results.append({
                'Estimator': name,
                'Fit Time': fit_time,
                'Test Score (F1)': test_score,
                'Train Score (F1)': train_score
            })
        
        except Exception as e:
            print(f"Error with estimator {name}: {e}")
            results.append({
                'Estimator': name,
                'Fit Time': None,
                'Test Score (F1)': None,
                'Train Score (F1)': None
            })

    return pd.DataFrame(results)