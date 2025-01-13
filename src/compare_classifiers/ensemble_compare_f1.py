from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.model_selection import cross_validate
import pandas as pd
from sklearn.linear_model import LogisticRegression

def ensemble_compare_f1(estimators, X_train, y_train, method='voting'):
    """
    Show cross validation results, including fit time and f1 scores by stacking and voting the estimators.

    Parameters
    ----------
    estimators : list of tuples
        A list of (name, estimator) tuples, consisting of individual estimators to be processed through the voting or stacking classifying ensemble. Each tuple contains a string: name/label of estimator, and a model: the estimator, which implements
        the scikit-learn API (`fit`, `predict`, etc.).
    
    X_train : Pandas data frame
        Data frame containing training data along with n features.
        
    y_train : Pandas series
        Target class labels for data in X_train.

    method : str, optional (default='voting')
        The ensemble method to use. Options are 'voting' or 'stacking'.

    Returns
    -------
    Pandas data frame
        A data frame showing cross validation results on training data, with 3 columns: fit_time, test_score, train_score and 2 rows: voting, stacking.
    
    Example:
    --------
    >>> estimators = [
    ...     ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ...     ('svm', make_pipeline(StandardScaler(), LinearSVC(random_state=42)))
    ... ]
    >>> ensemble_compare_f1(estimators, X, y)
    """
    if method not in ['voting', 'stacking']:
        raise ValueError("Method must be either 'voting' or 'stacking'")

    if method == 'voting':
        ensemble = VotingClassifier(estimators=estimators, voting='hard')
    if method == 'stacking':
        ensemble = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

    cv_results = cross_validate(ensemble, X_train, y_train, cv=5, scoring='f1_macro', return_train_score=True)
    

    results_df = pd.DataFrame({
        'fit_time': cv_results['fit_time'],
        'test_f1_score': cv_results['test_score'],
        'train_f1_score': cv_results['train_score']
    }, index=range(len(cv_results['fit_time'])))

    return results_df


# Example usage:
# estimators = [('lr', LogisticRegression()), ('rf', RandomForestClassifier())]
# X_train = ... # feature matrix for training
# y_train = ... # target vector for training
# method = 'voting'
# result = ensemble_compare_f1(estimators, X_train, y_train, method='stacking')
# print(result[['fit_time', 'test_f1_score', 'train_f1_score']])

# %%

if __name__ == "__main__":

    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    # Load example data
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define estimators
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
        ('svm', make_pipeline(StandardScaler(), LinearSVC(random_state=42)))
    ]

    # Call the ensemble_compare_f1 function with voting method
    result_voting = ensemble_compare_f1(estimators, X_train, y_train, method='voting')
    print("Voting method results:")
    print(result_voting)

    # Call the ensemble_compare_f1 function with stacking method
    result_stacking = ensemble_compare_f1(estimators, X_train, y_train, method='stacking')
    print("Stacking method results:")
    print(result_stacking)
# %%
