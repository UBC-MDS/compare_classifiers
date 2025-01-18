# %%
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from compare_classifiers.ensemble_compare_f1 import ensemble_compare_f1

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Get test data
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

estimators = [
    ('KNN', KNeighborsClassifier(n_neighbors=5)),
    ('LogisticRegression', LogisticRegression(random_state=42))
]

# Test voting method
def test_ensemble_compare_f1_voting():
    result = ensemble_compare_f1(estimators, X_train, y_train)
    voting_result = result[result['method'] == 'voting']
    assert 'fit_time' in voting_result.columns
    assert 'test_f1_score' in voting_result.columns
    assert 'train_f1_score' in voting_result.columns
    assert voting_result['test_f1_score'].between(0, 1).all()  # Verify the range of test_f1_score
    assert voting_result['train_f1_score'].between(0, 1).all()  # Verify the range of train_f1_score

# Test stacking method
def test_ensemble_compare_f1_stacking():
    result = ensemble_compare_f1(estimators, X_train, y_train)
    stacking_result = result[result['method'] == 'stacking']
    assert 'fit_time' in stacking_result.columns
    assert 'test_f1_score' in stacking_result.columns
    assert 'train_f1_score' in stacking_result.columns
    assert stacking_result['test_f1_score'].between(0, 1).all()  # Verify the range of test_f1_score
    assert stacking_result['train_f1_score'].between(0, 1).all()  # Verify the range of train_f1_score

# Test with different estimators
def test_ensemble_compare_f1_different_estimators():
    new_estimators = [
        ('RandomForest', RandomForestClassifier(n_estimators=10, random_state=42)),
        ('SVC', SVC(kernel='linear', random_state=42))
    ]
    result = ensemble_compare_f1(new_estimators, X_train, y_train)
    assert not result.empty
    assert 'fit_time' in result.columns
    assert 'test_f1_score' in result.columns
    assert 'train_f1_score' in result.columns

# Test with empty estimators list
def test_ensemble_compare_f1_empty_estimators():
    empty_estimators = []
    try:
        result = ensemble_compare_f1(empty_estimators, X_train, y_train)
    except ValueError as e:
        assert str(e) == "Invalid 'estimators' parameter: empty list"



# %%
test_ensemble_compare_f1_voting()
test_ensemble_compare_f1_stacking()
test_ensemble_compare_f1_different_estimators()
test_ensemble_compare_f1_empty_estimators()


# %%
