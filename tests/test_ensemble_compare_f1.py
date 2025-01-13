# %%
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from compare_classifiers.ensemble_compare_f1 import ensemble_compare_f1


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

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
    result = ensemble_compare_f1(estimators, X_train, y_train, method='voting')
    assert 'fit_time' in result.columns
    assert 'test_f1_score' in result.columns
    assert 'train_f1_score' in result.columns
    assert result['test_f1_score'].between(0, 1).all()  # Verify the range of test_f1_score
    assert result['train_f1_score'].between(0, 1).all()  # Verify the range of train_f1_score

# Test stacking method
def test_ensemble_compare_f1_stacking():
    result = ensemble_compare_f1(estimators, X_train, y_train, method='stacking')
    assert 'fit_time' in result.columns
    assert 'test_f1_score' in result.columns
    assert 'train_f1_score' in result.columns
    assert result['test_f1_score'].between(0, 1).all()  # Verify the range of test_f1_score
    assert result['train_f1_score'].between(0, 1).all()  # Verify the range of train_f1_score


# %%
test_ensemble_compare_f1_voting()
test_ensemble_compare_f1_stacking()

# %%
