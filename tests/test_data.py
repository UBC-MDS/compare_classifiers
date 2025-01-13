import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

import os

seed = 524

def test_data():
    """Create training and test data for function tests. Dataset is checked, cleaned and StandardScaled."""
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'test_data.csv'))
    data['is_red'] = data['color'].apply(lambda x: 1 if x == 'red' else 0)
    data = data.drop(['color'], axis=1)
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=seed)
    X_train, X_test, y_train, y_test = (train_df.drop(columns='quality'), test_df.drop(columns='quality'),
                                    train_df['quality'], test_df['quality']
                                    )
    ss = StandardScaler()
    X_train_ss = ss.fit_transform(X_train)
    X_test_ss = ss.transform(X_test)

    rs = RobustScaler()
    rs.fit(X_train)
    X_test_rs = rs.transform(X_test)

    return {'X_train': X_train, 'X_train_ss': X_train_ss, 'X_test_ss': X_test_ss, 'X_test_rs': X_test_rs, 'y_train': y_train, 'y_test': y_test}


def models():
    """Create models as estimators for function tests.
    Note: Please use individual classifiers with X_train_ss and X_test_ss and pipeline with X_train and X_test_rs"""
    rf = RandomForestClassifier(n_estimators=10, random_state=seed)
    svm = SVC(kernel='rbf', decision_function_shape='ovr', random_state=seed)
    logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=seed)
    gb = GradientBoostingClassifier(random_state=seed)
    knn5 = KNeighborsClassifier(n_neighbors=5)
    mnp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=seed)
    pipe = make_pipeline(RobustScaler(), svm)
    
    return {'rf': rf, 'svm': svm, 'logreg': logreg, 'gb': gb, 'knn5': knn5, 'mnp': mnp, 'pipe': pipe}

