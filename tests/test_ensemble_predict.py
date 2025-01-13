import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from compare_classifiers.ensemble_predict import ensemble_predict

from tests.test_data import test_data, models

import numpy as np

from sklearn.ensemble import VotingClassifier


data_dict = test_data()
X_train_ss = data_dict['X_train_ss']
X_test_ss = data_dict['X_test_ss']
X_test_rs = data_dict['X_test_rs']
y_train = data_dict['y_train']
y_test = data_dict['y_test']

model_dict = models()
knn5_and_mnb = [
    ('knn5', model_dict['knn5']),
    ('mnp', model_dict['mnp'])
]

def test_voting_success():
    vc = VotingClassifier(knn5_and_mnb)
    vc = vc.fit(X_train_ss, y_train)
    predictions = vc.predict(X_test_ss)
    assert(np.all(ensemble_predict(knn5_and_mnb, X_train_ss, y_train, 'voting', X_test_ss) == predictions))