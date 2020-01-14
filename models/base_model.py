import pandas as pd

from catboost import CatBoostClassifier
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.model_selection import RandomizedSearchCV

import pickle


class AbstractModel:

    def __init__(self):
        pass

    def fit(self, features, labels, **kwargs):
        self.model.fit(features, labels, **kwargs)

    def predict(self, features):
        predictions = self.model.predict_proba(features)[:, 1]
        return predictions

    def score(self, features, labels):
        model_predictions = self.predict(features)
        auc = roc_auc_score(labels, model_predictions)
        gini = 2 * auc - 1
        return gini

    def ROC(self, features, labels):
        model_predictions = self.predict(features)
        fpr, tpr, thresholds = roc_curve(labels, model_predictions)
        return fpr, tpr

    def save(self, filename: str):
        pickle.dump(self, open(filename, 'wb'))


class CatBoostModel(AbstractModel):

    def __init__(self, **kwargs):
        self.model = CatBoostClassifier(thread_count=4, loss_function='Logloss',eval_metric='AUC', verbose=10, l2_leaf_reg=5, **kwargs)