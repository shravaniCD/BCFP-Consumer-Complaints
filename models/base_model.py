import pandas as pd

from catboost import CatBoostClassifier,Pool
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.model_selection import RandomizedSearchCV

import pickle
import shap

import matplotlib.pyplot as plt


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

    def shap_sum(self, features, labels,filename:str, **kwargs):
        shap_values=self.model.get_feature_importance(Pool(features, label=labels, **kwargs),type='ShapValues')
        filename="D:\Profile\lmw\Desktop\Kaggle\Consumer_complaints\data\\" + filename + '.sav'
        pickle.dump(shap_values, open(filename,'wb'))
        print('Summary of SHAP values saved!')
        shap_values=shap_values[:,:-1]
        shap.initjs()
        shap.summary_plot(shap_values, features)
        plt.show()
    
    def shap_ind(self, features, ind,filename:str):
        filename="D:\Profile\lmw\Desktop\Kaggle\Consumer_complaints\data\\" + filename + '.sav'
        shap_values=pickle.load(open(filename,'rb'))
        expected_value=shap_values[0,-1]
        shap_values=shap_values[:,:-1]
        #shap.force_plot(expected_value, shap_values[ind,:], features.iloc[ind,:])
        return expected_value,shap_values[ind,:], features.iloc[ind,:]

class CatBoostModel(AbstractModel):

    def __init__(self, **kwargs):
        self.model = CatBoostClassifier(thread_count=4, loss_function='Logloss',eval_metric='AUC', verbose=10, l2_leaf_reg=1, **kwargs)