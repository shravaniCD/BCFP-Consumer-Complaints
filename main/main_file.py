import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#from data.load_data import *
from features.feature_build import *
from models.base_model import *
import pickle

#%%

'''
Load data 
Rearrange the categories in the prodcut and Issue columns as some of them are redundant
Save the modified data
'''
# =============================================================================

# data=clean_load_dat('../data/Consumer_Complaints.csv')

# filename='D:\Profile\lmw\Desktop\Kaggle\Consumer_complaints\data\mdf_Consumer_Complaints.csv'
# pickle.dump(data, open(filename, 'wb'))

# Load saved dataframe
data=pickle.load(open('D:\Profile\lmw\Desktop\Kaggle\Consumer_complaints\data\mdf_Consumer_Complaints.csv', 'rb'))

#%%
# Prediction of Untimely response across the states/issues
'''
Use CatBoost to predict if there will be 'Untimely response'.
The data contains 97.6% False values  and 2.4% True values making this a highly imbalanced target problem.
Hence, it important to choose a metric that accounts for this imbalance. One of them is Gini=2*AUC -1, that maximizes the true positive predictions while minimising the false positives.  
'''
# =============================================================================

to_drop=['Sub-issue','Sub-product','Date received','Consumer complaint narrative',
         'Company public response','Company response to consumer', 
         'Consumer consent provided?', 'Consumer disputed?']

ndata=data.drop(to_drop,axis=1)

cat_feats = ndata.select_dtypes('object').columns.tolist()

target='Untimely response?'

X_train, X_test,y_train, y_test = feature_split(ndata,target)

cat_up= CatBoostModel()
cat_up.fit(X_train,y_train,cat_features=cat_feats)


print('Gini for train set =',cat_up.score(X_train,y_train))
print('Gini for test set =',cat_up.score(X_test,y_test))