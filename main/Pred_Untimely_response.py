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
print('Modified data loaded!')
#%%
# Prediction of Untimely response
'''
Use CatBoost to predict if there will be 'Untimely response'.
The data contains 97.6% False values  and 2.4% True values making this a highly imbalanced target problem.
Hence, it important to choose a metric that accounts for this imbalance. One of them is Gini=2*AUC -1, that maximizes the true positive predictions while minimising the false positives.  

The features that are relevant for this prediction are: Product, Issue, State, 
                                                        Company, Submitted via, 
                                                        the no.of days it took for the cpmany to receive the complaint. 
                                                        
All the other features are a result of (Un)timely response. Therefore, We'll ignore the Company response to consumer, 
                                                                                        Company public response, 
                                                                                        Consumer consent provided?, 
                                                                                        Consumer disputed?,
                                                                                        Consumer complaint narrative
                                                            
Apart from these, the absolute date of the complaint, sub-issue, sub-product will also be ignored as they are highly correalted to NO.of days, Product and sub-issue. 
'''
# =============================================================================
to_drop=['Sub-issue','Sub-product','Date received','Consumer complaint narrative',
         'Company public response','Company response to consumer', 
         'Consumer consent provided?', 'Consumer disputed?']

ndata=data.drop(to_drop,axis=1)
print('Features included in the Trainiing:',ndata.columns.tolist())

target='Untimely response?'

X_train, X_test, y_train, y_test = feature_target_split(ndata,target)

# Select the categorical features
cat_feats = ndata.select_dtypes('object').columns.tolist()

cat_up= CatBoostModel()
cat_up.fit(X_train,y_train,cat_features=cat_feats)

print('Gini score for Predicition of Untimely response from company: ')
print('Train set =',cat_up.score(X_train,y_train))
print('Test set =',cat_up.score(X_test,y_test))
#
cat_up.save('D:\Profile\lmw\Desktop\Kaggle\Consumer_complaints\data\Pred_Untimely_response.sav')
#%%
cat_up= pickle.load(open('D:\Profile\lmw\Desktop\Kaggle\Consumer_complaints\data\Pred_Untimely_response.sav','rb'))