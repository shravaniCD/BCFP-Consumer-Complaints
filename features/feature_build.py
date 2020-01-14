import pandas as pd
#import numpy as np

from sklearn.model_selection import train_test_split

import dython.nominal as dn

def feature_noNaNs(df):
    
    '''
    Select the features with no-missing values
    '''
    miss_val=(df.isnull().sum(axis=0))/len(df)*100
    cols=miss_val[miss_val == 0].index.tolist()
    df=df[cols]
    
    return df


def feature_OHEConv(df,cat_cols):
   '''
   Convert features listed in cat_cols to categories
   '''
    # Convert features to categories using one-hot-encoding 
   df[cat_cols]=df[cat_cols].astype('category')
    
   # data=pd.get_dummies(data, columns=cat_cols, drop_first=False,dummy_na=True)
   df=pd.get_dummies(df, columns=cat_cols, drop_first=False)

   print(df.info())
    
   return df


def feature_target_split(df,target,*args):
    '''
    Remove the columns in *args from dataframe
    '''
    y=df[target]
    X=df.drop(target,axis=1)

    X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=123)
            
    return X_train, X_test,y_train, y_test



def feature_corr(df,cat_feats):
    
    dn.associations(df,nominal_columns=cat_feats,theil_u=True)
    
    return