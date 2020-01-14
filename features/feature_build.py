import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from dython.nominal import *

def feature_build_noNaN_cols(data):
    
    '''
    Select the features with no-missing values
    '''
    miss_val=(data.isnull().sum(axis=0))/len(data)*100
    cols=miss_val[miss_val == 0].index.tolist()
    fdata=data[cols]
    
    return fdata


def feature_catconv(df,cat_cols):
   '''
   Convert features listed in cat_cols to categories
   '''
    # Convert features to categories using one-hot-encoding 
   df[cat_cols]=df[cat_cols].astype('category')
    
   # data=pd.get_dummies(data, columns=cat_cols, drop_first=False,dummy_na=True)
   df=pd.get_dummies(df, columns=cat_cols, drop_first=False)

   print(data.info())
    
   return data


def feature_split(df,target,*args):
    '''
    Remove the columns in *args from dataframe
    '''
    y=df[target]
    X=df.drop(target,axis=1)

    X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=123)
            
    return X_train, X_test,y_train, y_test



def corr_return(training,validation,threshold):
    
    corr_matrix=training.corr().abs()
    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    
    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    return to_drop  



def selective_corr(training, validation, threshold, ofloat: True):
    
    if ofloat:
        train = training.select_dtypes(include='float64')
        valid = validation.select_dtypes(include='float64')
        
        to_drop = corr_return(train,valid,threshold)
        
        train=training.drop(training[to_drop], axis=1)    
        valid=validation.drop(training[to_drop], axis=1)
        
        return train, valid
    else:
        
        to_drop = corr_return(training,validation,threshold)
        train=training.drop(training[to_drop], axis=1)    
        valid=validation.drop(training[to_drop], axis=1)

    return train, valid