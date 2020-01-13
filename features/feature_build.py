import pandas as pd
import numpy as np

def uniq_Prod_Iss(data):
    
    # PRODUCT
    # merging credit reporting with general case of credit reporting
    data.loc[((data['Product'] == 'Credit reporting') & (data['Sub-product'].isnull())), 'Sub-product']='Credit reporting'
    data.loc[((data['Product'] == 'Credit reporting') & (data['Sub-product'] == 'Credit reporting')), 'Product']='Credit reporting, credit repair services, or other personal consumer reports'
    
    # merging 'paydayloan' with general case of loans
    data.loc[((data['Product'] == 'Payday loan') & (data['Sub-product'].isnull())), 'Sub-product']='Payday loan'
    data.loc[((data['Product'] == 'Payday loan') & (data['Sub-product'] == 'Payday loan')), 'Product']='Payday loan, title loan, or personal loan'
    
    # separating the 'vehicle lease/loan' from consumer loan and merge with main product vehicle lease/loan
    data.loc[((data['Product'] == 'Consumer Loan') & ((data['Sub-product']=='Vehicle loan') | (data['Sub-product']=='Vehicle lease'))), 'Product']='Vehicle loan or lease'
    data.loc[((data['Product'] == 'Vehicle loan or lease') & (data['Sub-product'] == 'Vehicle loan')), 'Sub-product']='Loan'
    data.loc[((data['Product'] == 'Vehicle loan or lease') & (data['Sub-product'] == 'Vehicle lease')), 'Sub-product']='Lease'
    
    # merge the general loans with 'consumer loan'
    data.loc[(data['Product'] == 'Payday loan, title loan, or personal loan'), 'Product']='Consumer Loan'
    
    # merge the indivisual products virtual conrrency/money transfer with general case
    data.loc[data['Product'] == 'Virtual currency', 'Product']='Money transfer, virtual currency, or money service'
    data.loc[data['Product'] == 'Money transfers', 'Product']='Money transfer, virtual currency, or money service'
    
    # merge the ban account servce
    data.loc[data['Product'] == 'Checking or savings account', 'Product']='Bank account or service'
    data.loc[((data['Product'] == 'Bank account or service') & (data['Sub-product'] == '(CD) Certificate of deposit')), 'Sub-product']='CD (Certificate of Deposit)'
    data.loc[((data['Product'] == 'Bank account or service') & (data['Sub-product'] == 'Other banking product or service')), 'Sub-product']='Other bank product/service'
    
    data.loc[((data['Product'] == 'Credit card') & (data['Sub-product'].isnull())), 'Sub-product']='General-purpose credit card or charge card'
    data.loc[((data['Product'] == 'Credit card') & (pd.DatetimeIndex(data['Date received']).year<=2017)),'Product']='Credit card or prepaid card'
    
    data.loc[((data['Product'] == 'Prepaid card') & (data['Sub-product']=='General purpose card')), 'Sub-product']='General-purpose prepaid card'
    data.loc[((data['Product'] == 'Prepaid card') & (data['Sub-product']=='Government benefit payment card')), 'Sub-product']='Government benefit card'
    data.loc[((data['Product'] == 'Prepaid card') & (pd.DatetimeIndex(data['Date received']).year<=2017)), 'Product']='Credit card or prepaid card'
    
    data['State']=data['State'].str.replace('UNITED STATES MINOR OUTLYING ISLANDS', 'USIs', regex=True)
    data['Product']=data['Product'].str.replace('Credit reporting, credit repair services, or other personal consumer reports', 'Credit reporting etc', regex=True)
    data['Product']=data['Product'].str.replace('Money transfer, virtual currency, or money service', 'Money transfer etc', regex=True)
    data['Product']=data['Product'].str.replace('Credit card or prepaid card', 'Credit/Prepaid card', regex=True)
    data['Product']=data['Product'].str.replace('Checking or savings account', 'Checking/savings account', regex=True)
    data['Product']=data['Product'].str.replace('Bank account or service', 'Bank account/service', regex=True)
    data['Product']=data['Product'].str.replace('Vehicle loan or lease', 'Vehicle loan/lease', regex=True)
    
    
    # ISSUE
    # Removing the redundant Issue types..that differ only by an article or punctuation
    comon_is=dict()
    for subi in data['Issue'].unique().tolist():
        for subj in data['Issue'].unique().tolist():
            if subi!=subj and subj!=np.nan and subi!=np.nan:
                if len(set(str.split(subi)))>1 and len(set(str.split(subi)))>1:
                    if abs(len(set(str.split(subi))&set(str.split(subj))))>=len(str.split(subi))-1 and abs(len(set(str.split(subi))&set(str.split(subj))))>=len(str.split(subj))-1:  
                        if (subj not in comon_is.keys()) or (subj in comon_is.values()):
                            comon_is[subi]=subj

    to_del1=['Managing an account','Late fee','Closing/Cancelling account','Struggling to pay your loan','Other fee','Credit determination','Billing disputes','Balance transfer fee','Other transaction issues']
    for i in to_del1:
        del comon_is[i]
        
    data.Issue=data.Issue.replace(comon_is,regex=True)

    return data

def feature_build_noNaN_cols(data):
    
    '''
    Select the features with no-missing values
    '''
    miss_val=(data.isnull().sum(axis=0))/len(data)*100
    cols=miss_val[miss_val == 0].index.tolist()
    fdata=data[cols]
    
    return fdata


def feature_catconv(data,cat_cols):
   '''
   Convert features listed in cat_cols to categories
   '''
    # Convert features to categories using one-hot-encoding 
   data[cat_cols]=data[cat_cols].astype('category')
    
   # data=pd.get_dummies(data, columns=cat_cols, drop_first=False,dummy_na=True)
   data=pd.get_dummies(data, columns=cat_cols, drop_first=False)

   print(data.info())
    
   return data


def feature_rm(data,*args):
    '''
    Remove the columns in *args from dataframe
    '''
    for i in args:
        data=data.drop(i,axis=1)
        
    return data



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