import pandas as pd
import numpy as np

def load_dat(path):
    data=pd.read_csv(path,index_col='Complaint ID',encoding='ISO-8859-1', parse_dates = ['Date received','Date sent to company'],infer_datetime_format=True,na_values=['NaN'])
#    data=pd.read_csv(path,index_col='Complaint ID',true_values=['Yes'],false_values=['No'],encoding='ISO-8859-1', parse_dates = ['Date received','Date sent to company'],infer_datetime_format=True,na_values=['NaN'])
    
    data['Timely response?']=data['Timely response?'].map({'Yes':0,'No':1})
    data.rename(columns={'Timely response?': 'Untimely response?'}, inplace=True)
    
    # drop the 'nan' entries of 'state' and 'company resonse to consumer'
    data=data.dropna(subset=['State'])
    data=data.dropna(subset=['Company response to consumer'])

    # drop 'tags'column as it is sparse
    data=data.drop('Tags',axis=1)
    data=data.drop('ZIP code',axis=1)
        
    # introduce new feature 'delay in complaint transmission to company'
    data['SentToCompanyIn(days)']=((data['Date sent to company']-data['Date received']).astype('timedelta64[D]'))
    data=data.drop('Date sent to company',axis=1)
    
    print('Removed features: Tags, ZIP code, Date sent to company')
    print('Created features: SentToCompanyIn(days)')
    
    return data
    

def uniq_Prod_Iss(data):
    print('Removing the redundant product/Issue types...')
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


def clean_load_dat(path):
    
    data=load_dat(path)
    data=uniq_Prod_Iss(data)
    
    return data