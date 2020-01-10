import pandas as pd

def load_dat(path):
    data=pd.read_csv(path,index_col='Complaint ID',true_values=['Yes'],false_values=['No'],encoding='ISO-8859-1', parse_dates = ['Date received','Date sent to company'],infer_datetime_format=True,na_values=['NaN'])
    
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
    
    
    
