from data.load_data import load_dat
from features.feature_build import *



data=load_dat('../data/Consumer_Complaints.csv')
data=uniq_prod(data)

fdata=feature_build_noNaN(data)