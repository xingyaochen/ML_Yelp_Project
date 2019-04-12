#Split data from Yelp Dataset Challenge into train and test sets
from constants import *
from sklearn.model_selection import *
import pandas as pd
import numpy as np


bizfile = DIRECTORY + "filtered_business.csv"
bizData = pd.read_csv(bizfile, encoding= "latin-1")

def splitBusinessID(data, percTrain):
    """Splits data with specified training set percentage 
    returns train and test dataframes"""
    # train, test= train_test_split(data, test_size= percTest, random_state = 1234)
    # return train, test
    np.random.seed(1234)
    msk = np.random.rand(len(biz_id)) < percTrain
    train_biz = set(biz_id[msk])
    test_biz = set(biz_id[msk == False])
    return train_biz, test_biz
