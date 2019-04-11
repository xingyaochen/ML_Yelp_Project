#Split data from Yelp Dataset Challenge into train and test sets
from constants import *
from sklearn.model_selection import *
import pandas as pd
import numpy as np


bizfile = DIRECTORY + "filtered_business.csv"
bizData = pd.read_csv(bizfile, encoding= "latin-1")

def splitDat(data, percTest):
    """Splits data with specified test set percentage"""
    train, test= train_test_split(data, test_size= percTest)
    return train, test