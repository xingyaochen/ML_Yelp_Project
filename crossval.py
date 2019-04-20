import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt 
import random
from rating_time import *
from explore import *

#hold out percentage
#train on %50 of training data and then validate on following 10%
# choose random int between 0 and 40 train on 50% of all reviews starting from that point
# then validate on 10% of reviews directly after
currdata = pd.read_csv(DIRECTORY+"training.csv", encoding= "utf-8")

def singleBizInterval(startperc,singleBiz):
    """Indexes into dataframe of reviews for a single business (singleBiz).
    Gets 50% of all reviews starting at startperc and then puts the following 10% into
    a validation set"""
    #get number of reviews for specific business
    numreviews = singleBiz.shape[0]
    #get starting and ending rows for train and validation reviews
    trainstart = int(numreviews*startperc)
    trainend = int((numreviews * 0.5) + trainstart)
    valstart = trainend
    valend = int(valstart + (numreviews * 0.1))
    #index into df and just get rows between review start and review end
    trainData = singleBiz.iloc[trainstart:trainend]
    valData = singleBiz.iloc[valstart:valend]
    return trainData, valData

def getBizIntervals(startperc, train_data):
    """For each business in train_data, grabs 50% of that business' reviews
    starting at startperc and then grabs the following 10% to be used as validation"""
    #get list of businesses
    all_biz = np.unique(train_data['business_id'])
    #for each business separate the correct number of reviews
    validationdf = []
    traindf = []
    for biz in all_biz:
        currentdf = train_data[train_data['business_id'] == biz]
        singTrain, singVal = singleBizInterval(startperc, currentdf)
        validationdf.append(singVal)
        traindf.append(singTrain)
    #put all the reviews back into dataframes
    validationdf = pd.concat(validationdf, axis =0)
    traindf = pd.concat(traindf, axis=0)
    return validationdf, traindf

def crossValidation(train_data, numfolds):
    """Returns list where each element is a list containing a 
    training dataframe and validation dataframe"""
    #figuring out starting percents
    sepVal = 40/numfolds
    i = 0
    startVals = []
    while i <= 40:
        startVals.append(i/100)
        i+= sepVal
    #now get train and validation sets given each start value
    crossValSets = []
    for start in startVals:
        crossValSets.append([getBizIntervals(start, train_data)])
    print(len(crossValSets))
    return crossValSets
