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
folds = 4
def get_cv_fold(cross_validation, fold_num):
    data_CV = cross_validation.loc[cross_validation['foldNum']==fold_num]
    train_CV = data_CV.loc[data_CV['set'] == 'training']
    validate_CV = data_CV.loc[data_CV['set'] == 'validation']
    print(train_CV.shape, validate_CV.shape)
    return train_CV, validate_CV

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
    crossValSets = getBizIntervals(startVals, train_data)
    crossValSets.to_csv(DIRECTORY+"crossVal.csv", encoding="latin-1", index=False)
    print('done')

def getBizIntervals(startVals, train_data):
    """For each business in train_data, grabs 50% of that business' reviews
    starting at startperc and then grabs the following 10% to be used as validation"""
    #get list of businesses
    all_biz, indx = np.unique(train_data['business_id'], return_index=True)
    indx = sorted(indx) + [train_data.shape[0]]
    indx_ranges = [(indx[i], indx[i+1]) for i in range(len(indx)-1)]
    #for each business separate the correct number of reviews
    finaldf = []
    for indx_r in indx_ranges:
        currentdf = train_data.iloc[indx_r[0]:indx_r[1]]
        finaldf.append(singleBizInterval(startVals, currentdf))
        # print(indx_r)
    #put all the reviews back into dataframes
    finaldf = pd.concat(finaldf, axis =0)
    return finaldf

def singleBizInterval(startVals,singleBiz):
    """Indexes into dataframe of reviews for a single business (singleBiz).
    Gets 50% of all reviews starting at startperc and then puts the following 10% into
    a validation set"""
    #get number of reviews for specific business
    numreviews = singleBiz.shape[0]
    dataList = []
    for i in range(len(startVals)):
        #get starting and ending rows for train and validation reviews
        trainstart = int(numreviews*startVals[i])
        trainend = int((numreviews * 0.5) + trainstart)
        valstart = trainend
        valend = int(valstart + (numreviews * 0.1))
        #index into df and just get rows between review start and review end
        trainData = singleBiz.iloc[trainstart:trainend]
        trainData.is_copy = None
        trainData['set'] = 'training'
        valData = singleBiz.iloc[valstart:valend]
        valData.is_copy = None
        valData['set'] = 'validation'
        #concatenate the dataframes with set labels and add a label for the fold
        allDat = pd.concat([trainData,valData], axis=0)
        allDat['foldNum'] = i
        dataList.append(allDat)
    returndf = pd.concat(dataList)
    return returndf

def main():
    currdata = pd.read_csv(DIRECTORY+"training.csv", encoding= "utf-8")
    folds = 4
    crossValidation(currdata, folds)

if __name__ == "__main__":
    main()