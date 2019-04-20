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

#n*36 pandas frame metadata on same row as review
#y = n cumulative review ratings
def singleBizInterval(startperc,singleBiz):
    #get number of reviews for specific business maybe use num rows in single biz
    numreviews = 
    #get starting and ending rows for train and validation reviews
    trainstart = numreviews*startperc
    trainend = (numreviews * 0.5) + trainstart
    valstart = trainend + 1 
    valend = valstart + (numreviews * 0.1)
    #index into df and just get rows between review start and review end
    return trainx, trainy, valx, valy

def getBizIntervals(startperc):
    #need to add in dataframe here
    businesses = np.unique(['business_id'])
    for biz in businesses:
        currentdf = df.loc[df['business_id'] == biz]
        #call single biz interval here
    #somehow concatenate all of these into one dataframe again
    
    
#get smaller data frame for each business



# #get reviews for all businesses
# reviewfile = DIRECTORY + "review.csv"
# #parse the ratings

# # print(reviews)

# #chunk reviews by business id
# dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
# review = pd.read_csv(DIRECTORY + "review.csv",  parse_dates = ['date'], date_parser = dateparse, encoding = "latin-1")
# review_sorted = review.sort_values(by = ['business_id', 'date'])
# review_sorted.to_csv(DIRECTORY + "sorted_reviews.csv")
# rateTime = parseRatingOverTime(DIRECTORY + "sorted_reviews.csv")

# print (rateTime)