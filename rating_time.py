import pandas as pd
import csv
import json
from constants import *
import datetime
import numpy as np
def parseRatingOverTime(reviewfile,rolling_span=4):
    """
    parse the review csv and returns a pandas df with calculated ratings
    """
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    #read in the review csv
    review = pd.read_csv(reviewfile, parse_dates = ['date'], date_parser = dateparse, encoding = "latin-1")
    #group the reviews by business_id and sort within each business by date of the reviews
    group_biz = review.groupby(["business_id"], as_index=False).apply(lambda x: x.sort_values(['date'], ascending = True)).reset_index(drop=True)
    # group_biz['average_over_span'] = group_biz.groupby('business_id')['stars'].rolling(rolling_span).mean()
    average_over_span=group_biz.groupby('business_id')['stars'].rolling(rolling_span).mean()
    group_biz["average_over_span"] = average_over_span.reset_index(level=0, drop=True)
    # group_biz["running_average"] = group_biz.groupby("business_id", as_index=False).apply(lambda x: x["stars"].expanding().mean())
    running_average= group_biz.groupby("business_id", as_index=False).apply(lambda x: x["stars"].expanding().mean())
    group_biz["running_average"] = running_average.reset_index(level=0, drop=True)
    group_biz = group_biz[['business_id','date','average_over_span','running_average', 'review_id','text']]
    reviewfile_name=reviewfile.split(".")[0]
    group_biz.to_csv(reviewfile_name+"_ratingOverTime.csv", encoding="latin-1", index=False)
    return group_biz
    

def main():
    reviewfile = DIRECTORY + "review.csv"
    # group_biz=
    parseRatingOverTime(reviewfile)
    

if __name__ == "__main__":
    main()
