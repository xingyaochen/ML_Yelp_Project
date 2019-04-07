import panda as pd
import csv
import json
from constants import *
import datetime
import numpy as np
def parseRatingOverTime(reviewfile,rolling_span=4):
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    #read in the review csv
    review = pd.read_csv(reviewfile, parse_dates = ['date'], date_parser = dateparse, encoding = "latin-1")
    #group the reviews by business_id and sort within each business by date of the reviews
    group_biz= review.groupby(["business_id"]).apply(lambda x: x.sort_values(['date'], ascending = True)).reset_index(drop=True)
    group_biz['average_over_span'] = group_biz.groupby(["business_id"]).rolling(rolling_span).group_biz["stars"].mean()
    group_biz["running_average"] = group_biz.groupby("business_id").apply(lambda x: x["stars"].expanding().mean())
    group_biz = group_biz[['business_id','date','average_over_span','running_average']]
    reviewfile_name=reviewfile.split(".")[0]
    group_biz.to_csv(DIRECTORY+reviewfile_name+"_ratingOverTime.csv", encoding="latin-1", index=False)
    


def main():
    reviewfile=DIRECTORY + "review.csv"
    parseRatingOverTime(reviewfile)

if __name__ == "__main__":
    main()