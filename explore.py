# review = pd.read_csv(DIRECTORY + "sorted_reviews.csv",  parse_dates = ['date'], date_parser = dateparse, encoding = "latin-1")
# biz = pd.read_csv(DIRECTORY+"filtered_business.csv", encoding= "latin-1")


import argparse
import collections
import csv
import json
from constants import *
import pandas as pd
import datetime
import numpy as np


review_sorted = review.sort_values(by = 'business_id', 'date')
review_sorted.to_csv(DIRECTORY + "sorted_reviews.csv")


def extract_reviews_next(reader, review_file, colNames):
    review_df = pd.DataFrame(columns = colNames)
    first_time = 1
    for i, row in enumerate(reader):
        # do something here with `row`
        # print(row)
        if first_time:
            biz_id = row[1]
            first_time = 0 
        if not first_time:
            if row[1] != biz_id:
                break 
        review_df.loc[len(review_df)] = row 
    return review_df, reader

review_file = DIRECTORY+'sorted_reviews.csv'

with open(review_file, newline='') as f:
    reader = csv.reader(f)
    colNames = next(reader)
    while next(reader):
        review_df, reader =  extract_reviews_next(reader, review_file, colNames)
        print(review_df.shape)
        # do some analysis on this set of biz review
