
# bizData = "yelp_dataset/business.json"
# biz_colNames = ["business_id", "name", "city", "state", \
#      "postal code",  "latitude", "longitude",  "stars", "review_count", \
#          "is_open", "attributes",  "categories", "hours"] 
# csv_file_path = "yelp_dataset/business.csv"

# -*- coding: utf-8 -*-
"""Convert the Yelp Dataset Challenge dataset from json format to csv.
For more information on the Yelp Dataset Challenge please visit http://yelp.com/dataset_challenge
"""
import argparse
import collections
import csv
import json
from constants import *
import pandas as pd
import datetime


def read_and_write_csv(json_file_path, csv_file_path, filterD = {}):
    """Read in the json dataset file and write it out to a csv file, given the column names."""
    with open(csv_file_path, 'w') as fout:
        csv_file = csv.writer(fout)
        with open(json_file_path) as fin:
            for line in fin:
                write_flag = 1
                line_contents = json.loads(line)
                line_contents_k = sorted(line_contents.keys())
                column_names = line_contents_k
                break
        csv_file.writerow(column_names)
        with open(json_file_path) as fin:
            for line in fin:
                write_flag = 1
                line_contents = json.loads(line)
                if filterD:
                    for k, val in filterD.items():
                        value = line_contents.get(k)
                        if value and value not in val:                
                            write_flag = 0
                if write_flag:
                    line_contents_val = [line_contents[key] for key in column_names]
                    csv_file.writerow(line_contents_val)

#make business csv
filterD = {'city': {"Las Vegas"}}
json_file_path = DIRECTORY + 'business.json'
csv_file_path = DIRECTORY + 'business.csv'
read_and_write_csv(json_review_path, csv_review_path, filterD_reviews)


# make review csv
biz = pd.read_csv(DIRECTORY+"business.csv", encoding= "latin-1")
biz_id = biz['business_id']
filterD_reviews = {'business_id': set(biz_id)}
json_review_path = DIRECTORY + 'review.json'
csv_review_path = DIRECTORY + 'review.csv'
read_and_write_csv(json_review_path, csv_review_path, filterD_reviews)
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
review = pd.read_csv(DIRECTORY+"review.csv",  parse_dates=['date'], date_parser=dateparse, encoding= "latin-1")

