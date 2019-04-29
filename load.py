# bizData = "yelp_dataset/business.json"
# biz_colNames = ["business_id", "name", "city", "state", \
#      "postal code",  "latitude", "longitude",  "stars", "review_count", \
#          "is_open", "attributes",  "categories", "hours"] 
# csv_file_path = "yelp_dataset/business.csv"

# -*- coding: utf-8 -*-
""" Convert the Yelp Dataset Challenge dataset from json format to csv.
    For more information on the Yelp Dataset Challenge please visit http://yelp.com/dataset_challenge
"""
import argparse
import collections
import csv
import json
from constants import *
import pandas as pd
import datetime
import numpy as np

def read_and_write_csv(json_file_path, csv_file_path, filterD = {}):
    """ Read in the json dataset file and write it out to a csv file, given the column names. """
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
                        if value:
                            if isinstance(val, float) or isinstance(val, int):
                                if value < val:                
                                    write_flag = 0
                            elif isinstance(val, set) and isinstance(value, str):
                                valueSet = set(value.split(', '))
                                if not val.intersection(valueSet):
                                    write_flag = 0
                if write_flag:
                    line_contents_val = [line_contents[key] for key in column_names]
                    csv_file.writerow(line_contents_val)   


# make business csv

# # filterD = {'city': {"Las Vegas"}}
def main():
    filterD = {'city': {'Las Vegas'}, 'review_count': 3500, 'categories': {'Food', 'Restaurants','Bars', 'Breakfast', 'Lunch', 'Dinner', 'Eatertainment'}}

    json_file_path = DIRECTORY + 'business.json'
    csv_file_path = DIRECTORY + 'business.csv'
    read_and_write_csv(json_file_path, csv_file_path, filterD)
    business = pd.read_csv (DIRECTORY + "business.csv", encoding = "latin-1")
    city, counts = np.unique(business['city'], return_counts=True)

    # change this value to change the threshold value of number of businesses in the area
    threshold = 0

    # filter by city and number of businesses in that city
    cities_alot = city[counts > threshold]
    business.set_index('city', inplace=True)
    business_filtered = business.loc[cities_alot]
    business_filtered.to_csv(DIRECTORY + "filtered_business.csv")

    # make review csv
    # # change the "business.csv" to "filtered_business.csv" if the location filtering is desired
    biz = pd.read_csv(DIRECTORY+"filtered_business.csv", encoding= "latin-1")
    biz_id = biz['business_id']
    filterD_reviews = {'business_id': set(biz_id)}
    json_review_path = DIRECTORY + 'review.json'
    csv_review_path = DIRECTORY + 'review.csv'
    read_and_write_csv(json_review_path, csv_review_path, filterD_reviews)
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    
    review = pd.read_csv(DIRECTORY + "review.csv",  parse_dates = ['date'], date_parser = dateparse, encoding = "latin-1")
    print(review.shape)
    # review_sorted = review.sort_values(by = ['business_id', 'date'])
    # review_sorted.to_csv(DIRECTORY + "sorted_reviews.csv")

if __name__ == "__main__":
    # pass
    main()


# filterD = {'city': {'Las Vegas'}, 'review_count': 3500, 'categories': {'Food', 'Restaurants','Bars', 'Breakfast', 'Lunch', 'Dinner', 'Eatertainment'}}

# json_file_path = DIRECTORY + 'business.json'
# csv_file_path = DIRECTORY + 'business.csv'
# read_and_write_csv(json_file_path, csv_file_path, filterD)
# business = pd.read_csv (DIRECTORY + "business.csv", encoding = "latin-1")
# city, counts = np.unique(business['city'], return_counts=True)

# # change this value to change the threshold value of number of businesses in the area
# threshold = 0

# # filter by city and number of businesses in that city
# cities_alot = city[counts > threshold]
# business.set_index('city', inplace=True)
# business_filtered = business.loc[cities_alot]
# business_filtered.to_csv(DIRECTORY + "filtered_business.csv")

# # make review csv
# # # change the "business.csv" to "filtered_business.csv" if the location filtering is desired
# biz = pd.read_csv(DIRECTORY+"filtered_business.csv", encoding= "latin-1")
# biz_id = biz['business_id']
# filterD_reviews = {'business_id': set(biz_id)}
# json_review_path = DIRECTORY + 'review.json'
# csv_review_path = DIRECTORY + 'review.csv'
# read_and_write_csv(json_review_path, csv_review_path, filterD_reviews)
# dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

# review = pd.read_csv(DIRECTORY + "review.csv",  parse_dates = ['date'], date_parser = dateparse, encoding = "latin-1")
# # print(review.shape)
