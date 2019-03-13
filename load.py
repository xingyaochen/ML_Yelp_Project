
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

def read_and_write_csv(json_file_path, csv_file_path, column_names, filerD = {}):
    """Read in the json dataset file and write it out to a csv file, given the column names."""
    with open(csv_file_path, 'w') as fout:
        csv_file = csv.writer(fout)
        csv_file.writerow(column_names)
        with open(json_file_path) as fin:
            for line in fin:
                write_flag = 1
                line_contents = json.loads(line)
                if filerD:
                    for k, val in filerD.items():
                        value = line_contents.get(key)
                        if value and value != val:                
                            write_flag = 0
                if write_flag:
                    csv_file.writerow(line_contents.values())

json_file_path = DIRECTORY + 'business.json'
csv_file_path = DIRECTORY + 'business.csv'
column_names = BIZ_NAMES 
