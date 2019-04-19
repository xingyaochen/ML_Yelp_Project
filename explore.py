# review = pd.read_csv(DIRECTORY + "sorted_reviews.csv",  parse_dates = ['date'], date_parser = dateparse, encoding = "latin-1")
# biz = pd.read_csv(DIRECTORY+"filtered_business.csv", encoding= "latin-1")

import ast
import argparse
import collections
import csv
import json
from constants import *
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt 
from rating_time import *
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import OneHotEncoder

def extract_reviews_next(reader, review_file, colNames):
    review_df = pd.DataFrame(columns = colNames)
    first_time = 1
    for i, row in enumerate(reader):
        if first_time:
            biz_id = row[1]
            first_time = 0 
        else:
            if row[1] != biz_id:
                break 
        review_df.loc[len(review_df)] = row 
    return review_df, reader

# review_sorted = review.sort_values(by = 'business_id', 'date')
# review_sorted.to_csv(DIRECTORY + "sorted_reviews.csv")

review_file = DIRECTORY+'sorted_reviews.csv'

# rolling_span = 4
# with open(review_file, newline='') as f:
#     reader = csv.reader(f)
#     colNames = next(reader)
#     while next(reader):
#         review_df, reader =  extract_reviews_next(reader, review_file, colNames)
#         average_over_span=review_df['stars'].rolling(rolling_span).mean()
#         running_average= review_df["stars"].expanding().mean()
#         review_df['average_over_span'] = average_over_span
#         review_df['running_average'] = running_average 
#         break
    
def jsonL_to_df(jsonL):
    feature_names_set= set()
    for json_item in jsonL:
        if isinstance(json_item, str):
            d_attri = ast.literal_eval(json_item)
            if not d_attri:
                pass
            else:
                d_attri_names = set(d_attri.keys())
                feature_names_set = feature_names_set.union(d_attri_names)
        features_d = {feat: [] for i, feat in enumerate(list(feature_names_set))}
    # column_names = feature_names_d
    for json_item in jsonL:
        if isinstance(json_item, str):
            d_attri = ast.literal_eval(json_item)
            for key in features_d:
                if d_attri:
                    if d_attri.get(key):
                        features_d[key].append(d_attri[key])
                    else:
                        features_d[key].append(None)
                else:
                    features_d[key].append(None)
        else:
            for key in features_d:
                features_d[key].append(None)
    features_df = pd.DataFrame(features_d)
    return features_df



def strL_to_df(strL):
    feature_names_set= set()
    for s_group in strL:
        if isinstance(s_group, str):
            s_groupL = s_group.split(", ")
            if not s_groupL:
                pass
            else:
                d_attri_names = set(s_groupL)
                feature_names_set = feature_names_set.union(d_attri_names)
        features_d = {feat: [] for feat in list(feature_names_set)}
    # column_names = feature_names_d
    for s_group in strL:
        if isinstance(s_group, str):
            s_groupL = set(s_group.split(", "))
            # print(s_groupL)
            for key in features_d:
                if key in s_groupL:
                    features_d[key].append(True)
                else:
                    features_d[key].append(None)
        else:
            for key in features_d:
                features_d[key].append(None)
    features_df = pd.DataFrame(features_d)
    return features_df


def get_hours(str_time):
    if str_time:
        before, after = str_time.split('-')
        before_h = int(before.split(":")[0])
        after_h = int(after.split(":")[0])
        total = after_h - before_h
        if total < 0:
            total+=24 
        return total 
    else:
        return 0 
        


def construct_meta_features(biz_df):
    attri_string = biz_df['attributes']
    features_df = jsonL_to_df(attri_string)
    amb_json = features_df['Ambience']
    amb_df = jsonL_to_df(amb_json)

    goodMeal_json = features_df['GoodForMeal']
    goodMeal_df = jsonL_to_df(goodMeal_json)

    bizPark_json = features_df['BusinessParking']
    bizPark_df = jsonL_to_df(bizPark_json)

    feature_colnames = list(features_df)
    feature_colnames.remove('Ambience')
    feature_colnames.remove('GoodForMeal')
    feature_colnames.remove('BusinessParking') 

    categoryL = biz_df['categories']
    categories_df = strL_to_df(categoryL)

    hr_json = biz_df['hours']
    hr_df = jsonL_to_df(hr_json)
    for day in list(hr_df):
        hr_df[day] = hr_df[day].apply(lambda x: get_hours(x))

    features_df = features_df[feature_colnames]

    features_df_full = pd.concat([features_df, amb_df, goodMeal_df, bizPark_df, hr_df, categories_df], axis = 1)

    features_df_clean = features_df_full.dropna(axis = 1, thresh = int(0.2*features_df_full.shape[0]))
    return features_df_clean

def factorize_features(features_df_clean):
    colnames_clean = list(features_df_clean)
    features_factor = pd.DataFrame(data= np.empty(features_df_clean.shape), columns = colnames_clean)
    feature_mapping = {}
    for ft  in colnames_clean:
        feat = features_df_clean[ft] 
        # if isinstance(feat[0], str):
        try:
            feat = feat.str.replace("u'", "")
            feat = feat.str.replace("'", "")
            feat = [None if f == 'None' else f for f in feat]
            factors, labels = pd.factorize(feat, sort = True)
            feat_map = {} 
            sub = 0
            if labels[0] == "None":
                sub = 1 
            for i, lab in enumerate(labels):
                feat_map[i-sub] = lab 
            feature_mapping[ft] = feat_map
            features_factor[ft] = factors 
        except:
            # feature_mapping[ft] = feat_map
            print("hi", ft)
            features_factor[ft] = feat 
    return features_factor, feature_mapping

def ohe_features(features_df_clean):
    # line_contents_val = [d_attri[key] for key in column_names]
    daysOfWeek= ['Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday']
    colnames_clean = list(features_df_clean)
    ohe_list = []
    for ft  in colnames_clean:
        if ft in daysOfWeek:
            continue
        feat = features_df_clean[ft] 
        # if isinstance(feat[0], str):
        try:
            feat = feat.str.replace("u'", "")
            feat = feat.str.replace("'", "")
            feat = [None if f == 'None' else f for f in feat]
            feat_ohe = pd.get_dummies(feat)
            feat_ohe.columns  = [ft+"_" + o for o in list(feat_ohe)]
            ohe_list.append(feat_ohe)
        except:
            print(ft)
            feat_ohe = pd.get_dummies(feat)
            print(list(feat_ohe))
            feat_ohe.columns  = [ft+"_" + o for o in list(feat_ohe)]
            ohe_list.append(feat_ohe)
            
    features_ohe =  pd.concat(ohe_list, axis = 1)
    features_ohe_names = [n for n in list(features_ohe) if 'False' not in n]
    features_ohe = features_ohe[features_ohe_names]
    features_ohe_cont =  pd.concat([features_ohe, features_df_clean[daysOfWeek]], axis = 1)
    return features_ohe_cont


def abline(slope, intercept, axes = None):
    """Plot a line from slope and intercept"""
    if not axes:
        axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, 'r-', color="red")


def save_train_test(biz_file, review_file):
    biz = pd.read_csv(DIRECTORY+biz_file, encoding= "utf-8")

    biz_id = biz['business_id']
    features_df_clean = construct_meta_features(biz)

    featues_ohe = ohe_features(features_df_clean)

    linked_featues_ohe = pd.concat([biz_id, featues_ohe], axis= 1)

    reviewfile = DIRECTORY + review_file
    review = parseRatingOverTime(reviewfile)
    review_sorted = review.sort_values(by = ['business_id', 'date'])

    all_data = linked_featues_ohe.merge(review_sorted, left_on = 'business_id', right_on = 'business_id')

    all_biz = np.unique(all_data['business_id'])

    all_train_y = []
    all_test_y = []
    all_train_X = []
    all_test_X = []

    for biz_id in all_biz:
        biz_data = all_data[all_data['business_id'] == biz_id]
        biz_X = biz_data[list(linked_featues_ohe)[1:] + ['review_id']]
        biz_y = biz_data['running_average']
        biz_y_train, biz_y_test = biz_y[:int(len(biz_y)*0.8)], biz_y[int(len(biz_y)*0.8):]
        biz_X_train, biz_X_test = biz_X.iloc[:int(len(biz_y)*0.8)], biz_X.iloc[int(len(biz_y)*0.8):]
        all_train_X.append(biz_X_train)
        all_test_X.append(biz_X_test)
        all_train_y.append(biz_y_train)
        all_test_y.append(biz_y_test)

    all_train_X_pd  = pd.concat(all_train_X, axis = 0)
    all_train_y_pd  = pd.concat(all_train_y, axis = 0)
    all_train =  pd.concat([all_train_X_pd, all_train_y_pd], axis = 1)

    all_test_X_pd  = pd.concat(all_test_X, axis = 0)
    all_test_y_pd  = pd.concat(all_test_y, axis = 0)

    all_test =  pd.concat([all_test_X_pd, all_test_y_pd], axis = 1)
    all_train.to_csv(DIRECTORY+"training.csv")
    all_test.to_csv(DIRECTORY+"testing.csv")
