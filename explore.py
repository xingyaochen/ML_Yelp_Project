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



rolling_span = 4
with open(review_file, newline='') as f:
    reader = csv.reader(f)
    colNames = next(reader)
    while next(reader):
        review_df, reader =  extract_reviews_next(reader, review_file, colNames)
        average_over_span=review_df['stars'].rolling(rolling_span).mean()
        running_average= review_df["stars"].expanding().mean()
        review_df['average_over_span'] = average_over_span
        review_df['running_average'] = running_average 
        break
        # do some analysis on this set of biz review




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


# biz = pd.read_csv(DIRECTORY+"filtered_business.csv", encoding= "utf-8")
# # biz = biz.loc[biz['attributes'].isna()]


# attri_string = biz['attributes']


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

    features_df = features_df[feature_colnames]

    features_df_full = pd.concat([features_df, amb_df, goodMeal_df, bizPark_df], axis = 1)

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
    colnames_clean = list(features_df_clean)
    ohe_list = []
    for ft  in colnames_clean:
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
            feat_ohe.columns  = [ft+"_" + o for o in list(feat_ohe)]
            ohe_list.append(feat_ohe)
            
    featues_ohe =  pd.concat(ohe_list, axis = 1)
    featues_ohe_names = [n for n in list(featues_ohe) if 'False' not in n]
    return featues_ohe[featues_ohe_names]



biz = pd.read_csv(DIRECTORY+"filtered_business.csv", encoding= "utf-8")
biz_id = biz['business_id']
features_df_clean = construct_meta_features(biz)
featues_ohe = ohe_features(features_df_clean)

linked_featues_ohe = pd.concat([biz_id, featues_ohe], axis= 1)

np.random.seed(1234)
msk = np.random.rand(len(biz_id)) < 0.8
train_biz = set(biz_id[msk])
test_biz = set(biz_id[msk == False])

linked_featues_ohe_train = linked_featues_ohe.loc[linked_featues_ohe['business_id'].isin(train_biz)]
linked_featues_ohe_test = linked_featues_ohe.loc[linked_featues_ohe['business_id'].isin(test_biz)]

reviewfile = DIRECTORY + "sorted_reviews.csv"
reviewfile = DIRECTORY + "review.csv"
review = parseRatingOverTime(reviewfile)

train_data = linked_featues_ohe_train.merge(review, left_on = 'business_id', right_on = 'business_id')
test_data = linked_featues_ohe_test.merge(review, left_on = 'business_id', right_on = 'business_id')

# train_data = train_data.dropna(how = 'any')
# test_data = test_data.dropna(how = 'any')

X_train = train_data[list(linked_featues_ohe)[1:]]
y_train = train_data['running_average']

X_test = test_data[list(linked_featues_ohe)[1:]]
y_test = test_data['running_average']

reg = LassoCV(cv=8, random_state=0).fit(X_train, y_train)
reg.score(X_test, y_test)