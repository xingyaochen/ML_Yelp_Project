from constants import *
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import conda

conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
os.environ["PROJ_LIB"] = proj_lib

from mpl_toolkits.basemap import Basemap

def map_plot(longs, lats, ratings, count, counter):
    zoom_scale = 1
    bbox = [np.min(lats)-zoom_scale,np.max(lats)+zoom_scale, np.min(longs)-zoom_scale,np.max(longs)+zoom_scale]

    plt.figure(figsize=(12,6))

    m = Basemap(projection='merc',llcrnrlat=bbox[0], urcrnrlat=bbox[1], llcrnrlon=bbox[2],urcrnrlon=bbox[3],lat_ts=10,resolution='h')
    m.drawcoastlines(color='gray')
    m.fillcontinents(color='#CCCCCC', zorder=0)
    m.drawstates(color='black')
    m.drawcountries(color='gray')

    m.drawparallels(np.arange(bbox[0],bbox[1],(bbox[1]-bbox[0])/5),labels=[1,0,0,0])
    m.drawmeridians(np.arange(bbox[2],bbox[3],(bbox[3]-bbox[2])/5),labels=[0,0,0,1],rotation=45)

    m.scatter(longs, lats, latlon=True, c=ratings, s=count, cmap='Reds', alpha=0.3) 

    plt.colorbar(label=r'rating')
    plt.clim(1, 5)

    plt.title("Restaurant Distribution")
    
    plt.savefig('figures/restaurant_NV_' + str(counter) + '.png', format='png', dpi=500)
    plt.close()


def parse_csv(bizData):
    lats = bizData['latitude'].values
    longs = bizData['longitude'].values 
    ratings = bizData['stars'].values
    states = bizData['state'].values
    count = bizData['review_count'].values 
    return longs, lats, states, ratings, count

def plotOverTime_init(reviewSortedData):
    # this function doesn't quite work
    id_list = np.unique(reviewSortedData['business_id'].values)
    rating_list=[[0 for i in range(len(id_list))]for i in range(169)]
    review_count=[[0 for i in range(len(id_list))] for i in range(169)]

    # s = 0
    monthFlag = 1
    curr_month = 10
    curr_year = 2004
    curr_avg = 0
    curr_num_ratings = 0
    curr_id = 0
    curr_ind = -1
    count = 0

    for index, row in reviewSortedData.iterrows():
        # get data from row
        bz_id = row['business_id']
        rating = row['stars']
        date = row['date']
        date_list = date.split('-')
        
        # check if date_list is in form ['year', 'month', 'date time']
        if len(date_list) > 1:
            tmp_year = int(date_list[0])
            tmp_month = int(date_list[1])
        else:
            break
        
        # check if legitimate year
        if (tmp_year == 0 or tmp_month == 0) or (tmp_month > 13):
            break

        if bz_id != curr_id:   
            # find index of bz id
            for i in range(len(id_list)):
                if bz_id == id_list[i]:
                    curr_ind = i
                    break
            # bz id is not in our list (which shouldn't happen, but just in case)
            if curr_ind == -1:
                break
            else:
                # reset all values
                curr_id = bz_id
                monthFlag = 1
                curr_month = 10
                curr_year = 2004
                curr_avg = 0
                curr_num_ratings = 0
                curr_ind = -1
                count = 0

        # check business id
        if bz_id == curr_id:
            # no rating stored for this month yet, and we have review that is of that month/year
            if curr_month == tmp_month and curr_year == tmp_year and monthFlag:
                curr_num_ratings += 1
                curr_avg = float((curr_avg + rating)/curr_num_ratings)
                rating_list[count][curr_ind] = curr_avg
                review_count[count][curr_ind] = curr_num_ratings
                if curr_month == 12:
                    curr_month = 1
                    curr_year += 1
                else:
                    curr_month += 1
                monthFlag = 0
            # already have a rating for this month
            elif curr_month == tmp_month and curr_year == tmp_year and (not monthFlag):
                curr_num_ratings += 1
                continue
            # no reviews for some time Q_Q
            elif curr_month < tmp_month and curr_year <= tmp_year:
                diff_year = tmp_year - curr_year
                diff_month = tmp_month - curr_month

                rating_list[count][curr_ind] = rating_list[count][curr_ind]
                review_count[count][curr_ind] = curr_num_ratings
                if curr_month == 12:
                    curr_month = 1
                    curr_year += 1
                else:
                    curr_month += 1

def plotOverTime(reviewSortedData, bizData): 
    # this is the one you probably want
    id_list = np.unique(reviewSortedData['business_id'].values)
    curr_rating = [0 for i in range(len(id_list))]
    actual_rating = [0 for i in range(len(id_list))]
    num_ratings = [0 for i in range(len(id_list))]
    actual_num_ratings = [0 for i in range(len(id_list))]
    bz = []
    counter = 0
    month = 10
    year = 2004

    lats = []
    longs = []
    loc = -1

    for curr_id in id_list:
        for i, biz_id in enumerate(bizData['business_id'].values):
            if curr_id == biz_id:
                loc = i
                break
        lat = bizData['latitude'].values[loc]
        long = bizData['longitude'].values[loc]
        lats.append(lat)
        longs.append(long)

    for index, row in reviewSortedData.iterrows():
        # get data from row
        bz_id = row['business_id']
        rating = row['stars']
        date = row['date']
        if type(date) is not str:
            break
        date_list = date.split('-')
        
        # check if date_list is in form ['year', 'month', 'date time']
        if len(date_list) > 1:
            tmp_year = int(date_list[0])
            tmp_month = int(date_list[1])
        else:
            continue
        ind = -1
        # check if legitimate year
        if (tmp_year == 0 or tmp_month == 0) or (tmp_month > 13):
            continue

        if  tmp_month > month or tmp_year > year:
            month = tmp_month
            year = tmp_year
            curr_rating = np.asarray(curr_rating)
            num_ratings = np.asarray(num_ratings)
            print(curr_rating)
            print(num_ratings)
            map_plot(longs, lats, curr_rating, num_ratings, counter)
            curr_rating = actual_rating
            num_ratings = actual_num_ratings
            bz = []
            counter += 1

        # check if same month and year
        if tmp_month == month and tmp_year == year: 
            # find the correct index of business
            for i in range(len(id_list)):
                if bz_id == id_list[i]:
                    ind = i
                    break
            if ind == -1:
                continue
            # we already did this business 
            elif bz_id in bz:
                num_rating = actual_num_ratings[ind]
                actual_num_ratings[ind] += 1
                new_a = float((actual_rating[ind] * num_rating + rating)/(actual_num_ratings[ind]))
                actual_rating[ind] = new_a
                continue
            else:
                sum_ratings = actual_rating[ind] * actual_num_ratings[ind]
                num_ratings[ind] += 1
                actual_num_ratings[ind] += 1
                new_avg = float((sum_ratings + rating)/(num_ratings[ind]))
                curr_rating[ind] = new_avg
                actual_rating[ind] = new_avg
                bz.append(bz_id)


def main():
    bizfile = DIRECTORY + "filtered_business.csv"
    bizData = pd.read_csv(bizfile, encoding= "latin-1")

    sorted_reviews = DIRECTORY + "sorted_reviews.csv"
    reviewData = pd.read_csv(sorted_reviews, encoding= "latin-1")
    reviewSData = reviewData.sort_values(by = ['date'])

    # print(list(reviewSData))
    # 'Unnamed: 0', 'business_id', 'cool', 'date', 'funny', 'review_id', 'stars', 'text', 'useful', 'user_id'
    # print(reviewSData.iloc[[0, -2], 3])
    # earliest time:  2004-10-19 03:05:42
    # latest time: 2018-11-14 18:12:40
    # print(reviewSData.shape)

    plotOverTime(reviewSData, bizData)

    # longs, lats, states, ratings, count = parse_csv(bizData)
    # map_plot(longs, lats, ratings, count, 166)
    
if __name__ == "__main__":
    main()
