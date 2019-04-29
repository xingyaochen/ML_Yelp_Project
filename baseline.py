from rating_time import *
from constants import *
import pandas as pd
from sklearn.linear_model import LassoCV
from explore import *
from sklearn import metrics 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from crossval import *
from lasso_regression import *

# def abline(slope, intercept, axes = None):
#     """Plot a line from slope and intercept"""
#     if not axes:
#         axes = plt.gca()
#     x_vals = np.array(axes.get_xlim())
#     y_vals = intercept + slope * x_vals
#     plt.plot(x_vals, y_vals, 'r-', color="red")


# use only if y_train is average_over_span
# train_data = train_data.dropna(how = 'any')
# test_data = test_data.dropna(how = 'any')


train_data = pd.read_csv(DIRECTORY+"training.csv", encoding= "utf-8")

features = ['RestaurantsAttire_casual', 'RestaurantsAttire_dressy', \
     'BusinessAcceptsCreditCards_True', 'RestaurantsDelivery_True', 'Alcohol_beer_and_wine',\
     'Alcohol_full_bar', 'Alcohol_none', 'Caters_True', 'WiFi_free', 'WiFi_no', 'WiFi_paid', \
     'BikeParking_True', 'NoiseLevel_average', 'NoiseLevel_loud', 'NoiseLevel_quiet', \
     'NoiseLevel_very_loud', 'HasTV_True', 'OutdoorSeating_True', 'RestaurantsTakeOut_True', \
     'RestaurantsReservations_True', 'GoodForKids_True', 'RestaurantsPriceRange2_1', \
     'RestaurantsPriceRange2_2', 'RestaurantsPriceRange2_3', 'RestaurantsPriceRange2_4', \
     'RestaurantsGoodForGroups_True', 'Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', \
     'Tuesday', 'Wednesday']
train_data.dropna(inplace = True)
labels = ['running_average']


train_data_X = train_data[features]
train_data_y = train_data[labels]

alphas = [10**a for a in range(-5, 2)]
n_splits = 5
ts_CV = TimeSeriesSplit(n_splits=n_splits)

regList, rmse_list_test, rmse_list_train, r2_list_test, r2_list_train = regressionCV_new(train_data, features, labels, alphas, n_splits) 


best_regMod = regList[np.argmin(rmse_list_test)]
min_rmse = np.amin(rmse_list_test)


features +=['running_average_past_bin']

regList_pr, rmse_list_test_pr, rmse_list_train_pr, r2_list_test_pr, r2_list_train_pr = regressionCV_new(train_data, features, labels, alphas, n_splits) 


best_regMod_pr = regList_pr[np.argmin(rmse_list_test_pr)]
min_rmse_pr = np.amin(rmse_list_test_pr)



errors = [rmse_list_test, rmse_list_train, rmse_list_test_pr, rmse_list_train_pr]
labs = ['rmse test baseline', 'rmse train baseline', 'rmse test baselinePR', 'rmse train baselinePR']
fig, ax = plt.subplots()
for i, y in enumerate(errors):
     ax.plot(np.log10(alphas), y, label=labs[i], linewidth = 1)
plt.legend(loc='best', fontsize = 11)
plt.xlabel("logged alpha")
plt.ylabel("RMSE")
plt.title("Cross Validation Results | RMSE")
plt.savefig('figs/cv_baselines_rmse.png')
plt.close()



r2s = [r2_list_test, r2_list_train, r2_list_test_pr, r2_list_train_pr]
labs = ['r2 test baseline', 'r2 train baseline', 'r2 test baselinePR', 'r2 train baselinePR']

fig, ax = plt.subplots()
for i, y in enumerate(r2s):
     ax.plot(np.log10(alphas), y, label=labs[i], linewidth = 1)
plt.legend(loc='best', fontsize = 11)
plt.xlabel("logged alpha")
plt.ylabel("R^2")
plt.title("Cross Validation Results | R^2")
plt.savefig('figs/cv_baselines_r2.png')
plt.close()



# results 
# rmse_list_test, rmse_list_train, r2_list_test, r2_list_train
# (array([0.20950877, 0.22347733, 0.24765749, 0.26320678, 0.26320678,
#        0.26320678]), array([0.21765042, 0.23071828, 0.25561351, 0.2699682 , 0.2699682 ,
#        0.2699682 ]), array([ 0.20382766,  0.15074805,  0.05885594, -0.00023058, -0.00023058,
#        -0.00023058]), array([0.19387796, 0.14545508, 0.05325227, 0.        , 0.        ,
#        0.        ]))