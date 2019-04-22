from rating_time import *
from constants import *
import pandas as pd
from sklearn.linear_model import Lasso
from explore import *
from sklearn import metrics 
from sklearn.model_selection import train_test_split
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


# train_data = pd.read_csv(DIRECTORY+"training.csv", encoding= "utf-8")

features = ['RestaurantsGoodForGroups_True', 'HasTV_True', 'WheelchairAccessible_True', 'RestaurantsAttire_casual', 'RestaurantsAttire_dressy',\
 'RestaurantsAttire_formal', 'OutdoorSeating_True', 'NoiseLevel_average', 'NoiseLevel_loud', 'NoiseLevel_quiet', 'NoiseLevel_very_loud', \
     'RestaurantsTableService_True', 'BusinessAcceptsCreditCards_True', 'WiFi_free', 'WiFi_no', 'WiFi_paid', 'RestaurantsReservations_True',\
          'Caters_True', 'Alcohol_beer_and_wine', 'Alcohol_full_bar', 'Alcohol_none', 'RestaurantsDelivery_True', 'RestaurantsTakeOut_True', 'GoodForKids_True',\
               'RestaurantsPriceRange2_1', 'RestaurantsPriceRange2_2', 'RestaurantsPriceRange2_3', 'RestaurantsPriceRange2_4', 'BikeParking_True', 'Friday',\
                    'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday']

labels = ['running_average']

cv_filename = "cross_validation.csv"
regList, rmse_list_test, rmse_list_train, r2_list_test, r2_list_train = regressionCV(cv_filename, features, labels)


# .fit(X_train, y_train)
# reg.score(X_train, y_train)

# metrics.mean_squared_error(y_train, reg.predict(X_train))
# metrics.mean_squared_error(y_test, reg.predict(X_test))

# slope = np.min(reg.coef_)
# feature_max = list(X_train)[np.argmin(reg.coef_)]

# x = X_train[feature_max]
# plt.scatter(x, y_train, s = 5)
# abline(slope, reg.intercept_)
# plt.xlabel(feature_max, fontsize=14)
# plt.ylabel("Running Rating Average", fontsize=14)
# plt.show()