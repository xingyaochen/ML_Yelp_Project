from rating_time import *
from constants import *
import pandas as pd
from sklearn.linear_model import Lasso
from explore import *
from sklearn import metrics 
from sklearn.model_selection import train_test_split
from crossval import *


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

def regressionCV(cv_filename, features, labels):
    cross_validation = pd.read_csv(DIRECTORY+cv_filename, encoding= "utf-8")
    numFolds = int(np.max(cross_validation['foldNum']))
    alphas = [0.001, 0.01, 0.1, 1, 10, 100]
    rmse_alpha_scores_train = np.empty((len(alphas), numFolds+1))
    rmse_alpha_scores_test= np.empty((len(alphas), numFolds+1))

    r2_alpha_scores_train = np.empty((len(alphas), numFolds+1))
    r2_alpha_scores_test = np.empty((len(alphas), numFolds+1))

    regList = [Lasso(alpha = a, random_state=0) for a in alphas]
    for i in range(numFolds+1):
        print("Running CV fold", i)
        test_CV = cross_validation.loc[cross_validation['foldNum']==i]
        train_CV = cross_validation.loc[cross_validation['foldNum']!=i]

        test_CV_X = test_CV[features]
        test_CV_y = test_CV[labels]

        train_CV_X = train_CV[features]
        train_CV_y = train_CV[labels]
        rmse_train = []
        rmse_test = []
        r2_train, r2_test = [], []

        for j, a in enumerate(alphas):
            reg_m = regList[j]
            reg_m.fit(train_CV_X, train_CV_y)

            r2_train.append(reg_m.score(train_CV_X, train_CV_y))
            r2_test.append(reg_m.score(test_CV_X, test_CV_y))

            rmse_train.append(metrics.mean_squared_error(train_CV_y, reg_m.predict(train_CV_X)))
            rmse_test.append(metrics.mean_squared_error(test_CV_y, reg_m.predict(test_CV_X)))
        
        rmse_alpha_scores_train[:,i] = rmse_train
        rmse_alpha_scores_test[:,i] = rmse_test 

        r2_alpha_scores_train[:,i] = r2_train
        r2_alpha_scores_test[:,i] = r2_test


    rmse_list_test = np.mean(rmse_alpha_scores_test, axis = 1)
    rmse_list_train = np.mean(rmse_alpha_scores_train, axis = 1)

    r2_list_test = np.mean(r2_alpha_scores_test, axis = 1)
    r2_list_train = np.mean(r2_alpha_scores_train, axis = 1)

    return regList, rmse_list_test, rmse_list_train, r2_list_test, r2_list_train



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