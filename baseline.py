from rating_time import *
from constants import *
import pandas as pd
from sklearn.linear_model import LassoCV
from explore import *
from sklearn import metrics 
from sklearn.model_selection import train_test_split



# use only if y_train is average_over_span
# train_data = train_data.dropna(how = 'any')
# test_data = test_data.dropna(how = 'any')


# reg = LassoCV(cv=8, random_state=0).fit(X_train, y_train)
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