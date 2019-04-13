from rating_time import *
from constants import *
import pandas as pd
from sklearn.linear_model import LassoCV
from explore import *

biz = pd.read_csv(DIRECTORY+"filtered_business.csv", encoding= "utf-8")

biz_id = biz['business_id']
features_df_clean = construct_meta_features(biz)

featues_ohe = ohe_features(features_df_clean)
# featues_ohe['postal_code'] = biz['postal_code']
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

# use only if y_train is average_over_span
# train_data = train_data.dropna(how = 'any')
# test_data = test_data.dropna(how = 'any')

X_train = train_data[list(linked_featues_ohe)[1:]]
y_train = train_data['running_average']

X_test = test_data[list(linked_featues_ohe)[1:]]
y_test = test_data['running_average']

reg = LassoCV(cv=8, random_state=0).fit(X_train, y_train)
reg.score(X_train, y_train)


def abline(slope, intercept, axes = None):
    """Plot a line from slope and intercept"""
    if not axes:
        axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, 'r-', color="red")


slope = np.max(reg.coef_)
feature_max = list(X_train)[np.argmax(reg.coef_)]

x = X_train[feature_max]
plt.scatter(x, y_train, s = 5)
abline(slope, reg.intercept_)
plt.xlabel(feature_max, fontsize=14)
plt.ylabel("Running Rating Average", fontsize=14)
plt.show()