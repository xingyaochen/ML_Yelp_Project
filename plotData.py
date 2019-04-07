# Split data from Yelp dataset challenge into train and test sets
# import argparse
# import collections
# import csv
# import json
from constants import *
import pandas as pd
import matplotlib as matplotlib
import matplotlib.pyplot as plt
# # import datetime
# import numpy as np


biz = pd.read_csv(DIRECTORY+"filtered_business.csv", encoding= "latin-1")

y_labels = biz[['business_id','stars']]
y_labels = y_labels.values
stars = y_labels[:,1]
# stars = y_labels[1:5,]
histvals = [x[1] for x in y_labels]

print(histvals)
plt.hist(histvals)

plt.show()
# print(y_labels)

