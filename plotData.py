#Plot data from Yelp dataset
from constants import *
import pandas as pd
import matplotlib as matplotlib
import matplotlib.pyplot as plt
import numpy as np

#filter by stars
def plot_ratings(bizData):
    """Plots histogram of business ratings"""
    #get ratings
    ratings = bizData[['stars']]
    ratings = ratings.values

    #group and plot
    cats, heights = np.unique(ratings,return_counts=True)
    plt.bar(cats, heights, width = 0.5)
    plt.xlabel("Restaurant Rating")
    plt.ylabel("Number of Occurrences")
    plt.show()

def plot_locations(bizData):
    """Plots barchart of business by state"""
    locations = bizData[['state']]

    #group and plot
    cats, heights = np.unique(locations,return_counts=True)
    plt.bar(cats, heights, width = 0.5)
    plt.xlabel("Restaurant Location")
    plt.ylabel("Number of Occurrences")
    plt.show()



def main():
    bizfile = DIRECTORY + "filtered_business.csv"
    bizData = pd.read_csv(bizfile, encoding= "latin-1")
    plot_ratings(bizData)
    plot_locations(bizData)

if __name__ == "__main__":
    main()
