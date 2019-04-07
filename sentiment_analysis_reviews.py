from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.probability import FreqDist
import pandas as pd
from constants import *
import csv

def parseReviewDF(df):
    """
    parse the review pandas df and return a new pandas df with only the text and review_Id
    """
    # dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    # #read in the review csv
    # review = pd.read_csv(reviewfile, parse_dates = ['date'], date_parser = dateparse, encoding = "latin-1")
    review_text=df[['review_id','text']]
    return review_text

def tokenizeWords(review_text):
    """
    tokenize each review text into a list of words also add a col for length of the tokenized text
    """
    #tokenize and removes stop words
    tokenizeWords= review_text["text"].apply(word_tokenize)
    #remove stop words
    tokenizeWords= tokenizeWords.apply(removeStopWords)
    #convert to stem of the word (linguistic)
    tokenizeWords= tokenizeWords.apply(convertToStem)
    
    review_text["tokenizeWords"]=tokenizeWords.reset_index(level=0, drop=True)
    sents_length = review_text.apply(lambda row: len(row['tokenizeWords']), axis=1)
    review_text["sents_length"]=sents_length.reset_index(level=0, drop=True)

    return review_text

def removeStopWords(tokenized_sent):
    stop_words=set(stopwords.words("english"))
    filtered_sent=[]
    for w in tokenized_sent:
        if w not in stop_words:
            filtered_sent.append(w)
    return filtered_sent

def convertToStem(tokenized_sent):
    ps = PorterStemmer()
    for word in tokenized_sent:
        word=ps.stem(word)
    return tokenized_sent

# def frequencyDistribution(review_text):
#     """
#     Input: pandas df with tokenized text in the "tokenizeWords" column
#     Plots a frequency distribution and returns the frequency distribution
#     """

