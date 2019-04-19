import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.tokenize import *
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.probability import FreqDist
import pandas as pd
import numpy as np
from constants import *
import csv
import rating_time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC

def parseReviewDF(df,cumulative_rating=False):
    """
    parse the review pandas df and return a new pandas df with only the text and review_Id

    param
    -----------
    df: the pandas dataframe of the review csv file
    cumulative_rating: whether we are using cumulative or average rating over some fixed time span
    """
    # dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    # #read in the review csv
    # review = pd.read_csv(reviewfile, parse_dates = ['date'], date_parser = dateparse, encoding = "latin-1")
    if cumulative_rating:
        review_text=df[['review_id','text','running_average']]
        review_text.rename(columns={'running_average':'rating'}, inplace=True)
    else:
        review_text=df[['review_id','text','average_over_span']]
        review_text.rename(columns={'average_over_span':'rating'}, inplace=True)
    return review_text

def preprocess(review_text,binary_rating=True):
    """
    tokenize each review text into a list of words also add a col for length of the tokenized text.
    Set binary_rating to true if we want to deal with 0(bad) and 1(good) as labels instead of 1-5 scaled ratings
    """
    #calculate the average rating as a dividing line between bad and good rating
    #which we can use as a proxy for positive/negative label
    averageRating=review_text["rating"].mean()

    #helper function for converting
    def changeToBinaryRating(rating):
        if rating>=averageRating:
            return 1
        else: return 0

    for index_label, row_series in review_text.iterrows():
   # For each row update the 'Bonus' value to it's double
        tokenizeWords=word_tokenize(row_series['text'])
        tokenizeWords=removeStopWords(tokenizeWords)
        tokenizeWords=convertToStem(tokenizeWords)
        review_text.at[index_label , 'text'] = tokenizeWords

        if binary_rating:
            review_text.at[index_label , 'rating']=changeToBinaryRating(row_series['rating'])
   
  
    # review_text["sents_length"]=sents_length.reset_index(level=0, drop=True)

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

def extract_dictionary(allwords) :
    """
    para: allwords is a df column
    Returns
    --------------------
        word_list -- dictionary, (key, value) pairs are (word, index)
    """
    
    word_list = {}
   #TODO: check if the words have been converted to lower case
    current_index=0
    for words in allwords.tolist():
        for word in words:
            if word not in word_list:
                word_list[word]=current_index
                current_index+=1

    return word_list

def extract_feature_vectors(tokenizeWords, word_list) :
    """
    Produces a bag-of-words representation of a the tokenized words column of the df 
    based on the dictionary word_list.
    
    Parameters
    --------------------
        tokenizeWords  -- pandas df column,each row is a list of tokenized, preprocessed words
        word_list      -- dictionary, (key, value) pairs are (word, index)
    
    Returns
    --------------------
        feature_matrix -- numpy array of shape (n,d)
                          boolean (0,1) array indicating word presence in a string
                            n is the number of non-blank lines in the text file
                            d is the number of unique words in the text file
    """
 
    num_lines = tokenizeWords.count()
    num_words = len(word_list)
    feature_matrix = np.zeros((num_lines, num_words))
 
    for i, words in enumerate(tokenizeWords.tolist()):
        #loop through each word in the tweet
        for word in words:
            #find the index in the dictionary
            index=word_list[word]
            #set the value in the matrix equal to 1
            feature_matrix[i][index]=1


    
    return feature_matrix

def analysis(review_text):
    """
    currently only dealing with polarity as label
    """
    X=review_text["text"]
    y=review_text["rating"]
    #turning label into numpy array and the X into on-hot numpy array
    y=y.values
    #USING BAD OF WORDS
    dictionary = extract_dictionary(X)
    X = extract_feature_vectors(X, dictionary)

    #train test split
  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


    clf = MultinomialNB().fit(X_train, y_train)
    predicted= clf.predict(X_test)
    print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))
    print("MultinomialNB f1:",metrics.f1_score(y_test,predicted))

    linear=SVC(C=1,kernel='linear')
    linear.fit(X_train,y_train)
    pred_linear=linear.decision_function(X_test)
    y_label = np.sign(pred_linear)
    y_label[y_label==0] = 1 # map points of hyperplane to +1
    acc=metrics.accuracy_score(y_test,y_label,normalize=True)
    f1=metrics.f1_score(y_test,y_label)
    print("linear SVC Accuracy: ",acc)
    print("linear SVC f1: ",f1)
    
    # rbf=SVC(C=32,gamma=0.0078125,kernel='rbf')
    # rbf.fit(X_train,y_train)
    # pred_rbf=rbf.decision_function(X_train)
    # majority=DummyClassifier(strategy="most_frequent")
    # majority.fit(X_train,y_train)
    # pred_majority=majority.predict(X_train)




    # #plot the distribution in the dataset
    # Sentiment_count=review_text.groupby('rating').count()
    # plt.bar(Sentiment_count.index.values, Sentiment_count['text'])
    # plt.xlabel('Review Sentiments')
    # plt.ylabel('Number of Review')
    # plt.show()

    # training_set=
    # sentim_analyzer = SentimentAnalyzer()
    # if trainer_=="NaiveBayes":
    #     trainer = NaiveBayesClassifier.train
    
    # classifier = sentim_analyzer.train(trainer, training_set)


def main():
    #load in pandas df or csv
    reviewfile=DIRECTORY + "review_ratingOverTime.csv"
    # dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    #read in the review csv
    print("Reading in csv...")
    review = pd.read_csv(reviewfile, encoding = "latin-1")
    #temporarily slice only a part to test code
    sample_review=review[:10000]
    print("Parsing dataframe...")
    review_text=parseReviewDF(sample_review,cumulative_rating=False)
    print("Preprocess review text and convert to polarity...")
    review_text=preprocess(review_text)
    print("Train and predict...")
    analysis(review_text)

if __name__ == "__main__":
    main()