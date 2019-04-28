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
from sklearn.feature_extraction.text import TfidfVectorizer
from crossval import *

def performance(y_true, y_pred, metric="accuracy") :
    """
    Calculates the performance metric based on the agreement between the 
    true labels and the predicted labels.
    
    Parameters
    --------------------
        y_true -- numpy array of shape (n,), known labels
        y_pred -- numpy array of shape (n,), (continuous-valued) predictions
        metric -- string, option used to select the performance measure
                  options: 'accuracy', 'f1_score', 'auroc', 'precision',
                           'sensitivity', 'specificity'        
    
    Returns
    --------------------
        score  -- float, performance score
    """
    # map continuous-valued predictions to binary labels
    y_label = np.sign(y_pred)
    y_label[y_label==0] = 1 # map points of hyperplane to +1
    
    ### ========== TODO : START ========== ###
    # part 2a: compute classifier performance
    if metric == 'accuracy':
        return metrics.accuracy_score(y_true, y_label)
    if metric == 'f1_score':
        return metrics.f1_score(y_true, y_label)
    if metric == 'auroc':
        return metrics.roc_auc_score(y_true, y_pred, average='micro')
    if metric == 'precision':
        return metrics.precision_score(y_true, y_label)
    if metric == 'recall' or metric == 'sensitivity':
        return metrics.recall_score(y_true, y_label)
    if metric == 'specificity':
        confusionMatrix = metrics.confusion_matrix(y_true, y_label, labels = [1, -1])
        falsePos = confusionMatrix[1][0]
        falseNeg = confusionMatrix[0][1]
        truePos = confusionMatrix[0][0]  
        trueNeg = confusionMatrix[1][1]
        if (trueNeg+falsePos) > 0:
            return float(trueNeg)/(trueNeg+falsePos)
        return 0

def addSentimentToFeature(business_df,business_sentiment):
    """
    params
    -----------
    business_df:               a pandas df shape (n,x) where x is the 
                                current number of features after one-hot encoding
    business_sentiment:        a pandas df column, shape (n,2), first column business id
                                second column sentiment polarity within [0,1]

    return
    -----------
    full_feature:                a pandas df with shape (n,x+1) 

    """
    return pd.merge(business_df, business_sentiment, on='business_ids', how='outer')

def calculateAveragePolarity(business_ids,predicted_polarity):
    """
    calculate the average polarity (0,1) 
    params
    -----------
    business_ids:               a pandas df colum, shape (n,)
    predicted_polarity:         a pandas df column, shape (n,)

    the shape of business_ids and predicted_polarity should be the
    same dimension so each entry correspond to one review, and there might be many
    rows with the same business_ids

    return
    -----------
    business_sentiment:         a pandas df with shape (x,2) 
                                    where x is the number of unique businesses

    """
    busi_idz=business_ids.values
    polarity=predicted_polarity.values
    data = pd.DataFrame({'business_ids':busi_idz,'avg_sentiment':polarity})
    business_sentiment=data.groupby(["business_id"]).mean()

    return business_sentiment


def parseReviewDF(df,cumulative_rating=True):
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
        review_text=df[['business_id','review_id','text','running_average']]
        review_text.rename(columns={'running_average':'rating'}, inplace=True)
    else:
        review_text=df[['business_id','review_id','text','average_over_span']]
        review_text.rename(columns={'average_over_span':'rating'}, inplace=True)
    return review_text

def preprocess(review_text_,n_gram=1,binary_rating=True,Tfidf=False):
    """
    tokenize each review text into a list of words also add a col for length of the tokenized text.
    Set binary_rating to true if we want to deal with 0(bad) and 1(good) as labels instead of 1-5 scaled ratings
    """
    #calculate the average rating as a dividing line between bad and good rating
    #which we can use as a proxy for positive/negative label
    review_text=review_text_.copy(deep=True)
    averageRating=review_text["rating"].mean()

    #helper function for converting
    def changeToBinaryRating(rating):
        if rating>=averageRating:
            return 1
        else: return 0
    
    for index_label, row_series in review_text.iterrows():
   # For each row update the 'Bonus' value to it's double
        if Tfidf==False:
            #only change text to BOG if Tfidf is false
            tokenizeWords=word_tokenize(row_series['text'])
            tokenizeWords=preprocess_Review(tokenizeWords)
            if n_gram==1: pass
            elif n_gram==2:
                tokenizeWords=bigram(tokenizeWords)
            elif n_gram==3:
                tokenizeWords=trigram(tokenizeWords)
            else:
                raise Exception("ONLY support n_gram for n= 1, 2, 3!!!")
            
            review_text.at[index_label , 'text'] = tokenizeWords

        if binary_rating:
            review_text.at[index_label , 'rating']=changeToBinaryRating(row_series['rating'])
   
  
    # review_text["sents_length"]=sents_length.reset_index(level=0, drop=True)

    return review_text

def preprocess_Review(tokenized_sent):
    stop_words=set(stopwords.words("english"))
    ps = PorterStemmer()
    filtered_sent=[]
    for w in tokenized_sent:
        w=w.lower()
        if w not in stop_words:
            word=ps.stem(w)
            filtered_sent.append(word)
    return filtered_sent

def bigram(tokenized_sent):
    return list(nltk.bigrams(tokenized_sent))

def trigram(tokenized_sent):
    return list(nltk.trigrams(tokenized_sent))


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
                            n is the number of reviews
                            d is the number of unique words in the text file
    """
 
    num_lines = tokenizeWords.count()
    num_words = len(word_list)
    feature_matrix = np.zeros(num_words)
 
    for i, words in enumerate(tokenizeWords.tolist()):
        for word in words:
            #find the index in the dictionary
            index=word_list[word]
            #set the value in the matrix equal to 1
            feature_matrix[i][index]=1


    
    return feature_matrix

def train_model(X_train, y_train, trainer="MultinomialNB"):
    """
    trains the classifier
    """
    if trainer=="MultinomialNB":
        clf = MultinomialNB()
        clf.fit(X_train, y_train) 

    elif trainer=="LinearSVC":
        clf=LinearSVC(dual=False)
        clf.fit(X_train,y_train)
    else:
        raise Exception("WHYYYY???? You only have two options:'MultinomialNB' and 'LinearSVC' ")

    return clf





def analysis(review_text,Tfidf=False):
    """
    currently only dealing with polarity as label
    """
    
    
    if Tfidf:
        tf=TfidfVectorizer()
        X= tf.fit_transform(review_text["text"])
    else:
        X=review_text["text"]
        #USING BAD OF WORDS
        dictionary = extract_dictionary(X)
        X = extract_feature_vectors(X, dictionary)


    y=review_text["rating"]
    #turning label into numpy array and the X into on-hot numpy array
    y=y.values
    
    #train test split
  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


    clf = train_model(X_train, y_train, trainer="MultinomialNB")
    predicted= clf.predict(X_test)
    print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))
    print("MultinomialNB f1:",metrics.f1_score(y_test,predicted))

  
    linear=train_model(X_train, y_train, trainer="LinearSVC")
    y_label=linear.predict(X_test)
    
    acc=metrics.accuracy_score(y_test,y_label,normalize=True)
    f1=metrics.f1_score(y_test,y_label)
    print("linear SVC Accuracy: ",acc)
    print("linear SVC f1: ",f1)
    
 

def cross_validation(clf,metrics=["accuracy","f1_score"],Tfidf=False) :
    """
    Splits the data, X and y, into k-folds and runs k-fold cross-validation.
    Trains classifier on k-1 folds and tests on the remaining fold.
    Calculates the k-fold cross-validation performance metric for classifier
    by averaging the performance across folds.
    
    Parameters
    --------------------
        clf     -- classifier (instance of SVC)
        X       -- numpy array of shape (n,d), feature vectors
                     n = number of examples
                     d = number of features
        y       -- numpy array of shape (n,), binary labels {1,-1}
        kf      -- model_selection.KFold or model_selection.StratifiedKFold
        metrics -- list of m strings, metrics
    
    Returns
    --------------------
        scores  -- numpy array of shape (m,), average CV performance for each metric
    """
    #comment out later
    reviewfile=DIRECTORY + "review_ratingOverTime.csv"
    reviewdf = pd.read_csv(reviewfile, encoding = "latin-1")

    #do preprocessing here !!!!
    reviewdf=parseReviewDF(reviewdf)
    #using BoW
 
    if Tfidf==False:
        reviewdf=preprocess(reviewdf,Tfidf=False)
        texts=reviewdf["text"]
        #USING BAD OF WORDS
        dictionary = extract_dictionary(texts)
        bow=extract_feature_vectors(texts,dictionary)
        reviewdf.update({"text":bow})
    if Tfidf==True:
        reviewdf=preprocess(reviewdf,Tfidf=True)
        tf=TfidfVectorizer()
        X= tf.fit_transform(reviewdf["text"])
        reviewdf.update({"text":X})
    print(reviewdf["rating"].value_counts())


    #make a nested dictionary of a reviews, first key is column you want and second is review id
    reviewdict = reviewdf.set_index('review_id').to_dict()
    return cv_performance(clf,reviewdict,metrics)
    
    
    #----------------------------------------------------------
    #CROSS VALIDATION BELOW
def cv_performance(clf,reviewdict,metrics=["accuracy","f1_score"]) :

    crossValFile = DIRECTORY + "crossVal.csv"
    #get crossvalidation data
    crossValdf = pd.read_csv(crossValFile, encoding = "latin-1")
    m = len(metrics)
    k=len(np.unique(crossValdf['foldNum']))
    scores = np.empty((m, k))
    #For each fold
    for currfold in np.unique(crossValdf['foldNum']):
        #get its training set and validation set
        train_data = []
        validation_data = []

        #get the segments of data
        train_cv, validate_cv = get_cv_fold(crossValdf, currfold)


        #match review_ids and put them in correct dataframe
        for index, row in train_cv.iterrows():
            curr_id = row['review_id']
            train_data.append([reviewdict['text'][curr_id],reviewdict['rating'][curr_id]])
        for index, row in validate_cv.iterrows():
            curr_id = row['review_id']
            validation_data.append([reviewdict['text'][curr_id],reviewdict['rating'][curr_id]])
        traindf = pd.DataFrame(train_data, columns = ['text', 'rating'])
        validatedf = pd.DataFrame(validation_data, columns = ['text', 'rating'])
        # traindf.to_csv(DIRECTORY+"traindf"+str(currfold)+".csv", encoding="latin-1", index=False)
        # validatedf.to_csv(DIRECTORY+"validatedf"+str(currfold)+".csv", encoding="latin-1", index=False)
        y_train=traindf["rating"].values
        X_train=traindf["text"].values
        y_val=validatedf["rating"].values
        X_val=validatedf["text"].values

        clf = model.fit(X_train,y_train)
        y_pred= clf.predict(X_val)
        for m, metric in enumerate(metrics):
            score = performance(y_val, y_pred, metric)
            scores[m,k] = score

    return scores.mean(axis=1)
        


 def lineplot(x, y, label):
    """
    Make a line plot.
    
    Parameters
    --------------------
        x            -- list of doubles, x values
        y            -- list of doubles, y values
        label        -- string, label for legend
    """
    
    xx = range(len(x))
    plt.plot(xx, y, linestyle='-', linewidth=2, label=label)
    plt.xticks(xx, x)

def select_param_linear(X, y,metrics=["accuracy","f1_score"], plot=True,Tfidf=False) :
    """
    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting and metric,
    then selects the hyperparameter that maximizes the average performance for each metric.
    
    Parameters
    --------------------
        X       -- numpy array of shape (n,d), feature vectors
                     n = number of examples
                     d = number of features
        y       -- numpy array of shape (n,), binary labels {1,-1}
        kf      -- model_selection.KFold or model_selection.StratifiedKFold
        metrics -- list of m strings, metrics
        plot    -- boolean, make a plot
    
    Returns
    --------------------
        params  -- list of m floats, optimal hyperparameter C for each metric
    """
    
    C_range = 10.0 ** np.arange(-3, 3)
    scores = np.empty((len(metrics), len(C_range)))
    
    ### ========== TODO : START ========== ###
    # part 3b: for each metric, select optimal hyperparameter using cross-validation
    for j, c in enumerate(C_range):
        model_svc = LinearSVC(dual=False,C=c)
        # compute CV scores using cv_performance(...)
        scores[:,j] = cross_validation(model_svc,metrics,Tfidf=False)

    # get best hyperparameters
    best_params_ind = np.argmax(scores,  axis=1)    # dummy, okay to change
    best_params = C_range[best_params_ind]
    ### ========== TODO : END ========== ###
    
    # plot
    if plot:
        plt.figure()
        ax = plt.gca()
        ax.set_ylim(0, 1)
        ax.set_xlabel("C")
        ax.set_ylabel("score")
        for m, metric in enumerate(metrics) :
            lineplot(C_range, scores[m,:], metric)
        plt.legend()
        plt.savefig("linear_param_select.png")
        plt.close()
    
    return best_params
 


def main():
    # crossValFile = DIRECTORY + "crossVal.csv"
    # #get crossvalidation data
    # crossValdf = pd.read_csv(crossValFile, encoding = "latin-1")
    cv_performance()
    print('done!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
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
    bow_review_text=preprocess(review_text)
    print("Train and predict(BOW)...")
    analysis(bow_review_text)

    # print("Preprocess review text and convert to polarity...")
    # tfidf_review_text=preprocess(review_text,Tfidf=True)
    # print("Train and predict(TF-IDF)...")
    # analysis(tfidf_review_text,Tfidf=True)

if __name__ == "__main__":
    main()