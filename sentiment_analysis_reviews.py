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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from crossval import get_cv_fold

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


def parseReviewDF(df):
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
 
    review_text=df[['business_id','review_id','text_past','running_average_past']]
    review_text.rename(columns={'running_average_past':'rating','text_past':"text"}, inplace=True)
   
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
    print('startig for loop')
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
    print('done with for loop')
  
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





def analysis(X_train, X_test, y_train, y_test,Tfidf=False):
    """
    currently only dealing with polarity as label
    """

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
    
 

def cross_validation_SVM(X_train,X_test,y_train,y_test,fold=5) :

    
    #---------------------------------------------------------------
    #grid search SVM
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},]
    scores = ['accuracy', 'f1']
    output=[]
    for score in scores:

        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(SVC(), tuned_parameters, cv=fold,
                        scoring='%s_macro' % score)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
      
        print("Best score:")
        print()
        print(clf.best_score_ )

        output.append((clf.best_score_,clf.best_params_))
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()
    
    return output
    

    
def cross_validation_MultinomialNB(X_train,X_test,y_train,y_test,fold=5):
    tuned_parameters={'clf__alpha': [1, 1e-1, 1e-2]}
    scores = ['accuracy', 'f1']
    output=[]
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(MultinomialNB(), tuned_parameters, cv=fold,
                        scoring='%s_macro' % score)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
      
        print("Best score:")
        print()
        print(clf.best_score_ )

        output.append((clf.best_score_,clf.best_params_))
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()

    return output


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



def main():
    reviewfile=DIRECTORY + "training.csv"
    reviewdf = pd.read_csv(reviewfile, encoding = "latin-1")
    reviewdf=parseReviewDF(reviewdf)
    #for SVM:
    SVM_scores=np.zeros((2,3,2))
    multinomial_scores=np.zeros((2,3,2))

    for i,tfidf in enumerate([True, False]):
        for j,n in enumerate([1,2,3]):
    
            if tfidf==False:
                reviewdf=preprocess(reviewdf,n_gram=n, Tfidf=False)
                texts=reviewdf["text"]
            
                #USING BAD OF WORDS
                dictionary = extract_dictionary(texts)
            
                X=extract_feature_vectors(texts,dictionary)
            
            if tfidf==True:
                reviewdf=preprocess(reviewdf,n_gram=n,Tfidf=True)
                tf=TfidfVectorizer()
                X= tf.fit_transform(reviewdf["text"])
            
            y=reviewdf["rating"].values
            
            print(reviewdf["rating"].value_counts())

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


            output=cross_validation_SVM( X_train, X_test, y_train, y_test ,fold=5) 
            SVM_scores[i,j,:]=output

            output=cross_validation_MultinomialNB( X_train, X_test, y_train, y_test ,fold=5)
            multinomial_scores[i,j,:]=output


    print(SVM_scores)
    print(multinomial_scores)

    return 


    
    


   
  
    

if __name__ == "__main__":
    main()