"""
Names:      : Xingyao Chen and Emily Zhao
Author      : Yi-Chieh Wu
Class       : HMC CS 158
Date        : 2019 Feb 22
Description : Twitter
"""

from string import punctuation
from clf_baseline import BaselineCLF
# numpy libraries
import numpy as np

# matplotlib libraries
import matplotlib.pyplot as plt

# scikit-learn libraries
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.metrics.scorer import make_scorer

######################################################################
# functions -- input/output
######################################################################

def read_vector_file(fname) :
    """
    Reads and returns a vector from a file.
    
    Parameters
    --------------------
        fname  -- string, filename
    
    Returns
    --------------------
        labels -- numpy array of shape (n,)
                    n is the number of non-blank lines in the text file
    """
    return np.genfromtxt(fname)


def write_label_answer(vec, outfile) :
    """
    Writes your label vector to the given file.
    
    Parameters
    --------------------
        vec     -- numpy array of shape (n,) or (n,1), predicted scores
        outfile -- string, output filename
    """
    
    # for this project, you should predict 70 labels
    if(vec.shape[0] != 70):
        print("Error - output vector should have 70 rows.")
        print("Aborting write.")
        return
    np.savetxt(outfile, vec)    


######################################################################
# functions -- feature extraction
######################################################################

def extract_words(input_string) :
    """
    Processes the input_string, separating it into "words" based on the presence
    of spaces, and separating punctuation marks into their own words.
    
    Parameters
    --------------------
        input_string -- string of characters
    
    Returns
    --------------------
        words        -- list of lowercase "words"
    """
    
    for c in punctuation :
        input_string = input_string.replace(c, ' ' + c + ' ')
    return input_string.lower().split()


def extract_dictionary(infile) :
    """
    Given a filename, reads the text file and builds a dictionary of unique
    words/punctuations.
    
    Parameters
    --------------------
        infile    -- string, filename
    
    Returns
    --------------------
        word_list -- dictionary, (key, value) pairs are (word, index)
    """
    
    word_list = {}
    with open(infile, 'rU') as fid :
        ### ========== TODO : START ========== ###
        # part 1a: process each line to populate word_list
        c = 0
        lines = fid.readlines()
        for line in lines:
            wordds = extract_words(line) 
            for w in wordds:
                if w not in word_list:
                    word_list[w] = c
                    c += 1
        ### ========== TODO : END ========== ###
    return word_list


def extract_feature_vectors(infile, word_list) :
    """
    Produces a bag-of-words representation of a text file specified by the
    filename infile based on the dictionary word_list.
    
    Parameters
    --------------------
        infile         -- string, filename
        word_list      -- dictionary, (key, value) pairs are (word, index)
    
    Returns
    --------------------
        feature_matrix -- numpy array of shape (n,d)
                          boolean (0,1) array indicating word presence in a string
                            n is the number of non-blank lines in the text file
                            d is the number of unique words in the text file
    """
    num_lines = sum(1 for line in open(infile,'rU'))
    num_words = len(word_list)
    feature_matrix = np.zeros((num_lines, num_words))
    
    with open(infile, 'rU') as fid :
        ### ========== TODO : START ========== ###
        # part 1b: process each line to populate feature_matrix
        lines = fid.readlines()
        for i, line in enumerate(lines):
            wordds = extract_words(line)
            for w in wordds:
                ind = word_list.get(w)
                feature_matrix[i, ind] = 1
        ### ========== TODO : END ========== ###
    return feature_matrix


def test_extract_dictionary(dictionary) :
    err = "extract_dictionary implementation incorrect"
    assert len(dictionary) == 1811, err
    
    exp = [('2012', 0),
           ('carol', 10),
           ('ve', 20),
           ('scary', 30),
           ('vacation', 40),
           ('just', 50),
           ('excited', 60),
           ('no', 70),
           ('cinema', 80),
           ('frm', 90)]
    act = [sorted(dictionary.items(), key=lambda it: it[1])[i] for i in range(0,100,10)]
    assert exp == act, err


def test_extract_feature_vectors(X) :
    err = "extract_features_vectors implementation incorrect"
    
    assert X.shape == (630, 1811), err
    
    exp = np.array([[ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
                    [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.],
                    [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.],
                    [ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,  1.],
                    [ 0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,  1.],
                    [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  1.],
                    [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.],
                    [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.],
                    [ 0.,  1.,  0.,  0.,  1.,  0.,  0.,  0.,  1.,  1.],
                    [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.]])
    act = X[:10,:10]
    assert (exp == act).all(), err


######################################################################
# functions -- evaluation
######################################################################
def custom_loss_fcn(y_pred, y_true):
    return performance(y_true, y_pred, metric="specificity")

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

    ### ========== TODO : END ========== ###


def test_performance() :
    # np.random.seed(1234)
    # y_true = 2 * np.random.randint(0,2,10) - 1
    # np.random.seed(2345)
    # y_pred = (10 + 10) * np.random.random(10) - 10
    
    y_true = [ 1,  1, -1,  1, -1, -1, -1,  1,  1,  1]
    #y_pred = [ 1, -1,  1, -1,  1,  1, -1, -1,  1, -1]
    # confusion matrix
    #          pred pos     neg
    # true pos      tp (2)  fn (4)
    #      neg      fp (3)  tn (1)
    y_pred = [ 3.21288618, -1.72798696,  3.36205116, -5.40113156,  6.15356672,
               2.73636929, -6.55612296, -4.79228264,  8.30639981, -0.74368981]
    metrics = ["accuracy", "f1_score", "auroc", "precision", "sensitivity", "specificity"]
    scores  = [     3/10.,      4/11.,   5/12.,        2/5.,          2/6.,          1/4.]
    
    import sys
    eps = sys.float_info.epsilon
    
    for i, metric in enumerate(metrics) :
        assert abs(performance(y_true, y_pred, metric) - scores[i]) < eps, \
            (metric, performance(y_true, y_pred, metric), scores[i])


def cv_performance(clf, X, y, kf, metrics=["accuracy"]) :
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

    k = kf.get_n_splits(X, y)
    m = len(metrics)
    scores = np.empty((m, k))

    for k, (train, test) in enumerate(kf.split(X, y)) :
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        clf.fit(X_train, y_train)
        # use SVC.decision_function to make ``continuous-valued'' predictions
        y_pred = clf.decision_function(X_test)
        for m, metric in enumerate(metrics) :
            score = performance(y_test, y_pred, metric)
            scores[m,k] = score
            
    return scores.mean(axis=1) # average across columns


def select_param_linear(X, y, kf, metrics=["accuracy"], plot=True) :
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
        model_svc = SVC(C=c, kernel='linear')
        # compute CV scores using cv_performance(...)
        scores[:,j] = cv_performance(model_svc, X, y, kf, metrics)

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


def select_param_rbf(X, y, kf, metrics=["accuracy"]) :
    """
    Sweeps different settings for the hyperparameters of an RBF-kernel SVM,
    calculating the k-fold CV performance for each setting and metric,
    then selects the hyperparameters that maximize the average performance for each metric.
    
    Parameters
    --------------------
        X       -- numpy array of shape (n,d), feature vectors
                     n = number of examples
                     d = number of features
        y       -- numpy array of shape (n,), binary labels {1,-1}
        kf      -- model_selection.KFold or model_selection.StratifiedKFold
        metrics -- list of m strings, metrics
    
    Returns
    --------------------
        params  -- list of m tuples, optimal hyperparameters (C,gamma) for each metric
    """
    
    ### ========== TODO : START ========== ###
    # part 4b: for each metric, select optimal hyperparameters using cross-validation
    
    # create grid of hyperparameters
    # hint: use a small 2x2 grid of hyperparameters for debugging
    C_range = 10.0 ** np.arange(-3, 3)        # dummy, okay to change
    gamma_range = 10.0 ** np.arange(-3, 3)    # dummy, okay to change
    scores = np.empty((len(metrics), len(C_range), len(gamma_range)))

    best_scores = []
    
    # compute CV scores using cv_performance(...)
    param_grid = dict(gamma=gamma_range, C=C_range)
    for j, c in enumerate(C_range):
        for k, gam in enumerate(gamma_range):
            clf = SVC(C = c, gamma = gam)
            performance = cv_performance(clf, X, y, kf, metrics)
            scores[:,j,k] = performance 
    c_opt_L = [] 
    gamma_opt_L = []
    for i, metric in enumerate(metrics):
        grid = scores[i, :, :] 
        c_opt_ind, gamma_opt_ind, best_score = grid_search(grid)
        c_opt = C_range[c_opt_ind]
        c_opt_L.append(c_opt)
        gamma_opt = gamma_range[gamma_opt_ind]
        gamma_opt_L.append(gamma_opt)
        best_scores.append(best_score)
        print "The best classifier for " + metric + " is C=" + str(c_opt) + ", gamma=" + str(gamma_opt)

    # get best hyperparameters 
    best_params = zip(c_opt_L, gamma_opt_L)
    ### ========== TODO : END ========== ###
    print best_scores 
    return best_params


def grid_search(grid):
    """ Helper function for grid search
    """
    c_ind, gam_ind = np.unravel_index(np.argmax(grid, axis=None), grid.shape)
    best_score = grid[c_ind, gam_ind]
    return c_ind, gam_ind, best_score


def performance_CI(clf, X, y, metric="accuracy") :
    """
    Estimates the performance of the classifier using the 95% CI.
    
    Parameters
    --------------------
        clf          -- classifier (instance of SVC or DummyClassifier)
                          [already fit to data]
        X            -- numpy array of shape (n,d), feature vectors of test set
                          n = number of examples
                          d = number of features
        y            -- numpy array of shape (n,), binary labels {1,-1} of test set
        metric       -- string, option used to select performance measure
    
    Returns
    --------------------
        score        -- float, classifier performance
        lower        -- float, lower limit of confidence interval
        upper        -- float, upper limit of confidence interval
    """

    try :
        y_pred = clf.decision_function(X)
    except :  # for dummy classifiers
        y_pred = clf.predict(X)
    score = performance(y, y_pred, metric)
    
    ### ========== TODO : START ========== ###
    # part 5c: use bootstrapping to compute 95% confidence interval
    # hint: use np.random.randint(...) to sample 
    n, d = X.shape  
    t = 1000
    score_list = np.empty(t)
    print "Bootstrapping for " + metric
    for i in range(t):
        X_boot_indices = np.random.randint(n, size=n)
        X_boot = X[X_boot_indices,:]
        y_boot = y[X_boot_indices]
        try :
            y_pred = clf.decision_function(X_boot)
        except :
            y_pred = clf.predict(X_boot)
        score_list[i] = performance(y_boot, y_pred, metric)
    
    # print(score_list)
    lower, upper = confidence_interval(score_list)
    return score, lower, upper
    ### ========== TODO : END ========== ###

def confidence_interval(stats, alpha=0.95):
    """ helper function to calculate 95% CI
    """
    p = ((1.0 - alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats, p))
    p = (alpha+ ((1.0 - alpha)/2.0)) * 100
    upper = min(1.0, np.percentile(stats, p))
    return lower, upper

######################################################################
# functions -- plotting
######################################################################

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


def plot_results(metrics, classifiers, figname,  *args):
    """
    Make a results plot.
    
    Parameters
    --------------------
        metrics      -- list of strings, metrics
        classifiers  -- list of strings, classifiers (excluding baseline classifier)
        args         -- variable length argument
                          results for baseline
                          results for classifier 1
                          results for classifier 2
                          ...
                        each results is a list of tuples ordered by metric
                          typically, each tuple consists of a single element, e.g. (score,)
                          to include error bars, each tuple consists of three elements, e.g. (score, lower, upper)
    """
    
    num_metrics = len(metrics)
    num_classifiers = len(args) - 1
    
    ind = np.arange(num_metrics)  # the x locations for the groups
    width = 0.7 / num_classifiers # the width of the bars
    
    fig, ax = plt.subplots()
    
    # loop through classifiers
    rects_list = []
    for i in xrange(num_classifiers):
        results = args[i+1] # skip baseline
        
        # mean
        means = [it[0] for it in results]
        rects = ax.bar(ind + i * width, means, width, label=classifiers[i])
        rects_list.append(rects)
        
        # errors
        if len(it) == 3:
            errs = [(it[0] - it[1], it[2] - it[0]) for it in results]
            ax.errorbar(ind + i * width, means, yerr=np.array(errs).T, fmt='none', ecolor='k')
    
    # baseline
    results = args[0]
    for i in xrange(num_metrics) :
        xlim = (ind[i] - 0.8 * width, ind[i] + num_classifiers * width - 0.2 * width)
        
        # mean
        mean = results[i][0]
        plt.plot(xlim, [mean, mean], color='k', linestyle='-', linewidth=2)
        
        # errors
        if len(results[i]) == 3:
            err_low = results[i][1]
            err_high = results[i][2]
            plt.plot(xlim, [err_low, err_low], color='k', linestyle='--', linewidth=2)
            plt.plot(xlim, [err_high, err_high], color='k', linestyle='--', linewidth=2)
    
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1)
    ax.set_xticks(ind + width / num_classifiers)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    def autolabel(rects):
        """Attach a text label above each bar displaying its height"""
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    '%.2f' % height, ha='center', va='bottom')
    
    for rects in rects_list:
        autolabel(rects)
    # save fig instead of show because show breaks my computer :(
    plt.savefig(figname)


######################################################################
# main
######################################################################

def main() :
    ### ========== TODO : START ========== ###
    # read the tweets and its labels, with unit tests
    # (nothing to implement, just make sure the tests pass)
    dictionary = extract_dictionary('../data/tweets.txt')
    test_extract_dictionary(dictionary)

    X = extract_feature_vectors('../data/tweets.txt', dictionary)
    test_extract_feature_vectors(X)
    y = read_vector_file('../data/labels.txt')
    ### ========== TODO : END ========== ###
    
    # shuffle data (since file has tweets ordered by movie)
    X, y = shuffle(X, y, random_state=0)
    
    # set random seed
    np.random.seed(1234)
    
    # split the data into training (training + cross-validation) and testing set
    X_train, X_test = X[:560], X[560:]
    y_train, y_test = y[:560], y[560:]
    
    # ========== TODO : START ========== ###
    # part 2a: metrics, with unit test
    # (nothing to implement, just make sure the test passes)
    metrics = ["accuracy", "f1_score", "auroc", "precision", "sensitivity", "specificity"]
    test_performance()
    ### ========== TODO : END ========== ###
    
    # folds
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=np.random.randint(1234))
    
    # hyperparameter selection for linear-SVM
    best_params = select_param_linear(X_train, y_train, skf, metrics)
    print best_params
    # hyperparameter selection using RBF-SVM
    best_params = select_param_rbf(X_train, y_train, skf, metrics)
    print best_params
    
    ### ========== TODO : START ========== ###
    # part 5a: train linear- and RBF-SVMs with selected hyperparameters
    # hint: use only linear-SVM (comment out the RBF-SVM) for debugging
    c_opt_linear = 1.0
    c_opt_rbf = 100.0
    gamma_opt = 0.01
    clf_bl = BaselineCLF()
    clf_linear = SVC(C = c_opt_linear, kernel = 'linear')
    clf_rbf = SVC(C = c_opt_rbf, gamma = gamma_opt)

    # train on entire training set 
    # measure performance on training set 
    # if error on training data is high --> high bias/underfitting 
    clf_bl.fit(X_train, y_train)
    clf_linear.fit(X_train, y_train)
    clf_rbf.fit(X_train, y_train)

    performances_bl = []
    performances_linear = []
    performances_rbf = []
    for metric in metrics:
        performances_bl.append(performance(y_train, clf_bl.predict(X_train),  metric))
        performances_linear.append(performance(y_train, clf_linear.predict(X_train), metric))
        performances_rbf.append(performance(y_train, clf_rbf.predict(X_train), metric))

    #tuplized everything
    performances_bl = [(i,) for i in list(performances_bl)]
    performances_linear = [(i,) for i in list(performances_linear)]
    performances_rbf = [(i,) for i in list(performances_rbf)]
    
    # part 5b: report performance on train data
    #          use plot_results(...) to make plot
    plot_results(metrics, ['linear', 'RBF'], "results.png", performances_bl, performances_linear, performances_rbf)
    
    # part 5d: use bootstrapping to report performance on test data
    #          use plot_results(...) to make plot

    performances_bl = []
    performances_linear = []
    performances_rbf = []
    for metric in metrics:
        print "CI for Baseline: "
        performances_bl.append(performance_CI(clf_bl, X_test, y_test, metric))
        print "CI for Linear: "
        performances_linear.append(performance_CI(clf_linear, X_test, y_test, metric))
        print "CI for RBF: "
        performances_rbf.append(performance_CI(clf_rbf, X_test, y_test, metric))

    plot_results(metrics, ['linear', 'RBF'], "results_bootstrap.png", performances_bl, performances_linear, performances_rbf)

    ### ========== TODO : END ========== ###
    
    ### ========== TODO : START ========== ###
    # part 6: identify important features

    # create reverse dictionary for important feature identification
    clf_linear.fit(X, y)
    rev_dict = {v: k for k, v in dictionary.items()}
    argsorted = np.argsort(clf_linear.coef_)[0]
    important_pwords = []
    important_nwords = []
    bottom10 = argsorted[:10]
    top10 = argsorted[(argsorted.size-10):]
    for i in list(top10):
        important_pwords.append(rev_dict[int(i)])
    for i in list(bottom10):
        important_nwords.append(rev_dict[int(i)])
    print clf_linear.coef_[0][top10]
    print important_pwords

    print clf_linear.coef_[0][bottom10]
    print important_nwords

    ### ========== TODO : END ========== ###
    
    ### ========== TODO : START ========== ###
    # Twitter contest
    # uncomment out the following, and be sure to change the filename
    """
    X_held = extract_feature_vectors('../data/held_out_tweets.txt', dictionary)
    # your code here
    # y_pred = best_clf.decision_function(X_held)
    write_label_answer(y_pred, '../data/yjw_twitter.txt')
    """
    ### ========== TODO : END ========== ###


if __name__ == "__main__" :
    main()
