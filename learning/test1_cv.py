from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os.path
import sklearn.svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.utils import shuffle
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import PCA
from sklearn.lda import LDA
from sklearn.feature_selection import RFECV
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

import pickle

def get_feature1(pithc_duration_hist, one_octave, pitch_only):    
    new_feature = {}
    for key in pithc_duration_hist.iterkeys():
        pitch, duration = key
        if one_octave:
            if pitch == 0:
                pitch = -1.0
            else:
                pitch = pitch % 12
        if pitch_only:
            duration = 0
        
        new_key = (pitch, duration)
        if new_key in new_feature:
            new_feature[(pitch, duration)] += pithc_duration_hist[key]
        else:
            new_feature[(pitch, duration)] = pithc_duration_hist[key]

    return new_feature


if __name__  == "__main__":
    # Get the pitch_duration histogram
    pitch_duration = pickle.load(open('pitchduration.p', 'rb'))
  
    # Get the secure Jos data
    jos_sec_set = set()
    with open('../data/Josquin_secure.txt') as f:
        for line in f:
            jos_sec_set.add(line.strip())
            
    others_set = set()
    with open('../data/others.txt') as f:
        for line in f:
            others_set.add(line.split('-')[0])
   
    #  feature extraction
    song_feature = []
    song_label = []
    song_name = []
    
    for name in pitch_duration.iterkeys():
        if name in jos_sec_set:
            label = 1
        elif name in others_set:
            label = 0 
        else:
            label = -1
        
        feature = get_feature1(pitch_duration[name], one_octave=True, pitch_only=False)
        song_feature.append(feature)
        song_label.append(label)
        song_name.append(name)
    
    vec = DictVectorizer(sparse=False)
    song_feature = vec.fit_transform(song_feature)
    song_label = np.array(song_label)
    song_name = np.array(song_name)
    
    train_index = (song_label != -1) 
    X = song_feature[train_index]
    y = song_label[train_index]
    
            
    ## fitting method            
    #clf = LogisticRegression(C=10, class_weight='auto')
            
    #estimators = [('normalizer', TfidfTransformer(norm='l1', use_idf=False)), ('logistic', LogisticRegression(C=1000, class_weight='auto'))]
    #clf = Pipeline(estimators)            
            
    #clf = MultinomialNB()
    #clf = sklearn.svm.LinearSVC(C=130, class_weight='auto')
    
    
    #estimators = [('normalizer', preprocessing.Normalizer(norm='l1')), ('svm', sklearn.svm.LinearSVC(C=130, class_weight='auto'))]
    #clf = Pipeline(estimators)
    
    #estimators = [('normalizer', TfidfTransformer(norm='l1')), ('svm', sklearn.svm.LinearSVC(C=130, class_weight='auto'))]
    #clf = Pipeline(estimators)
    
    #clf = DummyClassifier(strategy='most_frequent',random_state=0)
    #clf = DummyClassifier(strategy='stratified',random_state=0)
    #clf = DummyClassifier(strategy='uniform',random_state=0)
    #clf = LDA(n_components=3)


    ## Cross_validation
#    scores = cross_validation.cross_val_score(clf, X, y,
#                                              scoring='f1', cv=10, 
#                                              verbose=1 ,n_jobs=1)
#    score_mean = np.mean(scores)
#    score_std = np.std(scores)
#    
#    print 'f1', score_mean, score_std
    
    
    ## PCA/LDA visualization
    #pca = PCA(n_components=2)
    #X_r = pca.fit(X).transform(X)
    
    #lda = LDA(n_components=3)
    #X_r = lda.fit(X, y).transform(X)
    #
    ## Percentage of variance explained for each components
    #print('explained variance ratio (first two components): %s'
    #      % str(pca.explained_variance_ratio_))
    #
    #plt.figure()
    #for c, i, target_name in zip("rb", [0, 1], ['non-jos', 'jos']):
    #    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=target_name)
    #plt.legend()
    #plt.title('PCA of IRIS dataset')
    #
    #plt.show()
    
    
    ##backward feature reduction
    #rfecv = RFECV(estimator=clf, step=10, cv=cross_validation.StratifiedShuffleSplit(y, test_size=0.1),
    #              scoring='f1', verbose=1)
    #rfecv.fit(X, y)
    #
    #print("Optimal number of features : %d" % rfecv.n_features_)
    #
    ## Plot number of features VS. cross-validation scores
    #import pylab as pl
    #pl.figure()
    #pl.xlabel("Number of features selected")
    #pl.ylabel("Cross validation score (nb of misclassifications)")
    #pl.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    #pl.show()
    
###Learning Curve###    
#    train_size = range(10,int(y.shape[0]*0.8),10)
#    
#    train_errors = np.zeros((10,len(train_size)))
#    test_errors = np.zeros((10,len(train_size)))    
#
#    print "Learning Curve"
#    k_fold = cross_validation.StratifiedShuffleSplit(y, test_size=0.1)
#    for k, (train, test) in enumerate(k_fold):
#        print  "round" ,k
#        for i, p in enumerate(train_size): 
#            #make sure  we have the same ratio between two classes
#            train_index, test_index = next(iter(cross_validation.StratifiedShuffleSplit(y[train],train_size=p)))
#            #print len(train_index)
#            #fit data
#            clf.fit(X[train[train_index]], y[train[train_index]])
#            #get error
#            train_errors[k,i] = sklearn.metrics.f1_score(y[train[train_index]], 
#                                                         clf.predict(X[train[train_index]]))
#            test_errors[k,i]  = sklearn.metrics.f1_score(y[test], 
#                                                         clf.predict(X[test]))
#    
#    train_errors_mean = np.mean(train_errors,axis=0)
#    test_errors_mean = np.mean(test_errors,axis=0)
#    
#    plt.figure()
#    plt.plot(train_size, train_errors_mean, 'b.', label='train_error', hold=True) 
#    plt.plot(train_size, test_errors_mean, 'r*',label='test_error') 
#    plt.legend(loc=4)
#    plt.xlabel('Size of training set')
#    plt.ylabel('f1 score')
#    
#    plt.show()

#tuned_parameters = [{'svm__C': np.logspace(0,3,10)}]

###k-fold grid search
#k_fold = cross_validation.StratifiedShuffleSplit(y, test_size=0.1)
#for k, (train, test) in enumerate(k_fold):
#    print("# Tuning hyper-parameters for %s" % 'f1')
#    
#    clf_best = GridSearchCV(clf, tuned_parameters, cv=10, scoring='f1', verbose=1)
#    clf_best.fit(X[train], y[train])
#    
#    print "Best parameters set found on development set:"
#    print clf_best.best_params_
#    #print clf_best.best_score_
##    print "Grid scores on development set:"
##    print
##    for params, mean_score, scores in clf_best.grid_scores_:
##        print("%0.3f (+/-%0.03f) for %r"
##              % (mean_score, scores.std(), params))
#    
#    
##    print("Detailed classification report:")
##    print("The model is trained on the full development set.")
##    print("The scores are computed on the full evaluation set.")
##    y_true, y_pred = song_label[test], clf_best.predict(song_feature[test])
##    print(classification_report(y_true, y_pred))
#    print clf_best.score(X[test], y[test])
        