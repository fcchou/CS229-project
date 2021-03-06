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
from sklearn.neural_network import BernoulliRBM
from sklearn.ensemble import RandomForestClassifier

import pickle

def get_feature1(pithc_duration_hist, one_octave=False, pitch_only=False):    
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
            new_feature[new_key] += pithc_duration_hist[key]
        else:
            new_feature[new_key] = pithc_duration_hist[key]

    return new_feature

def get_counterpoint(counter_point_hist, one_octave=False):
    new_feature = {}
    for key in counter_point_hist.iterkeys():
        (v1, h, v2) = key
        if one_octave:
            newv1 = v1 % 7 - 7*(v1<7)
            newv2 = v2 - (v1-newv1)
        else:
            newv1, newv2 = v1, v2
            
        new_key = (newv1, h, newv2)
        if new_key in new_feature:
            new_feature[new_key] += counter_point_hist[key]
        else:
            new_feature[new_key] = counter_point_hist[key]

    return new_feature   

if __name__  == "__main__":
    # Get the pitch_duration histogram
    pitch_duration = pickle.load(open('pitchduration.p', 'rb'))
    counter_point = pickle.load(open('counterpoint.p', 'rb'))
    
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
    
    # prepare name list and label 
    for name in pitch_duration.iterkeys():
        if name in jos_sec_set: # secured set
            label = 1
        elif name in others_set: # negative set
            label = 0 
        elif name[:3] == 'Jos': # target set
            label = -1
        else:
            continue

        song_label.append(label)
        song_name.append(name)
        
    # prepare feature
    song_counter_point = []        
    song_pitch_duration = []
    for name in song_name:
        feature = get_feature1(pitch_duration[name], one_octave=False, pitch_only=False)
        song_pitch_duration.append(feature)
        
        feature = get_counterpoint(counter_point[name], one_octave=False)
        song_counter_point.append(feature)
        
    
    
    vec_counter_point = DictVectorizer(sparse=False)
    song_counter_point = vec_counter_point.fit_transform(song_counter_point)
    
    vec_pitch_duration = DictVectorizer(sparse=False)
    song_pitch_duration = vec_counter_point.fit_transform(song_pitch_duration)    
    
    song_label = np.array(song_label)
    song_name = np.array(song_name)
    
    train_index = (song_label != -1) 
    X_cp = song_counter_point[train_index]
    X_pd = song_pitch_duration[train_index]
    y = song_label[train_index]
    
    
    # test normalization
#    normalizer = TfidfTransformer(norm='l2', use_idf=True)
#    X_norm = normalizer.fit_transform(X_cp).toarray()
            
    ## fitting method            
#    clf = LogisticRegression(C=300, class_weight='auto')
            
#    estimators = [('normalizer', TfidfTransformer(norm='l1', use_idf=True)), 
#                  ('logistic', LogisticRegression(C=1000, class_weight='auto'))]
#    clf = Pipeline(estimators)            
            
    clf = MultinomialNB()    
    

#    estimators = [('normalizer', preprocessing.Normalizer(norm='l2')),     
#                  ('pca', sklearn.decomposition.TruncatedSVD(n_components=30)),
#                  ('svm', sklearn.svm.LinearSVC(C=130, class_weight='auto'))]
#    clf = Pipeline(estimators)    
    
#    estimators = [('normalizer', TfidfTransformer(norm='l2', use_idf=True)),
#                  ('svm', sklearn.svm.SVC(kernel='poly', degree=1, C=27825, 
#                                          class_weight='auto'))]
#    
#    clf = Pipeline(estimators)
#    
#    estimators = [('normalizer', TfidfTransformer(norm='l1', use_idf=True)),
#                  ('svm', sklearn.svm.LinearSVC(C=130, class_weight='auto'))]
#    clf = Pipeline(estimators)
    
#    tuned_parameters = {'svm__C': np.logspace(0,3,10)}
    
    #clf = DummyClassifier(strategy='most_frequent',random_state=0)
    #clf = DummyClassifier(strategy='stratified')
#    clf = DummyClassifier(strategy='uniform')
    #clf = LDA(n_components=3)
    #clf = RandomForestClassifier(n_estimators=10, max_features=30)    
    
#    clf = Pipeline(steps=[
#                          ('rbm', BernoulliRBM(learning_rate =0.005, n_iter=30, n_components=30, verbose=True)), 
#                          ('logistic', LogisticRegression(C = 100.0))])
    

    # simple test
#    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y,
#                                                                         test_size=0.1)
#    clf.fit(X_train, y_train)
#    y_predict = clf.predict(X_test)
#    print classification_report( y_test, y_predict)
        
    # Cross_validation
#    scores = cross_validation.cross_val_score(clf, X_cp, y,
#                                              scoring='precision', cv=10, 
#                                              verbose=1 ,n_jobs=1)
#    score_mean = np.mean(scores)
#    score_std = np.std(scores)
#    
#    print 'f1', score_mean, score_std



#from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
#X, y = shuffle(X_cp, y)
#
#accu = []
#prec = []
#reca = []
#f1 =[]
#cv = cross_validation.StratifiedKFold(y, n_folds=10)
#for train, test in cv:
#    clf.fit(X[train], y[train])
#    pred = clf.predict(X[test])
#    accu.append(accuracy_score(y[test], pred))
#    prec.append(precision_score(y[test], pred))
#    reca.append(recall_score(y[test], pred))
#    f1.append(f1_score(y[test], pred))
#print np.average(accu), np.average(prec), np.average(reca), np.average(f1)

    
#    ## PCA/LDA visualization
#    estimators = [('normalizer', preprocessing.Normalizer(norm='l1')),                  
#                  ('pca', PCA(n_components=3))]
#    clf = Pipeline(estimators)
#    pca = PCA(n_components=2)
#    X_r = clf.fit(X).transform(X)
#    
    #lda = LDA(n_components=3)
    #X_r = lda.fit(X, y).transform(X)
    #
    ## Percentage of variance explained for each components
    #print('explained variance ratio (first two components): %s'
    #      % str(pca.explained_variance_ratio_))
    #
#    plt.figure()
#    for c, i, target_name in zip("rb", [0, 1], ['non-jos', 'jos']):
#        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=target_name)
#    plt.legend()
#    plt.title('PCA of IRIS dataset')
#    #
#    plt.show()
#    
    
    ##backward feature reduction
#    rfecv = RFECV(estimator=clf, step=10, cv=cross_validation.StratifiedShuffleSplit(y, test_size=0.1),
#                  scoring='f1', verbose=1)
#    rfecv.fit(X, y)
#    
#    print("Optimal number of features : %d" % rfecv.n_features_)
    
    # Plot number of features VS. cross-validation scores
#    pl.figure()
#    pl.xlabel("Number of features selected")
#    pl.ylabel("Cross validation score (nb of misclassifications)")
#    pl.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
#    pl.show()
    
##Learning Curve: single feature###    
    train_size = range(10,int(y.shape[0]*0.8),10)
    
    train_errors = np.zeros((10,len(train_size)))
    test_errors = np.zeros((10,len(train_size)))    
    
    # set target feature
    X_data = X_cp
    
    print "Learning Curve"
    k_fold = cross_validation.StratifiedShuffleSplit(y, test_size=0.1)
    for k, (train, test) in enumerate(k_fold):
        print  "round" ,k
        for i, p in enumerate(train_size):       
            
            #make sure  we have the same ratio between two classes
            train_index, test_index = next(iter(cross_validation.StratifiedShuffleSplit(y[train],train_size=p)))
            #print len(train_index)
           
            X = X_data[train[train_index]]
            
#            svd = sklearn.decomposition.TruncatedSVD(n_components=100)
#            X = svd.fit_transform(X)
            
            #fit data
            clf.fit(X, y[train[train_index]])
            
            #get error
            
            train_errors[k,i] = sklearn.metrics.accuracy_score(y[train[train_index]], 
                                                         clf.predict(X))
                                                         
            X = X_data[test]            
            test_errors[k,i]  = sklearn.metrics.accuracy_score(y[test], 
                                                         clf.predict(X))
    
    train_errors_mean = np.mean(train_errors,axis=0)
    train_error_std = np.std(train_errors,axis=0)/3.16
    test_errors_mean = np.mean(test_errors,axis=0)
    test_error_std = np.std(test_errors,axis=0)/3.16
    
    plt.figure()
    plt.plot(train_size, train_errors_mean, 'b.', label='training', hold=True) 
    plt.plot(train_size, train_errors_mean+train_error_std, 'b-', hold=True) 
    plt.plot(train_size, train_errors_mean-train_error_std, 'b-', hold=True) 
    plt.plot(train_size, test_errors_mean, 'r*',label='testing') 
    plt.plot(train_size, test_errors_mean+test_error_std, 'r-', hold=True) 
    plt.plot(train_size, test_errors_mean-test_error_std, 'r-', hold=True) 
    plt.legend(loc=4)
    plt.xlabel('Size of training set')
    plt.ylabel('f1 score')
    
    plt.show()    

##Learning Curve: single feature with grid search###    
#    train_size = range(30,int(y.shape[0]*0.8),30)
#    
#    train_errors = np.zeros((10,len(train_size)))
#    test_errors = np.zeros((10,len(train_size)))    
#    
#    # set target feature
#    X_data = X_cp
#    
#    print "Learning Curve"
#    k_fold = cross_validation.StratifiedShuffleSplit(y, test_size=0.1)
#    for k, (train, test) in enumerate(k_fold):
#        print  "round" ,k
#        for i, p in enumerate(train_size):       
#            
#            #make sure  we have the same ratio between two classes
#            train_index, test_index = next(iter(cross_validation.StratifiedShuffleSplit(y[train],train_size=p)))
#            #print len(train_index)
#           
#            X = X_data[train[train_index]]
#            
##            svd = sklearn.decomposition.TruncatedSVD(n_components=100)
##            X = svd.fit_transform(X)
#            
#            clf_grid = GridSearchCV(clf, tuned_parameters, cv=10, scoring='f1', verbose=0)
#            
#            #fit data
#            clf_grid.fit(X, y[train[train_index]])
#            
#            #get error
#            
#            train_errors[k,i] = sklearn.metrics.f1_score(y[train[train_index]], 
#                                                         clf_grid.predict(X))
#                                                         
#            X = X_data[test]            
#            test_errors[k,i]  = sklearn.metrics.f1_score(y[test], 
#                                                         clf_grid.predict(X))
#    
#    train_errors_mean = np.mean(train_errors,axis=0)
#    test_errors_mean = np.mean(test_errors,axis=0)
#    
#    plt.figure()
#    plt.plot(train_size, train_errors_mean, 'b.', label='training', hold=True) 
#    plt.plot(train_size, test_errors_mean, 'r*',label='testing') 
#    plt.legend(loc=4)
#    plt.xlabel('Size of training set')
#    plt.ylabel('f1 score')
#    
#    plt.show()    

###Learning Curve: combined feature###    
#    train_size = range(10,int(y.shape[0]*0.8),30)
#    
#    train_errors = np.zeros((10,len(train_size)))
#    test_errors = np.zeros((10,len(train_size)))    
#
#    print "Learning Curve"
#    k_fold = cross_validation.StratifiedShuffleSplit(y, test_size=0.1)
#    for k, (train, test) in enumerate(k_fold):
#        print  "round" ,k
#        for i, p in enumerate(train_size): 
#            clf = sklearn.svm.LinearSVC(C=130, class_weight='auto')        
#            
#            #make sure  we have the same ratio between two classes
#            train_index, test_index = next(iter(cross_validation.StratifiedShuffleSplit(y[train],train_size=p)))
#            #print len(train_index)
#            
#            normalizer_cp = TfidfTransformer(norm='l1' , use_idf=True)   
#            X_norm_cp = normalizer_cp.fit_transform(X_cp[train[train_index]]).toarray()  
#                      
#            normalizer_pd = TfidfTransformer(norm='l1' , use_idf=True)   
#            X_norm_pd = normalizer_pd.fit_transform(X_pd[train[train_index]]).toarray()         
#            
#            X = np.hstack((X_norm_cp, X_norm_pd))
#            
##            svd = sklearn.decomposition.TruncatedSVD(n_components=100)
##            X = svd.fit_transform(X)
#            
#            #print X[0]
#            #fit data
#            clf.fit(X, y[train[train_index]])
#            
#            #get error
#            
#            train_errors[k,i] = sklearn.metrics.f1_score(y[train[train_index]], 
#                                                         clf.predict(X))
#                                                         
#            X_norm_cp = normalizer_cp.fit_transform(X_cp[test]).toarray()                        
#            X_norm_pd = normalizer_pd.fit_transform(X_pd[test]).toarray()         
#            
#            X = np.hstack((X_norm_cp, X_norm_pd))               
##            X = svd.transform(X)
#            
#            test_errors[k,i]  = sklearn.metrics.f1_score(y[test], 
#                                                         clf.predict(X))
#    
#    train_errors_mean = np.mean(train_errors,axis=0)
#    test_errors_mean = np.mean(test_errors,axis=0)
#    
#    plt.figure()
#    plt.plot(train_size, train_errors_mean, 'b.', label='training', hold=True) 
#    plt.plot(train_size, test_errors_mean, 'r*',label='testing') 
#    plt.legend(loc=4)
#    plt.xlabel('Size of training set')
#    plt.ylabel('f1 score')
#    
#    plt.show()

tuned_parameters = [{'svm__C': np.logspace(0,5,10)}]

##k-fold grid search
#k_fold = cross_validation.StratifiedShuffleSplit(y, test_size=0.1)
#for k, (train, test) in enumerate(k_fold):
#    print("# Tuning hyper-parameters for %s" % 'f1')
#    
#    clf_best = GridSearchCV(clf, tuned_parameters, cv=10, scoring='f1', verbose=1)
#    clf_best.fit(X_cp[train], y[train])
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
#    y_true, y_pred = y[test], clf_best.predict(X_cp[test])
##    print(classification_report(y_true, y_pred))
#    print clf_best.score(y_true, y_pred)
        