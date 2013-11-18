import sys
sys.path.append('../')
from jos_learn.features import FeatureExtract

import numpy as np
import matplotlib.pyplot as plt
import sklearn.cross_validation as cv
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer
import cPickle as pickle
from sklearn.decomposition import TruncatedSVD, PCA, KernelPCA
from sklearn.feature_selection import SelectKBest, chi2

# Setup the features
extractor = FeatureExtract()
labels = extractor.labels
works = extractor.works

# Get shell's features
#folder = '../shell/pd_class_arrs'
#all_feature = []
#for work in works:
#    data = np.load('%s/%s.npy' % (folder, work))
#    hist = np.bincount(data)[1:]
#    length = hist.shape[0]
#    if length < 15:
#        hist = np.hstack((hist, np.zeros(15 - length)))
#    all_feature.append(hist)
#all_feature = np.array(all_feature)

feature1 = extractor.feature_cp[0]
#all_feature = np.hstack((all_feature, extractor.feature_ps_ql_pair[0]))

feature_shell = []
length = [2, 3]
for l in length:
    folder = '../shell/length%d_no_mirror' % l
    dict_list = []
    for work in works:
        data = pickle.load(open('%s/%s.pkl' % (folder, work), 'rb'))
        dict_list.append(data)
    feature, names = extractor._vectorize(dict_list)
    feature_shell.append(feature)
feature_shell = np.hstack(feature_shell)

    #folder = '../shell/length2'
#dict_list = []
#for work in works:
#    data = pickle.load(open('%s/%s.pkl' % (folder, work), 'rb'))
#    dict_list.append(data)
#all_feature1, names = extractor._vectorize(dict_list)
#
#all_feature = np.array(all_feature)

#feature1, feature_name1 = extractor.feature_cp
#feature2, feature_name2 = extractor.feature_ps_ql_pair
#
normalizer = TfidfTransformer()
feature1 = normalizer.fit_transform(feature1).toarray()
feature2 = normalizer.fit_transform(feature_shell).toarray()


SVD1 = TruncatedSVD(n_components=300)
SVD2 = TruncatedSVD(n_components=200)
feature1 = SVD1.fit_transform(feature1)
feature2 = SVD2.fit_transform(feature2)

feature1 = feature1[labels != 2]
feature2 = feature2[labels != 2]
labels = labels[labels != 2]

feature = np.hstack((feature1, feature2))
#feature2 = Sel.fit_transform(feature2, labels)

#feature = np.hstack((feature1, feature2))

#print all_feature.shape

#feature2_norm = normalizer.fit_transform(feature2).toarray()
#
#feature1_unsecure = feature1[labels == 2]
#feature2_unsecure = feature2[labels == 2]
#feature1_unsecure_norm = feature1_norm[labels == 2]
#feature2_unsecure_norm = feature2_norm[labels == 2]

def validate(clf, feature, k):
    sfk = cv.StratifiedShuffleSplit(labels, 100)
    scores = []
    for train, test in sfk:
        score = []
        train_set = feature[train]
        test_set = feature[test]
        #Sel = SelectKBest(k=k)
        #train_set = Sel.fit_transform(feature[train], labels[train])
        #test_set = Sel.transform(feature[test])
        clf.fit(train_set, labels[train])
        pred = clf.predict(test_set)
        score.append(accuracy_score(labels[test], pred))
        score.append(precision_score(labels[test], pred))
        score.append(recall_score(labels[test], pred))
        score.append(f1_score(labels[test], pred))
        scores.append(score)
    avg = np.average(scores, axis=0)
    return avg
print validate(SVC(C=10000, gamma=1), feature, 500)
print validate(LinearSVC(C=100), feature, 500)
print validate(LogisticRegression(C=100), feature, 500)

#
# Grid search for parameters
#sfk = cv.StratifiedShuffleSplit(labels, 40)
#params = {'C':[1, 100, 10000, 100000]}
#gs = GridSearchCV(
#        SVC(), params, scoring='f1',
#        n_jobs=8, cv=sfk)
#gs.fit(feature, labels)
#print gs.best_score_, gs.best_params_

#C_lsvc = 10
#
#C_logi_1 = 100
#C_logi_2 = 10000
#
#svm = LinearSVC(C=C_lsvc, class_weight='auto')
#logi1 = LogisticRegression(C=C_logi_1)
#logi2 = LogisticRegression(C=C_logi_2)
#nb = MultinomialNB()

# Compute the cv stat
#def validate(clf, feature):
#    sfk = cv.StratifiedShuffleSplit(labels, 100)
#    scores = []
#    for train, test in sfk:
#        score = []
#        clf.fit(feature[train], labels[train])
#        pred = clf.predict(feature[test])
#        score.append(accuracy_score(labels[test], pred))
#        score.append(precision_score(labels[test], pred))
#        score.append(recall_score(labels[test], pred))
#        score.append(f1_score(labels[test], pred))
#        scores.append(score)
#    avg = np.average(scores, axis=0)
#    return avg
#
#names = ['Lsvc_cp', 'Logi_cp', 'Nb_cp', 'Lsvc_pd', 'Logi_pd', 'Nb_pd']
#all_cv_scores = []
#all_cv_scores.append(validate(svm, feature1_norm))
#all_cv_scores.append(validate(logi1, feature1_norm))
#all_cv_scores.append(validate(nb, feature1))
#all_cv_scores.append(validate(svm, feature2_norm))
#all_cv_scores.append(validate(logi2, feature2_norm))
#all_cv_scores.append(validate(nb, feature2))
#
#for scores, name in zip(all_cv_scores, names):
#    print '%8s' % name,
#    for score in scores:
#        print '%7.2f' % (score * 100),
#    print ''

# Plot the learning curve

# Prepare the shrink dataset
#sfk = cv.StratifiedKFold(labels, 10)
#sel_idx = []
#data_size = []
#for train, test in sfk:
#    if not sel_idx:
#        sel_idx.append(test)
#    else:
#        new_idx = np.append(sel_idx[-1], test)
#        sel_idx.append(new_idx)
#    data_size.append(sel_idx[-1].shape[0])
#
#
#n = 2
#sel_idx = sel_idx[n:]
#data_size = data_size[n:]
#
#
#def get_learning_curve(clf, feature):
#    test_scores = []
#    train_scores = []
#    for idx in sel_idx:
#        feature_curr = feature[idx]
#        labels_curr = labels[idx]
#        # test error
#        sfk = cv.StratifiedShuffleSplit(labels_curr, 100)
#        score = cv.cross_val_score(
#            clf, feature_curr, labels_curr, 'f1', sfk, n_jobs=8)
#        test_scores.append(np.average(score))
#        # training error
#        clf.fit(feature_curr, labels_curr)
#        pred = clf.predict(feature_curr)
#        train_scores.append(f1_score(labels_curr, pred))
#    return test_scores, train_scores
#
#fig = plt.figure(figsize=(10, 5))
#
#
#def plot(clf, feature, idx):
#    fig.add_subplot(2, 3, idx)
#    test_scores, train_scores = get_learning_curve(clf, feature)
#    plt.plot(data_size, test_scores, 'r-o')
#    plt.plot(data_size, train_scores, 'k-o')
#    plt.tight_layout()
#    plt.ylim(0.3, 1.05)
#
#plot(nb, feature2, 1)
#plot(logi2, feature2_norm, 2)
#plot(svm, feature2_norm, 3)
#plot(nb, feature1, 4)
#plot(logi1, feature1_norm, 5)
#plot(svm, feature1_norm, 6)
#
#plt.show()


# Classification of unsecure set
#def unsecure_test(clf, test, train, tag):
#    clf.fit(train, labels)
#    pred = clf.predict(test)
#    print tag, np.sum(pred), test.shape[0] - sum(pred)
#    return pred
#
#all_pred = []
#all_pred.append(unsecure_test(nb, feature2_unsecure, feature2, 'NB_pd'))
#all_pred.append(unsecure_test(logi2, feature2_unsecure_norm, feature2_norm, 'Logi_pd'))
#all_pred.append(unsecure_test(svm, feature2_unsecure_norm, feature2_norm, 'SVM_pd'))
#all_pred.append(unsecure_test(nb, feature1_unsecure, feature1, 'NB_cp'))
#all_pred.append(unsecure_test(logi1, feature1_unsecure_norm, feature1_norm, 'Logi_cp'))
#all_pred.append(unsecure_test(svm, feature1_unsecure_norm, feature1_norm, 'SVM_cp'))
#
#intersect = all_pred[-1]
#for pred in all_pred[3:]:
#    intersect *= pred
#print np.sum(intersect), intersect.shape[0] - np.sum(intersect)
