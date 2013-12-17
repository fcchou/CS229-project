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

#feature1 = extractor.feature_cp[0]
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

#mask = np.logical_and(labels != 2, labels != -1)
mask = labels != 2
feature1 = feature1[mask]
feature2 = feature2[mask]
labels = labels[mask]

feature = np.hstack((feature1, feature2))
#feature = feature1
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
        train_label = labels[train]
        test_label = labels[test].copy()

        mask1_train = train_label != 0
        mask2_train = train_label != -1
        mask1_test = test_label != 0
        mask2_test = test_label != -1

        clf.fit(train_set[mask1_train], train_label[mask1_train])
        # pred1 = np.zeros_like(test_label)
        # pred1[mask1_test] = clf.predict(test_set[mask1_test])
        pred1 = clf.predict(test_set)

        clf.fit(train_set[mask2_train], train_label[mask2_train])
        # pred2 = np.zeros_like(test_label)
        # pred2[mask2_test] = clf.predict(test_set[mask2_test])
        pred2 = clf.predict(test_set)

        test_label[test_label == -1] = 0
        pred = np.zeros_like(test_label)
        pred[np.logical_and(pred1 == 1, pred2 == 1)] = 1

        score.append(accuracy_score(test_label, pred))
        score.append(precision_score(test_label, pred))
        score.append(recall_score(test_label, pred))
        score.append(f1_score(test_label, pred))
        scores.append(score)
    avg = np.average(scores, axis=0)
    return avg
print validate(SVC(C=10000, gamma=0.75), feature, 500)
print validate(LinearSVC(C=100), feature, 500)
print validate(LogisticRegression(C=100), feature, 500)
