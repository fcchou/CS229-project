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

feature1 = extractor.feature_cp[0]

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

normalizer = TfidfTransformer()
feature1 = normalizer.fit_transform(feature1).toarray()
feature2 = normalizer.fit_transform(feature_shell).toarray()


SVD1 = TruncatedSVD(n_components=400)
SVD2 = TruncatedSVD(n_components=400)
feature1 = SVD1.fit_transform(feature1)
feature2 = SVD2.fit_transform(feature2)

feature1 = feature1[labels != 2]
feature2 = feature2[labels != 2]
labels = labels[labels != 2]

feature = np.hstack((feature1, feature2))

from sklearn.feature_selection import RFECV

sfk = cv.StratifiedKFold(labels, 10)
scores = []
for train, test in sfk:
    score = []
    train_set = feature[train]
    test_set = feature[test]
    clf = RFECV(
        LinearSVC(C=100),
        cv=cv.StratifiedKFold(labels[train], 10),
        scoring='f1')
    clf.fit(train_set, labels[train])
    pred = clf.predict(test_set)
    score.append(accuracy_score(labels[test], pred))
    score.append(precision_score(labels[test], pred))
    score.append(recall_score(labels[test], pred))
    score.append(f1_score(labels[test], pred))
    scores.append(score)
avg = np.average(scores, axis=0)
print avg
