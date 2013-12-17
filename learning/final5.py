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
labels[labels == -1] = 0

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


SVD1 = TruncatedSVD(n_components=300)
SVD2 = TruncatedSVD(n_components=200)
feature1 = SVD1.fit_transform(feature1)
feature2 = SVD2.fit_transform(feature2)

feature = np.hstack((feature1, feature2))
feature_unsec = feature[labels == 2]
feature = feature[labels != 2]
labels = labels[labels != 2]

clf1 = SVC(C=10000, gamma=0.75, probability=True)
#clf2 = LinearSVC(C=100, probability=True)
clf2 = SVC(kernel='linear', C=100, probability=True)
clf3 = LogisticRegression(C=100)

sfk = cv.StratifiedShuffleSplit(labels, 100)
scores = []
for train, test in sfk:
    score = []
    train_set = feature[train]
    test_set = feature[test]

    clf1.fit(train_set, labels[train])
    pred1 = clf1.predict(test_set)
    prob1 = clf1.predict_proba(test_set)[:, 1]
    clf2.fit(train_set, labels[train])
    pred2 = clf2.predict(test_set)
    prob2 = clf2.predict_proba(test_set)[:, 1]
    clf3.fit(train_set, labels[train])
    pred3 = clf3.predict(test_set)
    prob3 = clf3.predict_proba(test_set)[:, 1]

    prob_avg = (prob1 + prob2 + prob3) / 3
    pred = np.zeros_like(pred1)
    pred[prob_avg > 0.7] = 1
    #pred = pred1 * pred2 * pred3

    score.append(accuracy_score(labels[test], pred))
    score.append(precision_score(labels[test], pred))
    score.append(recall_score(labels[test], pred))
    score.append(f1_score(labels[test], pred))
    scores.append(score)
avg = np.average(scores, axis=0)

print avg

