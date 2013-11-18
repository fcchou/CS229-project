from __future__ import division
import numpy as np
from glob import glob
import os.path
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
import sklearn.svm
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfTransformer

import sys
sys.path.append('../')
from jos_learn.features import FeatureExtract
normalizer = TfidfTransformer()

extractor = FeatureExtract()
labels = extractor.labels
#features, feature_names = extractor.get_ps_ql_pair()
feature1, feature_names1 = extractor.feature_cp
feature2, feature_names2 = extractor.feature_ps_ql_pair
#features = np.hstack((feature1, feature2))[labels != 2]
features =feature1[labels != 2]
features = normalizer.fit_transform(features).toarray()
labels = labels[labels != 2]
print features.shape

labels, features = shuffle(labels, features)
#clf = LogisticRegression()
#clf = MultinomialNB()
#clf = sklearn.svm.SVC()
clf = sklearn.svm.LinearSVC(C=10, class_weight='auto')

from sklearn.metrics import accuracy_score, f1_score


accu = []
f1 = []
cv = cross_validation.StratifiedShuffleSplit(labels, 60)
for train, test in cv:
    clf.fit(features[train], labels[train])
    pred = clf.predict(features[test])
    accu.append(accuracy_score(labels[test], pred))
    f1.append(f1_score(labels[test], pred))
print np.average(accu), np.average(f1)

'''
scores = cross_validation.cross_val_score(
    clf, features, labels, "average_precision", cv=cv, n_jobs=-1
)
print np.average(scores)
'''
