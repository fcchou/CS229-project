from __future__ import division
import numpy as np
from glob import glob
import os.path
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
import sklearn.svm
from sklearn.utils import shuffle

# Get the secure Jos data
jos_sec_set = set()
with open('../data/Josquin_secure.txt') as f:
    for line in f:
        jos_sec_set.add(line.strip())

features = []
labels = []
names = []

for name in glob('../data/correct_data/*feature1.npy'):
    arr = np.load(name)
    name = os.path.basename(name).split('_')[0]
    if name[:3] != 'Jos':  # Non-Jos composer
        label = 0
    elif name in jos_sec_set:  # Secure Jos
        label = 1
    else:
        continue
    labels.append(label)
    features.append(arr)
    names.append(name)

labels, features, names = shuffle(labels, features, names)
clf = LogisticRegression()
#clf = MultinomialNB()
#clf = sklearn.svm.SVC()
#clf = sklearn.svm.LinearSVC()

from sklearn.metrics import precision_score, accuracy_score


accu = []
prec = []
cv = cross_validation.StratifiedKFold(labels, n_folds=10)
for train, test in cv:
    clf.fit(features[train], labels[train])
    pred = clf.predict(features[test])
    accu.append(accuracy_score(labels[test], pred))
    prec.append(precision_score(labels[test], pred))
print np.average(accu), np.average(prec)

'''
scores = cross_validation.cross_val_score(
    clf, features, labels, "average_precision", cv=cv, n_jobs=-1
)
print np.average(scores)
'''
