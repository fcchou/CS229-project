import cPickle as pickle
score_dict = pickle.load(open('counterpoint.p'))
import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
import sklearn.svm
from sklearn.utils import shuffle


def cp_regularize1(cp):
    return cp
    #b, c = cp[1], cp[2] - cp[1]
    #return (b, c)

def cp_regularize2(cp):
    b, c = cp[1], cp[2] - cp[1]
    return (b, c)

def cp_regularize3(cp):
    if cp[0] >= 0:
        a = cp[0] % 7
    else:
        a = cp[0] % 7 - 7
    diff = cp[0] - a
    c = cp[2] - diff
    return (a, cp[1], c)

def cp_regularize(cp):
    return cp_regularize1(cp)

all_cp = {}
for score in score_dict.itervalues():
    for cp, count in score.iteritems():
        cp = cp_regularize(cp)
        if cp in all_cp:
            all_cp[cp] += count
        else:
            all_cp[cp] = count


# Get the secure Jos data
jos_sec_set = set()
with open('../data/Josquin_secure.txt') as f:
    for line in f:
        jos_sec_set.add(line.strip())

cp_dict = {}
for key in all_cp:
    cp_dict[key] = 0


features = []
labels = []
names = []
for name, score in score_dict.iteritems():
    if name[:3] != 'Jos':  # Non-Jos composer
        label = 0
    elif name in jos_sec_set:  # Secure Jos
        label = 1
    else:
        continue
    labels.append(label)
    names.append(name)
    for key in cp_dict:
        cp_dict[key] = 0

    for cp, count in score.iteritems():
        cp = cp_regularize(cp)
        cp_dict[cp] += count

    arr = np.load('../data/correct_data/%s_feature1.npy' % name)
    #arr = np.hstack((arr, cp_dict.values()))
    #arr = cp_dict.values()
    features.append(arr)

features = np.array(features)
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
features = transformer.fit_transform(features).toarray()

labels, features, names = shuffle(labels, features, names)
#clf = LogisticRegression()
#clf = MultinomialNB()
#clf = sklearn.svm.SVC(C=0.001, gamma =10000)
clf = sklearn.svm.LinearSVC(C=10)

from sklearn.metrics import precision_score, accuracy_score, recall_score


accu = []
prec = []
reca = []
cv = cross_validation.StratifiedKFold(labels, n_folds=10)
for train, test in cv:
    clf.fit(features[train], labels[train])
    pred = clf.predict(features[test])
    accu.append(accuracy_score(labels[test], pred))
    prec.append(precision_score(labels[test], pred))
    reca.append(recall_score(labels[test], pred))
print np.average(accu), np.average(prec), np.average(reca)
