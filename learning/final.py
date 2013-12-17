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
#SVD1 = PCA(n_components=300)
#SVD2 = PCA(n_components=200)
feature1 = SVD1.fit_transform(feature1)
feature2 = SVD2.fit_transform(feature2)

feature = np.hstack((feature1, feature2))
feature_unsec = feature[labels == 2]
unsec_works = works[labels == 2]

feature = feature[labels != 2]
labels = labels[labels != 2]

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

#def validate(clf, feature, k):
#    sfk = cv.StratifiedShuffleSplit(labels, 400)
#    scores = []
#    for train, test in sfk:
#        score = []
#        train_set = feature[train]
#        test_set = feature[test]
#        clf.fit(train_set, labels[train])
#        pred = clf.predict(test_set)
#        score.append(accuracy_score(labels[test], pred))
#        score.append(precision_score(labels[test], pred))
#        score.append(recall_score(labels[test], pred))
#        score.append(f1_score(labels[test], pred))
#        scores.append(score)
#    avg = np.average(scores, axis=0)
#    return avg
##print validate(SVC(kernel='poly', degree=2, C=1000000), feature, 500)
#print validate(SVC(C=10000, gamma=0.75), feature, 500)
#print validate(LinearSVC(C=100), feature, 500)
#print validate(LogisticRegression(C=100), feature, 500)

#clf = SVC(C=10000, gamma=0.75, probability=True)
#clf.fit(feature, labels)
#prob = clf.predict_proba(feature)

clf = SVC(C=10000, gamma=0.75, probability=True)
clf.fit(feature, labels)
pred = clf.predict(feature_unsec)
print np.sum(pred), pred.shape[0] - np.sum(pred)
prob1 = clf.predict_proba(feature_unsec)[:, 1]

clf = LogisticRegression(C=100)
clf.fit(feature, labels)
pred = clf.predict(feature_unsec)
print np.sum(pred), pred.shape[0] - np.sum(pred)
prob2 = clf.predict_proba(feature_unsec)[:, 1]

plt.figure(figsize=(3, 2.25))
plt.plot(prob1, prob2, 'x')
plt.xlabel('P_{SVM}')
plt.ylabel('P_{Logistic}')
plt.tight_layout()
plt.savefig('P_unsec.pdf')
#plt.savefig('P_unsec.png', dpi=600)
#print np.column_stack((prob1, prob2)) * 100

#clf = LogisticRegression(C=100)
#clf.fit(feature, labels)
#prob = clf.predict_proba(feature_unsec)[:, 1]
#sort_idx = np.argsort(prob)[::-1]
#unsec_works = unsec_works[sort_idx]
#prob = prob[sort_idx]
#out = open('unsec_pred.txt', 'w')
#out.write('%9s %15s\n' % ('Probility', 'Work'))
#for p, w in zip(prob, unsec_works):
#    out.write('%9.2f %15s\n' % (p * 100, w))
#out.close()
