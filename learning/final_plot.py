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
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, auc
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

#def validate(clf, feature, k):
#    sfk = cv.StratifiedShuffleSplit(labels, 100)
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

sfk = cv.StratifiedShuffleSplit(labels, 500)
clf1 = SVC(C=10000, gamma=0.75, probability=True)
all_prob = []
all_label = []
for train, test in sfk:
    clf1.fit(feature[train], labels[train])
    prob = clf1.predict_proba(feature[test])[:, 1]
    all_prob.append(prob)
    all_label.append(labels[test])
all_label = np.hstack(all_label)
all_prob = np.hstack(all_prob)

precision, recall, thresholds = precision_recall_curve(all_label, all_prob)
area = auc(recall, precision)
F1 = (precision * recall) / (precision + recall) * 2

np.savetxt('pr_svm.txt',np.column_stack((precision[:-1], recall[:-1], thresholds)), fmt='%4f')

#plt.figure(figsize=(3, 2.25))
#plt.plot(recall, precision)
#plt.xlabel('Recall')
#plt.ylabel('Precision')
#plt.title('Precision-Recall, SVM, AUC=%0.2f' % area)
#plt.xlim(0, 1)
#plt.ylim(0, 1.05)
#plt.tight_layout()

plt.figure(figsize=(3, 2.25))
plt.plot(thresholds, precision[:-1], label='Precision')
plt.plot(thresholds, recall[:-1], label='Recall')
plt.plot(thresholds, F1[:-1], label='F1')
plt.title('SVM')
plt.xlabel('Threshold')
plt.legend(loc=3, fontsize=8)
plt.tight_layout()

plt.savefig('prob_pr_svm.png', dpi=600)
plt.savefig('prob_pr_svm.pdf')

sfk = cv.StratifiedShuffleSplit(labels, 500)
clf = LogisticRegression(C=100)
all_prob = []
all_label = []
for train, test in sfk:
    clf.fit(feature[train], labels[train])
    prob = clf.predict_proba(feature[test])[:, 1]
    all_prob.append(prob)
    all_label.append(labels[test])
all_label = np.hstack(all_label)
all_prob = np.hstack(all_prob)

precision, recall, thresholds = precision_recall_curve(all_label, all_prob)
area = auc(recall, precision)
F1 = (precision * recall) / (precision + recall) * 2
#
#plt.figure(figsize=(3, 2.25))
#plt.plot(recall, precision)
#plt.xlabel('Recall')
#plt.ylabel('Precision')
#plt.title('Precision-Recall, Logistic, AUC=%0.2f' % area)
#plt.xlim(0, 1)
#plt.ylim(0, 1.05)
#plt.tight_layout()
#plt.savefig('pr_logi.png')
#plt.savefig('pr_logi.pdf')

plt.figure(figsize=(3, 2.25))
plt.plot(thresholds, precision[:-1], label='Precision')
plt.plot(thresholds, recall[:-1], label='Recall')
plt.plot(thresholds, F1[:-1], label='F1')
plt.xlabel('Threshold')
plt.title('Logistic')
plt.legend(loc=3, fontsize=8)
plt.tight_layout()

plt.savefig('prob_pr_logi.png', dpi=600)
plt.savefig('prob_pr_logi.pdf')


# Plot the learning curve

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
#n = 3
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
#        sfk = cv.StratifiedShuffleSplit(labels_curr, 500)
#        score = cv.cross_val_score(
#            clf, feature_curr, labels_curr, 'f1', sfk, n_jobs=8)
#        test_scores.append(np.average(score))
#        # training error
#        clf.fit(feature_curr, labels_curr)
#        pred = clf.predict(feature_curr)
#        train_scores.append(f1_score(labels_curr, pred))
#    return test_scores, train_scores
#
#fig = plt.figure(figsize=(6, 2.25))
#
#
#def plot(clf, feature, idx, title):
#    fig.add_subplot(1, 2, idx)
#    test_scores, train_scores = get_learning_curve(clf, feature)
#    plt.plot(data_size, test_scores, 'r-o')
#    plt.plot(data_size, train_scores, 'k-o')
#    plt.xlabel('Dataset size')
#    plt.ylabel('F1')
#    plt.title(title)
#    plt.tight_layout()
#    plt.ylim(0.8, 1.05)
#
#clf1 = LogisticRegression(C=100)
#clf2 = SVC(C=10000, gamma=0.75)
#plot(clf1, feature, 1, 'SVM')
#plot(clf2, feature, 2, 'Logistic')
#plt.show()
