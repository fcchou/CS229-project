import sys
sys.path.append('../')
from jos_learn.features import FeatureExtract

import numpy as np
import matplotlib.pyplot as plt
import sklearn.cross_validation as cv
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer


# Setup the features
extractor = FeatureExtract()
labels = extractor.labels
feature1, feature_name1 = extractor.feature_cp
feature2, feature_name2 = extractor.feature_ps_ql_pair

normalizer = TfidfTransformer()
feature1_norm = normalizer.fit_transform(feature1).toarray()
feature2_norm = normalizer.fit_transform(feature2).toarray()

feature1_unsecure = feature1[labels == 2]
feature2_unsecure = feature2[labels == 2]
feature1_unsecure_norm = feature1_norm[labels == 2]
feature2_unsecure_norm = feature2_norm[labels == 2]

feature1 = feature1[labels != 2]
feature2 = feature2[labels != 2]
feature1_norm = feature1_norm[labels != 2]
feature2_norm = feature2_norm[labels != 2]
labels = labels[labels != 2]

# Grid search for parameters
#sfk = cv.StratifiedShuffleSplit(labels, 40)
#params = {'alpha':[0.1, 0.3, 0.5, 0.8, 1, 2]}
#gs = GridSearchCV(
#        MultinomialNB(), params, scoring='f1',
#        n_jobs=4, cv=sfk)
#gs.fit(feature2, labels)
#print gs.best_score_, gs.best_params_

C_lsvc = 10

C_logi_1 = 100
C_logi_2 = 10000

svm = LinearSVC(C=C_lsvc, class_weight='auto')
logi1 = LogisticRegression(C=C_logi_1)
logi2 = LogisticRegression(C=C_logi_2)
nb = MultinomialNB()

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
def unsecure_test(clf, test, train, tag):
    clf.fit(train, labels)
    pred = clf.predict(test)
    print tag, np.sum(pred), test.shape[0] - sum(pred)
    return pred

all_pred = []
all_pred.append(unsecure_test(nb, feature2_unsecure, feature2, 'NB_pd'))
all_pred.append(unsecure_test(logi2, feature2_unsecure_norm, feature2_norm, 'Logi_pd'))
all_pred.append(unsecure_test(svm, feature2_unsecure_norm, feature2_norm, 'SVM_pd'))
all_pred.append(unsecure_test(nb, feature1_unsecure, feature1, 'NB_cp'))
all_pred.append(unsecure_test(logi1, feature1_unsecure_norm, feature1_norm, 'Logi_cp'))
all_pred.append(unsecure_test(svm, feature1_unsecure_norm, feature1_norm, 'SVM_cp'))

intersect = all_pred[-1]
for pred in all_pred[3:]:
    intersect *= pred
print np.sum(intersect), intersect.shape[0] - np.sum(intersect)
