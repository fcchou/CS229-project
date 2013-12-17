import sys
sys.path.append('../')
from jos_learn.features import FeatureExtract

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

feature1, cp_name = extractor.feature_cp
#print cp_name
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

# remove sampoles with zero conterpoints
sum_feature1 = feature1.sum(1)
print works[sum_feature1==0]


feature1 = feature1[sum_feature1 != 0]
feature2 = feature2[sum_feature1 != 0]
labels = labels[sum_feature1 != 0]

SVD1 = TruncatedSVD(n_components=300)
SVD2 = TruncatedSVD(n_components=200)
#SVD1 = PCA(n_components=300)
#SVD2 = PCA(n_components=200)
feature1 = SVD1.fit_transform(feature1)
feature2 = SVD2.fit_transform(feature2)

feature = np.hstack((feature1, feature2))



'''
PCA1 = PCA(n_components=400)
feature = PCA1.fit_transform(feature)
'''

feature_unsec = feature[labels == 2]
feature = feature[labels != 2]
labels = labels[labels != 2]
#feature = feature1
#feature2 = Sel.fit_transform(feature2, labels)


#Sel = SelectKBest(k=2)
#feature = Sel.fit_transform(feature, labels)

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
#print validate(SVC(kernel='poly', degree=2, C=1000000), feature, 500)
'''
print validate(SVC(C=10000, gamma=0.75), feature, 500)
print validate(LinearSVC(C=100), feature, 500)
print validate(LogisticRegression(C=100), feature, 500)
'''

#clf = SVC(C=10000, gamma=0.75, probability=True)
#clf.fit(feature, labels)
#prob = clf.predict_proba(feature)
#np.set_printoptions(precision=3)
#np.set_printoptions(suppress=True)
#print prob * 100

#clf = SVC(C=10000, gamma=0.75, probability=True)
#clf.fit(feature, labels)
#prob1 = clf.predict_proba(feature_unsec)[:, 1]

#print validate(LogisticRegression(C=100), feature, 100)
clf = LogisticRegression(C=100)
sfk = cv.StratifiedShuffleSplit(labels, 1,test_size=0.5)
for train_index, test_index in sfk:
    clf.fit(feature[train_index], labels[train_index])

    normal_vector = clf.coef_[0]
    intercept = np.zeros(normal_vector.shape)
    intercept[0] = -clf.intercept_[0]/normal_vector[0]
    decision_vector = np.zeros(normal_vector.shape)
    decision_vector[0], decision_vector[1] = -normal_vector[1], normal_vector[0]

    feature_projection = []
    for feat in feature[test_index,:]:
        feat1 = feat - intercept
        x2 = feat1.dot(normal_vector)
        x1 = feat1.dot(decision_vector)
        feature_projection.append([x1,x2])

    feature_projection = np.array(feature_projection)

    #feature_projection1 = []
    #for feat in feature_projection:
    #    v = feat[0]*decision_vector[0:2] + feat[1]*normal_vector[0:2] + intercept[0:2]
    #    feature_projection1.append(v)

    #feature_projection = np.array(feature_projection1)

    test_label = labels[test_index]
    fig = plt.figure(figsize=(3, 3))
    plt.plot(feature_projection[test_label==0,1],
             feature_projection[test_label==0,0], '.',label='non-Josquin')
    plt.plot(feature_projection[test_label==1,1],
             feature_projection[test_label==1,0], '^',label='Josquin')

    plt.plot([0, 0], [np.min(feature_projection[:,0])-0.5,np.max(feature_projection[:,0])+0.5], 'k-')
#plt.xlim([-0.3,0.3])
#plt.ylim([-0.3,0.3])
    plt.legend()
    plt.xlabel('Dimension normal to decision boundary')
    plt.ylabel('Dimension parallel to decision boundary')
    for ax in fig.axes:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    plt.legend()
    plt.tight_layout()
    plt.show()


'''
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

plot_label = labels==0
color='r'
marker='.'

oness=-.3*np.ones(feature[plot_label ,important_features[0]].shape)
ax.scatter(feature[plot_label ,important_features[0]],
            feature[plot_label ,important_features[1]],
            oness,
            c=color,marker=marker)
ax.scatter(feature[plot_label ,important_features[0]],
            -oness,
            feature[plot_label ,important_features[2]],
            c=color,marker=marker)
ax.scatter(oness,
            feature[plot_label ,important_features[1]],
            feature[plot_label ,important_features[2]],
            c=color,marker=marker)

ax.scatter(feature[plot_label ,important_features[0]],
            feature[plot_label ,important_features[1]],
            feature[plot_label ,important_features[2]],
            c=color,marker=marker)

plot_label = labels==1
color='b'
marker='^'
oness=-.3*np.ones(feature[plot_label ,important_features[0]].shape)

ax.scatter(feature[plot_label ,important_features[0]],
            feature[plot_label ,important_features[1]],
            oness,
            c=color,marker=marker)
ax.scatter(feature[plot_label ,important_features[0]],
            -oness,
            feature[plot_label ,important_features[2]],
            c=color,marker=marker)
ax.scatter(oness,
            feature[plot_label ,important_features[1]],
            feature[plot_label ,important_features[2]],
            c=color,marker=marker)

ax.scatter(feature[plot_label ,important_features[0]],
            feature[plot_label ,important_features[1]],
            feature[plot_label ,important_features[2]],
            c=color,marker=marker)

ax.set_xlim3d(-.3, .3)
ax.set_ylim3d(-.3, .3)
ax.set_zlim3d(-.3, .3)

ax.set_xlabel('feature 1')
ax.set_ylabel('feature 2')
ax.set_zlabel('feature 3')

plt.show()
#print np.column_stack((prob1, prob2)) * 100
'''
