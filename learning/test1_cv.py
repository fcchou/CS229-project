from __future__ import division
import numpy as np
import sklearn.svm
from glob import glob
import os.path
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.utils import shuffle
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import PCA
from sklearn.lda import LDA
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer


# Curate the dataset
all_comb_arr = np.load('../data/all_combination.npy')
all_comb = {}
for arr in all_comb_arr:
    all_comb[(arr[0], arr[1])] = 0


def get_feature1(npz_file):
    npz = np.load(npz_file)
    for key in all_comb:
        all_comb[key] = 0
    for arr_name in npz.files:
        arr = npz[arr_name]
        for idx in arr:
            all_comb[tuple(idx)] += 1
    return np.array(all_comb.values(), dtype=int)
    

def get_feature2(npz_file):
    npz = np.load(npz_file)
    for key in all_comb:
        all_comb[key] = 0
    for arr_name in npz.files:
        arr = npz[arr_name]
        for idx in arr:
            if idx[0] == 0:
                tup = (-1.0, idx[1])
            else:
                tup = (idx[0] % 12, idx[1])
            all_comb[tup] += 1
    return np.array(all_comb.values(), dtype=int)

# Get the secure Jos data
jos_sec_set = set()
with open('../data/Josquin_secure.txt') as f:
    for line in f:
        jos_sec_set.add(line.strip())
        
others_set = set()
with open('../data/others.txt') as f:
    for line in f:
        others_set.add(line.split('-')[0])

song_feature = []
song_label = []
song_name = []
in_training_set = set()
for name in glob('../data/correct_data/*.npz'):
    arr = get_feature2(name)
    name = os.path.basename(name)[:-4]
    if name[:3] != 'Jos':  # Non-Jos composer
        label = 0
    elif name in jos_sec_set:  # Secure Jos
        label = 1
    else:
        continue

    if name in jos_sec_set or name in others_set:
        if True:# name[:7] not in in_training_set:
            in_training_set.add(name[:7])
            song_feature.append(arr)
            song_label.append(label)
            song_name.append(name)
    
song_feature = np.array(song_feature, dtype=float)
song_label = np.array(song_label)
song_name = np.array(song_name)
        
#train = train/np.sum(train, axis=1).reshape(len(train),1)
#song_feature = preprocessing.normalize(song_feature)

#clf = LogisticRegression(C=10, class_weight='auto')
#clf = MultinomialNB()
#clf = sklearn.svm.LinearSVC(C=130, class_weight='auto')


#estimators = [('normalizer', preprocessing.Normalizer(norm='l1')), ('svm', sklearn.svm.LinearSVC(C=130, class_weight='auto'))]
#clf = Pipeline(estimators)

estimators = [('normalizer', TfidfTransformer(norm='l1')), ('svm', sklearn.svm.LinearSVC(C=130, class_weight='auto'))]
clf = Pipeline(estimators)

#clf = DummyClassifier(strategy='most_frequent',random_state=0)
#clf = DummyClassifier(strategy='stratified',random_state=0)
#clf = DummyClassifier(strategy='uniform',random_state=0)
#clf = LDA(n_components=3)


## Cross_validation
#scores = cross_validation.cross_val_score(clf, X, y,
#                                          scoring='f1', cv=10, 
#                                          verbose=1 ,n_jobs=1)
#score_mean = np.mean(scores)
#score_std = np.std(scores)
#
#print 'f1', score_mean, score_std


## PCA/LDA visualization
#pca = PCA(n_components=2)
#X_r = pca.fit(X).transform(X)

#lda = LDA(n_components=3)
#X_r = lda.fit(X, y).transform(X)
#
## Percentage of variance explained for each components
#print('explained variance ratio (first two components): %s'
#      % str(pca.explained_variance_ratio_))
#
#plt.figure()
#for c, i, target_name in zip("rb", [0, 1], ['non-jos', 'jos']):
#    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=target_name)
#plt.legend()
#plt.title('PCA of IRIS dataset')
#
#plt.show()


##backward feature reduction
#rfecv = RFECV(estimator=clf, step=10, cv=cross_validation.StratifiedShuffleSplit(y, test_size=0.1),
#              scoring='f1', verbose=1)
#rfecv.fit(X, y)
#
#print("Optimal number of features : %d" % rfecv.n_features_)
#
## Plot number of features VS. cross-validation scores
#import pylab as pl
#pl.figure()
#pl.xlabel("Number of features selected")
#pl.ylabel("Cross validation score (nb of misclassifications)")
#pl.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
#pl.show()


train_size = range(10,int(len(song_label)*0.9*0.9),5)

train_errors = np.zeros((10,len(train_size)))
test_errors = np.zeros((10,len(train_size)))

##Learning Curve###
print "Learning Curve"
k_fold = cross_validation.StratifiedShuffleSplit(song_label, test_size=0.1)
for k, (train, test) in enumerate(k_fold):
    print  "round" ,k
    for i, p in enumerate(train_size): 
        #make sure  we have the same ratio between two classes
        train_index, test_index = next(iter(cross_validation.StratifiedShuffleSplit(song_label[train],train_size=p)))
        #print len(train_index)
        #fit data
        clf.fit(song_feature[train[train_index]], song_label[train[train_index]])
        #get error
        train_predict_label = clf.predict(song_feature[train[train_index]])
        test_predict_label = clf.predict(song_feature[test])
        train_errors[k,i] = sklearn.metrics.f1_score(song_label[train[train_index]], train_predict_label)
        test_errors[k,i]  = sklearn.metrics.f1_score(song_label[test], test_predict_label)

train_errors_mean = np.mean(train_errors,axis=0)
test_errors_mean = np.mean(test_errors,axis=0)

plt.figure()
plt.plot(train_size, train_errors_mean, 'b.', label='train_error', hold=True) 
plt.plot(train_size, test_errors_mean, 'r*',label='test_error') 
plt.legend()
plt.title('PCA of IRIS dataset')

plt.show()

        