from __future__ import division
import numpy as np
import sklearn.svm
from glob import glob
import os.path
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.dummy import DummyClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn import preprocessing


# Create the dataset
all_comb_arr = np.load('../data/all_combination.npy')
all_comb = {}
for arr in all_comb_arr:
    all_comb[(arr[0], arr[1])] = 0


def get_feature(npz_file):
    npz = np.load(npz_file)
    for key in all_comb:
        all_comb[key] = 0
    for arr_name in npz.files:
        arr = npz[arr_name]
        for idx in arr:
            all_comb[tuple(idx)] += 1
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

for name in glob('../data/correct_data/*.npz'):
    arr = get_feature(name)
    name = os.path.basename(name)[:-4]
    if name[:3] != 'Jos':  # Non-Jos composer
        label = -1
    elif name in jos_sec_set:  # Secure Jos
        label = 1
    else:
        continue
    #print arr
    if name in jos_sec_set or name in others_set:
        song_feature.append(arr)
        song_label.append(label)
        song_name.append(name)

song_feature = np.array(song_feature)
song_label = np.array(song_label)
song_name = np.array(song_name)

#song_feature = song_feature/np.sum(song_feature, axis=1).reshape(len(song_feature),1)

 
# Set the parameters by cross-validation

#clf = LogisticRegression(C=1, class_weight='auto')
#tuned_parameters = [{'C': np.logspace(1,3,10)}]


clf = MultinomialNB()
tuned_parameters = [{'alpha':[ 0.5, 1, 1.5, 2]}]

#clf = sklearn.svm.SVC()
#tuned_parameters = [{'kernel': ['poly'], 'C': [1], 
#                     'gamma':np.logspace(-5,0,10), 'degree':[3], 'class_weight':['auto']}]


#clf = sklearn.svm.LinearSVC()
#tuned_parameters = [{'C': np.logspace(2,3,10), 'class_weight':['auto']}] 

#clf = DummyClassifier(strategy='most_frequent')
#clf = DummyClassifier(strategy='stratified')
#clf = DummyClassifier(strategy='uniform')
#tuned_parameters = [{}]

# k-fold grid search
k_fold = cross_validation.StratifiedShuffleSplit(song_label, test_size=0.1)
for k, (train, test) in enumerate(k_fold):
    print("# Tuning hyper-parameters for %s" % 'f1')
    
    clf_best = GridSearchCV(clf, tuned_parameters, cv=10, scoring='f1', verbose=1)
    clf_best.fit(song_feature[train], song_label[train])
    
    print "Best parameters set found on development set:"
    print clf_best.best_params_
    #print clf_best.best_score_
#    print "Grid scores on development set:"
#    print
#    for params, mean_score, scores in clf_best.grid_scores_:
#        print("%0.3f (+/-%0.03f) for %r"
#              % (mean_score, scores.std(), params))
    
    
#    print("Detailed classification report:")
#    print("The model is trained on the full development set.")
#    print("The scores are computed on the full evaluation set.")
#    y_true, y_pred = song_label[test], clf_best.predict(song_feature[test])
#    print(classification_report(y_true, y_pred))
    print clf_best.score(song_feature[test], song_label[test])