from __future__ import division
import numpy as np
import sklearn.svm
from glob import glob
import os.path
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

# Curate the dataset
all_comb_arr = np.load('../data/all_comb_mod12.npy')
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
            if idx[0] == 0:
                tup = (-1.0, idx[1])
            else:
                tup = (idx[0] % 12, idx[1])
            all_comb[tup] += 1
    return np.array(all_comb.values(), dtype=int)

# Curate the dataset
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

train = []
train_label = []
train_name = []

for name in glob('../data/correct_data/*.npz'):
    arr = get_feature(name)
    name = os.path.basename(name)
    if name[:3] != 'Jos':  # Non-Jos composer
        label = -1
    elif name[:-4] in jos_sec_set:  # Secure Jos
        label = 1
    else:
        continue

    train.append(arr)
    train_label.append(label)
    train_name.append(name)

X, y, y_name = shuffle(train, train_label, train_name , random_state=1)

#clf = LogisticRegression()
#clf = MultinomialNB()
#clf = sklearn.svm.SVC()
clf = sklearn.svm.LinearSVC()

scores = cross_validation.cross_val_score(clf, X, y,
                                          scoring='precision', cv=10, 
                                          verbose=1 ,n_jobs=1)
score_mean = np.mean(scores)
score_std = np.std(scores)

print 'precision_score', score_mean, score_std
