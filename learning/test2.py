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

# Get the secure Jos data
jos_sec_set = set()
with open('../data/Josquin_secure.txt') as f:
    for line in f:
        jos_sec_set.add(line.strip())

test = []
test_label = []
test_name = []
train = []
train_label = []
train_name = []

test_ratio = 0.2
for name in glob('../data/correct_data/*.npz'):
    arr = get_feature(name)
    name = os.path.basename(name)
    if name[:3] != 'Jos':  # Non-Jos composer
        label = -1
    elif name[:-4] in jos_sec_set:  # Secure Jos
        label = 1
    else:
        continue

    if np.random.rand() < test_ratio:
        test.append(arr)
        test_label.append(label)
        test_name.append(name)
    else:
        train.append(arr)
        train_label.append(label)
        train_name.append(name)


test = np.array(test)
train = np.array(train)
test_label = np.array(test_label)
train_label = np.array(train_label)

#clf = LogisticRegression()
clf = MultinomialNB()
#clf = sklearn.svm.SVC()
#clf = sklearn.svm.LinearSVC()
clf.fit(train, train_label)
pred_train = clf.predict(train)
pred_test = clf.predict(test)

def test_error(pred, true_class):
    n_correct = np.sum(pred == true_class)
    return 1 - n_correct / pred.shape[0]


def get_all_classification(pred, true_class):
    total = pred.shape[0]
    true_pos = np.sum((pred == 1) * (true_class == 1))
    true_neg = np.sum((pred == -1) * (true_class == -1))
    false_neg = np.sum((pred == -1) * (true_class == 1))
    false_pos = np.sum((pred == 1) * (true_class == -1))
    print true_pos, true_neg, false_neg, false_pos, total

print "Training error:", test_error(pred_train, train_label)
print "Testing error:", test_error(pred_test, test_label)
get_all_classification(pred_test, test_label)


'''
for i, j, k in zip(test_name, test_label, pred_test):
    if j == 1:
        j = 'T'
    else:
        j = 'F'
    if k == 1:
        k = 'T'
    else:
        k = 'F'
    print "%20s %s %s" % (i, j, k)'''
