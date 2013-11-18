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
import cPickle as pickle

# Setup the features
extractor = FeatureExtract()
labels = extractor.labels
works = extractor.works

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

folder = '../shell/length3'
dict_list = []
for work in works:
    data = pickle.load(open('%s/%s.pkl' % (folder, work), 'rb'))
    dict_list.append(data)
all_feature, names = extractor._vectorize(dict_list)

folder = '../shell/length2'
dict_list = []
for work in works:
    data = pickle.load(open('%s/%s.pkl' % (folder, work), 'rb'))
    dict_list.append(data)
all_feature1, names = extractor._vectorize(dict_list)

hist1 = np.sum(all_feature, axis=0)
hist1.sort()

plt.plot(hist1)
plt.show()
