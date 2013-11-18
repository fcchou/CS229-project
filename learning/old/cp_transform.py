import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from collections import Counter

import sys
sys.path.append('../')
from jos_learn.features import FeatureExtract

score_dict = pickle.load(open('counterpoint.p'))

def cp_convert(cp_tuple):
    a, b, c = cp_tuple
    if a < 0 or (a == 0 and c < 0):
        d = a + b + c
        a, b, c = -a, d, -c
    if a >= 7:
        a1 = a % 7
        c += a1 - a
        a = a1
    return (a, b, c)

dict_list = []
extractor = FeatureExtract()
for work in extractor.works:
    new_dict = Counter()
    for key, val in score_dict[work].iteritems():
        key = cp_convert(key)
        new_dict[key] += val
    dict_list.append(new_dict)

vectorizer = DictVectorizer()
feature = vectorizer.fit_transform(dict_list).toarray()
names = vectorizer.get_feature_names()
fname_feature = 'features/feature_cp'
fname_names = 'features/names_cp.pkl'
np.save(fname_feature, feature)
pickle.dump(names, open(fname_names, 'wb'))
