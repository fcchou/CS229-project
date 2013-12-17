from __future__ import division
import numpy as np
from glob import glob
import os.path
from collections import Counter
import pickle

n = 2
outdir = './length2_ng2/'

def get_mirror(pd):
    if pd < 1:
        print 'error'
        return pd
    elif pd < 4:
        return pd+12
    elif pd < 7:
        return pd+6
    elif pd < 10:
        return pd
    elif pd < 13:
        return pd-6
    elif pd < 16:
        return pd-12
    else:
        print 'error'
        return pd

for name in glob('./pd_class_ng_arrs/*.npy'):
    pd_class_arr = np.load(name)
    n_move = pd_class_arr.shape[0]
    dict_new = Counter()

    for ii in range(n_move-n+1):
        skip = 0
        mirror = 0 # 0:pending, -1:no flip, 1:flip
        for jj in range(n):
           if pd_class_arr[ii+jj,0] == 0:
               skip = 1

        if skip == 0:
            i_feature = np.zeros([n,2])
            if mirror < 1:
                for jj in range(n):
                    i_feature[jj] = pd_class_arr[ii+jj]
            else:
                for jj in range(n):
                    #i_feature[jj] = get_mirror(pd_class_arr[ii+jj])
                    i_feature[jj] = pd_class_arr[ii+jj]
            dict_new[tuple(i_feature.flatten())] += 1
    filename = os.path.basename(name)
    fulloutpath = outdir + filename[:-4]
    pickle.dump(dict_new, open(fulloutpath+'.pkl', 'wb'))

