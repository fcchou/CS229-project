from __future__ import division
import numpy as np
from glob import glob
import os.path

n = 3
n_feature = 9*pow(15,n-1)
outdir = './length3/'

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

for name in glob('./pd_class_arrs/*.npy'):
    feature_arr = np.zeros(n_feature, dtype=int)
    pd_class_arr = np.load(name)
    n_move = pd_class_arr.shape[0]
    
    for ii in range(n_move-n+1):
        skip = 0
        mirror = 0 # 0:pending, -1:no flip, 1:flip
        for jj in range(n):
           if pd_class_arr[ii+jj] == 0:
               skip = 1
           if (skip == 0) and (mirror == 0):
               if (pd_class_arr[ii+jj] > 0) and (pd_class_arr[ii+jj] < 7):
                   mirror = -1
               elif pd_class_arr[ii+jj] > 9:
                   mirror = 1
        
        if skip == 0:
            i_feature = 0
            if mirror < 1:
                for jj in range(n):
                    i_feature = i_feature + pow(15, n-jj-1)*(pd_class_arr[ii+jj]-1)
            else:
                for jj in range(n):
                    i_feature = i_feature + pow(15, n-jj-1)*(get_mirror(pd_class_arr[ii+jj])-1)
            feature_arr[i_feature] = feature_arr[i_feature] + 1
            
    filename = os.path.basename(name)
    fulloutpath = outdir + filename[:-4]
    np.save(fulloutpath, pd_class_arr)
