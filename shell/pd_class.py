from __future__ import division
import numpy as np
from glob import glob
import os.path

def get_pitch_class_from_notes(p0, p1):
    # determine the pitch change class for two consecutive notes
    p_change = p1 - p0
    if p_change < -4:
        return -2
    elif p_change < 0:
        return -1
    elif p_change == 0:
        return 0
    elif p_change < 5:
        return 1
    else:
        return 2

def get_duration_class_from_notes(d0, d1):
    # determine the duration change class for two consecutive notes
    if d1 < d0:
        return -1
    elif d1 == d0:
        return 0
    else:
        return 1

def get_pd_class(p, d):
    # determine the pitch/duration change class
    return 3*(p+2) + (d+2)

for name in glob('../data/correct_data/*.npz'):
    npz = np.load(name)
    combined_arr = np.array([[0,0]])
    for arr_name in npz.files:
        arr = npz[arr_name]
        combined_arr = np.vstack((combined_arr,arr))
        combined_arr = np.vstack((combined_arr,[[0,0]]))

    n_move = combined_arr.shape[0] - 1
    pd_class_arr = np.zeros(n_move, dtype=int)
    for i_note in range(n_move):
        if combined_arr[i_note,0] and combined_arr[i_note+1,0] != 0:
            p_class = get_pitch_class_from_notes(combined_arr[i_note,0], combined_arr[i_note+1,0])
            d_class = get_duration_class_from_notes(combined_arr[i_note,1], combined_arr[i_note+1,1])
            pd_class_arr[i_note] = get_pd_class(p_class, d_class)
    pd_class_arr = np.delete(pd_class_arr, 0)
    pd_class_arr = np.delete(pd_class_arr, -1)

    outdir = './pd_class_arrs/'
    filename = os.path.basename(name)
    fulloutpath = outdir + filename[:-4]
    np.save(fulloutpath, pd_class_arr)

