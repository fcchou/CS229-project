from __future__ import division
import glob
import numpy as np
from collections import OrderedDict
import itertools

all_files = glob.glob('correct_data/*.npz')

# Frist scan
names = []
all_arrs = []
all_ps_diff = set()
note_ql = set()
rest_ql = set()
all_ql_diff = set()
for filename in all_files:
    all_arr = []
    data =np.load(filename)
    names.append(filename)
    for arr_name in data.files:
        arr = data[arr_name]
        all_arr.append(arr)
        for i, note in enumerate(arr):
            ps, ql = note[0], note[1]
            if ps != 0:
                note_ql.add(ql)
                if i != 0:
                    ps0, ql0 = arr[i - 1][0], arr[i - 1][1]
                    if ps0 != 0:
                        all_ps_diff.add(ps - ps0)
                        all_ql_diff.add(ql - ql0)
            else:
                rest_ql.add(ql)
    all_arrs.append(all_arr)

# 2nd scan
all_ps_diff_dict = {}
all_ql_diff_dict = {}
for name, all_arr in itertools.izip(names, all_arrs):
    for k in all_ps_diff:
        all_ps_diff_dict[k] = 0
    for k in all_ql_diff:
        all_ql_diff_dict[k] = 0

    # Note histogram
    note_hist = np.zeros(12)
    bin_edges = np.arange(-0.5, 12.5, 1.0)
    for arr in all_arr:
        hist, be = np.histogram(
            arr[:, 0][arr[:, 0] != -1] % 12,
            bin_edges,
            weights=(arr[:, 1][arr[:, 0] != -1])
        )
        note_hist += hist
    #note_hist /= total_len  # Normalize

    # ql histogram
    note_ql_dict = OrderedDict()
    rest_ql_dict = OrderedDict()
    for i in note_ql:
        note_ql_dict[i] = 0
    for i in rest_ql:
        rest_ql_dict[i] = 0
    for arr in all_arr:
        for note in arr:
            if note[0] != 0:
                note_ql_dict[note[1]] += 1
            else:
                rest_ql_dict[note[1]] += 1

    note_ql_hist = np.array(note_ql_dict.keys())
    rest_ql_hist = np.array(rest_ql_dict.keys())

    # diff
    for arr in all_arr:
        for i, note in enumerate(arr):
            ps, ql = note[0], note[1]
            if i != 0 and ps != 0:
                ps0, ql0 = arr[i - 1][0], arr[i - 1][1]
                if ps0 != 0:
                    all_ps_diff_dict[ps - ps0] += 1
                    all_ql_diff_dict[ql - ql0] += 1
    features = np.hstack((
        all_ps_diff_dict.values(), all_ql_diff_dict.values()))
    # np.save(name.replace(".npz", "_feature1"), features)
    np.save(name.replace(".npz", "_feature1"), features)
