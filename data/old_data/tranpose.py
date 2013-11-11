from __future__ import division
import glob
import numpy as np
from collections import OrderedDict
import itertools

all_files = glob.glob('correct_data/*.npz')

# Frist scan
names = []
all_arrs = []
keys = []
all_comb = set()
all_ps_diff = set()
note_ql = set()
rest_ql = set()

for filename in all_files:
    all_arr = []
    last_notes = []
    part_len = []
    data = np.load(filename)
    for arr_name in data.files:
        arr = data[arr_name]
        arr[:, 0][arr[:, 0] == 0] = -1  # Change rest to be -1
        all_arr.append(arr)
        if arr[-1, 0] != -1:
            last_notes.append(arr[-1, 0] % 12)
    if not last_notes:  # Check not empty
        raise Exception("Zero last_notes in %s" % filename)
    last_notes.sort()
    notes = []
    curr_note = -1
    count = 0
    while last_notes:
        note = last_notes.pop()
        if note != curr_note:
            if curr_note != -1:
                notes.append((count, curr_note))
            curr_note = note
            count = 1
        else:
            count += 1
    notes.append((count, curr_note))
    notes.sort(reverse=True)

    # Get the key of the music
    key = 0
    if len(notes) == 1 or notes[0][0] != notes[1][0]:
        key = notes[0][1]
    elif len(notes) == 2 or notes[1][0] != notes[2][0]:
        # Try to find the key by ps difference
        diff = (notes[0][1] - notes[1][1]) % 12
        if diff in (5.0, 8.0, 9.0):
            key = notes[0][1]
        elif diff in (3.0, 4.0, 7.0):
            key = notes[1][1]
        else:
            print filename, notes
            print("Diff not in range: %.2f" % diff)
    elif len(notes) == 3 or notes[2][0] != notes[3][0]:
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            diff = (notes[i][1] - notes[j][1]) % 12
            if diff == 5.0:
                key = notes[i][1]
                break
            elif diff == 7.0:
                key = notes[j][1]
                break
        else:
            print "Cannot find key in %s" % filename

    names.append(filename)
    keys.append(key)
    all_arrs.append(all_arr)

    for arr in all_arr:
        for i, note in enumerate(arr):
            ps, ql = note[0], note[1]
            if ps != -1:
                ps_shft = (ps - key) % 12
            else:
                ps_shft = ps
            all_comb.add((ps_shft, ql))
            if i != 0:
                ps0 = arr[i - 1][0]
                if ps0 != -1 and ps != -1:
                    all_ps_diff.add(ps - ps0)

# 2nd scan
all_comb_dict = {}
all_ps_diff_dict = {}
for name, key, all_arr in itertools.izip(names, keys, all_arrs):
    for k in all_comb:
        all_comb_dict[k] = 0
    for k in all_ps_diff:
        all_ps_diff_dict[k] = 0

    # Get the feature
    n_parts = len(all_arr)
    total_len = 0
    for arr in all_arr:
        total_len += np.sum(arr[:,1])
    avg_len = total_len / n_parts

    # Combination and diff
    for arr in all_arr:
        for i, note in enumerate(arr):
            ps, ql = note[0], note[1]
            if ps != -1:
                ps_shft = (ps - key) % 12
            else:
                ps_shft = ps
            all_comb_dict[(ps_shft, ql)] += 1
            if i != 0:
                ps0 = arr[i - 1][0]
                if ps0 != -1 and ps != -1:
                    all_ps_diff_dict[ps - ps0] += 1
    features = np.hstack((
        all_ps_diff_dict.values()))
    np.save(name.replace(".npz", "_feature1"), features)
    #np.save(name.replace(".npz", "_feature1"), note_hist)

"""
    # ql histogram
    note_ql_dict = OrderedDict()
    rest_ql_dict = OrderedDict()
    for i in note_ql:
        note_ql_dict[i] = 0
    for i in rest_ql:
        rest_ql_dict[i] = 0
    for arr in all_arr:
        for note in arr:
            if note[0] != -1:
                note_ql_dict[note[1]] += 1
            else:
                rest_ql_dict[note[1]] += 1
    note_ql_count = np.array(note_ql_dict.values())
    rest_ql_count = np.array(rest_ql_dict.values())
    avg_ql_note = (
        note_ql_count * np.array(note_ql_dict.keys())
        / np.sum(note_ql_count)
    )
    if np.sum(rest_ql_count) == 0:
        avg_ql_rest = 0
    else:
        avg_ql_rest = (
            rest_ql_count * np.array(rest_ql_dict.keys())
            / np.sum(rest_ql_count)
        )
    note_ql_hist = note_ql_count * np.array(note_ql_dict.keys()) / total_len
    rest_ql_hist = rest_ql_count * np.array(rest_ql_dict.keys()) / total_len

"""
