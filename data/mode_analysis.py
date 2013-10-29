import numpy as np
from glob import glob

for name in glob("correct_data/*.npz"):
    data = np.load(name)
    all_pitch = np.zeros(12)
    last_note_ps = []
    for arr_name in data.files:
        for note in data[arr_name]:
            if note[0] == 0:
                continue
            all_pitch[note[0] % 12] += note[1]
        last_note_ps.append(note[0] % 12)
    np.array(last_note_ps)
    ps_sort = np.argsort(all_pitch)
    diff = ps_sort[-1] - ps_sort[-2]
    if diff < 0:
        diff += 12
    #if diff != 5 and diff != 7:
        #print name, diff
    print name
    print np.array(last_note_ps) - ps_sort[-1]
