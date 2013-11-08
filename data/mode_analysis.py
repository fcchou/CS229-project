import numpy as np
from glob import glob

for name in glob("correct_data/*.npz"):
    data = np.load(name)
    all_pitch = np.zeros(12)
    last_note_ps = []
    for arr_name in data.files:
        ps = data[arr_name][-1][0] % 12
        if ps == 0.0:
            continue
        last_note_ps.append(ps)
    print last_note_ps
