import glob
import numpy as np

all_files = glob.glob('correct_data/*.npz')

all_comb = set()
for filename in all_files:
    data = np.load(filename)
    for arr_name in data.files:
        for note in data[arr_name]:
            if note[0] == 0:
                all_comb.add((-1.0, note[1]))
            else:
                all_comb.add((note[0] % 12, note[1]))

np.save('all_comb_mod12', np.array(sorted(all_comb)))
