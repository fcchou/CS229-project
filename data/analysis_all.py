import music21
import glob
import os.path
import numpy as np

all_files = glob.glob('correct_data/*.krn') + glob.glob('correct_data/*.xml')

all_comb = set()
for filename in all_files:
    print 'Parsing', filename
    try:
        score = music21.converter.parse(filename)
    except Exception as err:
        print "ERROR in parsing", filename, err
        continue

    all_data = []
    for part in score.parts:
        data = []
        for note in part.flat:
            if isinstance(note, music21.note.Note):
                elem = (note.ps, note.quarterLength)
            elif isinstance(note, music21.note.Rest):
                elem = (0.0, note.quarterLength)
            else:
                continue
            all_comb.add(elem)
            data.append(elem)
        all_data.append(np.array(data))
    np.savez(filename[:-4], *all_data)

np.save('all_combination', np.array(sorted(all_comb)))

