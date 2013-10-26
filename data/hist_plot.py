import music21
import numpy as np
import matplotlib.pyplot as plt
import sys

pkl_file = sys.argv[1]
score = music21.converter.thaw(pkl_file)

note_ps = []
note_ql = []
for note in score.flat.getElementsByClass(music21.note.Note):
    note_ps.append(note.ps)
    note_ql.append(note.quarterLength)

note_ps = np.array(note_ps)
note_ql = np.array(note_ql)

be = np.arange(33.5, 81.5, 1)
hist, be = np.histogram(note_ps, be, weights=note_ql)
plt.bar(be[:-1] + 0.5, hist)
plt.show()
