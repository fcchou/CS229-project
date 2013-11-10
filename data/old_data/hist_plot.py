import music21
import numpy as np
import matplotlib.pyplot as plt
import sys


filename = sys.argv[1]
score = music21.converter.parse(filename)

note_ps = []
note_ql = []
for note in score.flat.getElementsByClass(music21.note.Note):
    note_ps.append(note.ps % 12)
    note_ql.append(note.quarterLength)

note_ps = np.array(note_ps)
note_ql = np.array(note_ql)

be = np.arange(-0.5, 12.5, 1)
hist, be = np.histogram(note_ps, be, weights=note_ql)
plt.bar(be[:-1] + 0.5, hist)
plt.show()
