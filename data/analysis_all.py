import music21
import glob
import os.path

folders = ['xml']

all_files = []
for folder in folders:
    all_files += glob.glob(folder + '/*.xml')
    all_files += glob.glob(folder + '/*.krn')

note_ps = set()
note_ql = set()
rest_ql = set()

for filename in all_files:
    print "Parsing %s" % filename
    pkl_file = filename[:-3] + 'pkl'
    try:
        if os.path.exists(pkl_file):
            score = music21.converter.thaw(pkl_file)
        else:
            score = music21.converter.parse(filename)
            music21.converter.freeze(score, fp=pkl_file)
    except Exception:
        print "ERROR in parsing", filename
        continue
    for note in score.flat.getElementsByClass(music21.note.Note):
        note_ps.add(note.ps)
        note_ql.add(note.quarterLength)
    for note in score.flat.getElementsByClass(music21.note.Rest):
        rest_ql.add(note.quarterLength)

print sorted(note_ps)
print sorted(note_ql)
print sorted(rest_ql)
