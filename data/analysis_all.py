import music21
import glob

#folders = ['Jos_all', 'Ock-20131020-224312', 'Others']
folders = ['Others']

all_files = []
for folder in folders:
    all_files += glob.glob(folder + '/*.krn')

note_ps = set()
note_ql = set()
rest_ql = set()

for krn_file in all_files:
    print "Parsing", krn_file
    try:
        score = music21.converter.parse(krn_file)
    except Exception:
        print "ERROR in parsing", krn_file
        continue
    for note in score.flat.getElementsByClass(music21.note.Note):
        note_ps.add(note.ps)
        note_ql.add(note.quarterLength)
    for note in score.flat.getElementsByClass(music21.note.Rest):
        rest_ql.add(note.quarterLength)

print sorted(note_ps)
print sorted(note_ql)
print sorted(rest_ql)
