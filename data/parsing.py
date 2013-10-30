import music21
import glob
import os.path
import numpy as np

all_files = glob.glob('correct_data/*.krn') + glob.glob('correct_data/*.xml')

filebase = os.path.abspath('./correct_data/music21')


all_comb = set()
for filename in all_files:
    print 'Parsing', filename
    try:
        score = music21.converter.parse(filename)
    except Exception as err:
        print "ERROR in parsing", filename, err
        continue
    
    print music21.converter.freeze(score, fmt='pickle', 
                                   fp=os.path.join(filebase, os.path.split(filename)[-1][:-4]+'.p'))

	
