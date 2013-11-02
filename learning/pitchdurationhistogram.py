import music21
import glob
import os.path
import numpy as np
import ckyfeature


all_files = glob.glob('../data/music21/*.p')

filebase = os.path.abspath('../data/music21')
for filename in all_files[71:-1]:
    print 'extract', filename
    score = music21.converter.thaw(fp=os.path.join(filebase, os.path.split(filename)[-1]))
    fe = ckyfeature.DurationWeightedPitchHistogram(score)
    f = fe.extract()
    np.savez(os.path.join('pitchdurationhistogram', os.path.split(filename)[-1][:-2]), 
             f)


