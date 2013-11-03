# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:         ckyfeature.py
# Purpose:      music21 feature extractors
#
# Authors:      FCC, YHK, HYY
#
# Copyright:    Copyright © 2013 the music21 Project
# License:      LGPL, see license.txt
#-------------------------------------------------------------------------------

'''cky music21 feature extractors.
'''



import unittest
import urllib
import re
import math
import music21
from music21.features import base as featuresModule
from music21 import text
from music21.musedata.base40 import pitchToBase40

from music21 import environment
_MOD = 'features/ckyfeature.py'
environLocal = environment.Environment(_MOD)


#-------------------------------------------------------------------------------
class CKYFeatureException(featuresModule.FeatureException):
    pass

#-------------------------------------------------------------------------------
class DurationWeightedPitchHistogram(featuresModule.FeatureExtractor):
    '''    '''
    id = 'cky0'
    def __init__(self, dataOrStream=None, *arguments, **keywords):
        featuresModule.FeatureExtractor.__init__(self, dataOrStream=dataOrStream,  *arguments, **keywords)

        self.name = 'Duration Weighted Pitch Histogram'
        self.description = 'Pitch histogram weighted by duration'
        self.isSequential = True
        self.dimensions = 129
        self.discrete = False
        self.normalize = False

    def _process(self):
        '''Do processing necessary, storing result in _feature.
        '''
        notes = self.data['flat.notes']
        for note in notes:
            if isinstance(note, music21.note.Note):
                self._feature.vector[note.midi] += note.quarterLength
            elif isinstance(note, music21.note.Rest):
                self._feature.vector[-1] += note.quarterLength
            elif isinstance(note, music21.chord.Chord):
                for pitch in note.pitches:
                    self._feature.vector[pitch.midi] += note.quarterLength 
            else:
                continue

class CounterpointTuple(featuresModule.FeatureExtractor):
    '''    '''
    id = 'cky1'
    def __init__(self, dataOrStream=None, *arguments, **keywords):
        featuresModule.FeatureExtractor.__init__(self, dataOrStream=dataOrStream,  *arguments, **keywords)

        self.name = 'Counter point '
        self.description = 'Pitch histogram weighted by duration'
        self.isSequential = False
        self.dimensions = 1
        self.discrete = False
        self.normalize = False

    def _process(self):
        '''Do processing necessary, storing result in _feature.
        '''
        if self.data.partsCount > 0:
            print self.data.stream
            # Too slow...            
            #VLQs = music21.theoryAnalysis.theoryAnalyzer.getVLQs(self.data.stream,0,self.data.partsCount-1)
        else:
            pass
        self._feature.vector[0] = VLQs[0]
            
#-------------------------------------------------------------------------------


featureExtractors = [
DurationWeightedPitchHistogram
]







if __name__ == "__main__":
    import music21

#------------------------------------------------------------------------------
# eof




