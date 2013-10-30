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

from music21.features import base as featuresModule
from music21 import text

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
        self.dimensions = 128
        self.discrete = False
        self.normalize = False

    def _process(self):
        '''Do processing necessary, storing result in _feature.
        '''
        notes = self.data['flat.notes']
        for note in notes:
            self._feature.vector[note.midi] += note.quarterLength
#-------------------------------------------------------------------------------


featureExtractors = [
DurationWeightedPitchHistogram
]







if __name__ == "__main__":
    import music21

#------------------------------------------------------------------------------
# eof




