import music21
import glob
import os.path
import pickle


def reduce_score(score):
    ''' for counterpoint calculation.  Return a stream with all vertical pairs, indcuding rest. '''
    numberofparts = len(score.parts)
    if numberofparts == 1:
        return score
    
    # preare a new score
    newscore = music21.stream.Score()
    newparts = [music21.stream.Part() for i in range(numberofparts)]
    for i, newpart in enumerate(newparts):
        newpart.insert(0,score.parts[i].flat.getClefs()[0])
        newpart.insert(0,score.parts[i].flat.getTimeSignatures()[0])
    
    # check if all parts have the same length
    durations = [part.duration.quarterLength for part in score.parts]
    flag = False    
    for i in range(1,len(durations)):
        if score.parts[i-1].duration.quarterLength != score.parts[i].duration.quarterLength:
            #print "!!!!!Problem!!!!!"
            flag = True
            break

    #align all measures by setting the offset of each measure of parts equalto those of the first parts
    for part in range(1, len(score.parts)):
        for i in range(1, len(score.parts[part])):                        
            score.parts[part][i].offset = score.parts[0][i].offset
            
    # change the duration if measures have different duratoin in each part.
    # All the measures of other parts will have the same duration as those of the first part
    if flag:
        for i in range(1,len(score.parts[0])-1):
            #print 'measure', i 
            for j in range(len(score.parts)):
                ratio = 1.0*score.parts[0][i].duration.quarterLength/score.parts[j][i].duration.quarterLength                         
                if ratio != 1:
                    flag = True
                    score.parts[j][i].augmentOrDiminish(ratio, inPlace=True)
  
    
    flatsparts = [part.flat.getElementsByClass([music21.note.Note, music21.note.Rest]) for part in score.parts]
    
    # if a few notes have the same offset, keep the fist one and remove others
    for part in flatsparts:
        toremove = []
        for i in range(1,len(part)):     
            if part[i].offset == part[i-1].offset:
                toremove.append(part[i])
        part.remove(toremove)     
    
    # find the total duratoin of the score
    final_offset = score.parts[0].duration.quarterLength

    # find vertical slices between two parts
    index = [0]*numberofparts
    current_offset = 0
    next_offset = 0
    tobreak = False
    while not tobreak:   
        #print index
        tobreak = True
        current_notes = [flatsparts[i][index[i]] for i in range(numberofparts)]
        
        next_notes_offset = [0]*numberofparts
        for part in range(numberofparts):
            if index[part] < len(flatsparts[part])-1:
                next_notes_offset[part] = flatsparts[part][index[part]+1].offset   
            else:
                next_notes_offset[part] = final_offset 
        current_offset = next_offset   
        next_offset = min(next_notes_offset) 
        
        for part in range(numberofparts):
            toappend = None
            quarterLength = next_offset - current_offset
            
            if isinstance(current_notes[part], music21.note.Note):
                toappend = music21.note.Note(current_notes[part].nameWithOctave,
                                             quarterLength=quarterLength)
            elif isinstance(current_notes[part], music21.note.Rest):
                toappend = music21.note.Rest(quarterLength=quarterLength)
            elif isinstance(current_notes[part], music21.chord.Chord):
                toappend = music21.note.Note(current_notes[part].pitches[0].nameWithOctave,
                                             quarterLength=quarterLength)
            
            newparts[part].append(toappend)
            
            if next_notes_offset[part] == next_offset:
                index[part] += 1

            tobreak = tobreak and index[part] == len(flatsparts[part])
            
    for part in newparts:
        newscore.insert(0, part)
    return newscore
    

def generateVLQs(reduced_score, part1=0, part2=None):
    '''take a reduced_score, generate music21 VQL pairs'''
    top = reduced_score.parts[part1].notesAndRests
    if part2:
        bottom = reduced_score.parts[part2].notesAndRests
    else:
        bottom = top
    VLQs = []
    for i in range(len(top)-1):
        if top[i].isNote and top[i+1].isNote\
                and bottom[i].isNote and bottom[i+1].isNote:
            VLQ = music21.voiceLeading.VoiceLeadingQuartet(v1n1=top[i], 
                                                           v1n2=top[i+1],
                                                           v2n1=bottom[i], 
                                                           v2n2=bottom[i+1])
            VLQs.append(VLQ)
    return VLQs
    

def generate_counter_point_feature(reduced_score, part1=0, part2=None):
    VLQs = generateVLQs(reduced_score, part1, part2)
    result = {}
    for VLQ in VLQs:
        vlqtuple = (VLQ.vIntervals[0].generic.staffDistance,
                       VLQ.hIntervals[1].generic.staffDistance,
                       VLQ.vIntervals[1].generic.staffDistance)
        if vlqtuple not in result:
            result[vlqtuple] = 1 
        else:
            result[vlqtuple] += 1                      
    
    return result

def generate_all_counter_point_feature(score, 
                                       revome_stationary=False,
                                       folded=False):
                                           
    reduced_score = reduce_score(score)
    
    VLQfeature = {}    
    if len(reduced_score.parts) == 1:  
        return VLQfeature
    
    for i in xrange(len(reduced_score.parts)):
        for j in xrange(i+1,len(reduced_score.parts)):
            VLQs = generate_counter_point_feature(reduced_score, i, j)
            
            #update dictionary
            for key in VLQs:
                if key in VLQfeature:
                    VLQfeature[key] += VLQs[key]
                else:
                    VLQfeature[key] = VLQs[key]
    
    if revome_stationary:
        to_remove = []
        for key in VLQfeature:
            if key[1] == 0 and key[0] == key[2]:
                to_remove.append(key)
        
        for key in to_remove:
            del VLQfeature[key]  
        
    return VLQfeature
    
  
if __name__ == '__main__':    
    all_files = glob.glob('../data/music21/*.p')
    filebase = os.path.abspath('../data/music21')
    counterpoint = {}
    
    for filename in all_files:        
        score = music21.converter.thaw(fp=os.path.join(filebase, os.path.split(filename)[-1]))
        print 'extract', filename

        cp = generate_all_counter_point_feature(score, revome_stationary=True)
        counterpoint[os.path.split(filename)[-1][:-2]] = cp

    pickle.dump(counterpoint, open( "counterpoint.p", "wb" ))
    