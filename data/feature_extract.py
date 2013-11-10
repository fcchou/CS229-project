import numpy as np
from sklearn.feature_extraction import DictVectorizer
from collections import Counter
import os.path as path


class FeatureExtract(object):
    def __init__(self, save_data=False, works_list='AllWorks.txt'):
        'Init by getting all the works.'
        self.self_dir = path.abspath(path.dirname(__file__))
        self.save_data = save_data
        self.works = []
        self.npz_data = None
        self.vectorizer = DictVectorizer()
        with open(path.join(self.self_dir, works_list)) as f:
            for line in f:
                self.works.append(line.split('-')[0])
        if self.save_data:
            np.save(path.join(self.self_dir, 'features/all_works', self.works))

    def get_labels(self, secure_list='Josquin_secure.txt'):
        '''
        Decide the labels
        2 for unsecure Jos, 1 for secure Jos, 0 for others
        '''
        #TODO consider labeling different authors for multiclass assignment
        with open(path.join(self.self_dir, secure_list)) as f:
            secure_jos = set(f.read().splitlines())
        self.labels = []
        for work in self.works:
            label = 0
            if 'Jos' in work:
                if work in secure_jos:
                    label = 1
                else:
                    label = 2
            self.labels.append(label)
        if self.save_data:
            np.save(path.join(self.self_dir, 'features/labels', self.works))
        return self.labels

    def generate_npz_data(self):
        '''
        Preprocess each work and save as npz files for quick access later.
        This uses music21.
        '''
        import music21
        for work in self.works:
            music_file = path.join(self.self_dir, 'correct_data/%s.krn' % work)
            if not path.isfile(music_file):
                music_file = music_file[:-3] + 'xml'
            if not path.isfile(music_file):
                raise Exception("Cannot find score for %s" % music_file[:-4])
            score = music21.converter.parse(music_file)
            all_arr = []
            for part in score.parts:
                arr = []
                for note in part.flat:
                    if isinstance(note, music21.note.Note):
                        elem = (note.ps, note.quarterLength)
                    elif isinstance(note, music21.note.Rest):
                        elem = (0.0, note.quarterLength)
                    else:
                        continue
                    arr.append(elem)
                all_arr.append(np.array(arr))
            saved_name = path.join(self.self_dir, 'npz_data/%s' % work)
            np.savez(saved_name, *all_arr)

    def load_npz_data(self):
        '''
        Load cached numpy npz data for all notes' ps and quarterLength (ql).
        Each woek contains multiple parts as nested list, each part is a
        np.adarray. 1st column is ps (0 for rest) and second column is ql.
        '''
        self.npz_data = []
        for work in self.works:
            all_data = path.join(self.self_dir, 'npz_data/%s.npz' % work)
            # Each work contains many parts. Loop through each one.
            self.npz_data.append([all_data[i] for i in all_data.files])

    def get_ps_diff(self):
        '''
        Get the ps differnce histogram feature.
        '''
        if self.npz_data is None:
            self.load_npz_data()
        dict_list = []
        for arrs in self.npz_data:
            for arr in arrs:
                is_note = arr[:, 0] != 0
                ps_diff = arr[1:, 0] - arr[:-1, 0]
                # Get rid of rests
                ps_diff = ps_diff[np.logical_and(is_note[1:], is_note[:-1])]
                dict_list.append(Counter(ps_diff))
        feature = self.vectorizer.fit_transform(dict_list).toarray()
        if self.save_data:
            np.save(
                path.join(self.self_dir, 'features/feature_ps_diff'), feature)
        self.feature_ps_diff = feature
        return feature

    def get_ql_diff(self):
        '''
        Get the quarterLength differnce histogram feature (for notes only).
        '''
        if self.npz_data is None:
            self.load_npz_data()
        dict_list = []
        for arrs in self.npz_data:
            for arr in arrs:
                is_note = arr[:, 0] != 0
                ql_diff = arr[1:, 1] - arr[:-1, 1]
                # Get rid of rests
                ql_diff = ql_diff[np.logical_and(is_note[1:], is_note[:-1])]
                dict_list.append(Counter(ql_diff))
        feature = self.vectorizer.fit_transform(dict_list).toarray()
        if self.save_data:
            np.save(
                path.join(self.self_dir, 'features/feature_ql_diff'), feature)
        self.feature_ql_diff = feature
        return feature

    def get_ps_ql_pair(self):
        '''
        Get the ps-ql pair histogram feature.
        '''
        if self.npz_data is None:
            self.load_npz_data()
        dict_list = []
        for arrs in self.npz_data:
            for arr in arrs:
                pairs = Counter()
                for ps, ql in arr:
                    # Reduce the features by folding the ps into range 1-12.
                    if ps != 0:
                        ps = ps % 12 + 1
                    pairs[(ps, ql)] += 1
            dict_list.append(pairs)
        feature = self.vectorizer.fit_transform(dict_list).toarray()
        if self.save_data:
            np.save(
                path.join(self.self_dir, 'features/feature_ps_ql_pair'),
                feature)
        self.feature_ps_ql_pair = feature
        return feature

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Feature extraction from scores.')
    parser.add_argument(
        '-preprocess_npz', action='store_true',
        help="Preprocess and save as .npz files")
    args = parser.parse_args()

    extractor = FeatureExtract(save_data=True)
    if args.preprocess_npz:
        extractor.generate_npz_data()
    extractor.get_labels()
    extractor.get_ps_diff()
    extractor.get_ql_diff()
    extractor.get_ps_ql_pair()
