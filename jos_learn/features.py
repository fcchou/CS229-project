import numpy as np
from sklearn.feature_extraction import DictVectorizer
from collections import Counter
import os
import os.path as path
import cPickle as pickle


class FeatureExtract(object):
    def __init__(self, save_data=True, use_saved_data=True):
        'Init by getting all the works.'
        self._self_dir = path.abspath(path.dirname(__file__))
        self.use_saved_data = use_saved_data
        self.save_data = save_data
        self.works = []
        self.npz_data = None
        self._vectorizer = DictVectorizer()

        if save_data and not path.isdir('features'):
            os.mkdir('features')

        with open(self._get_path('work_list/AllWorks.txt')) as f:
            for line in f:
                self.works.append(line.split('-')[0])
        if self.save_data:
            np.save('features/all_works', self.works)

    def clear_saved_data(clear_features=True, clear_npz=True):
        '''
        Clean up the cached data.
        '''
        import shutil
        if clear_features:
            shutil.rmtree('features')
        if clear_npz:
            shutil.rmtree('npz_data')

    def _get_path(self, *args):
        return path.join(self._self_dir, *args)

    def _vectorize(self, dict_list):
        feature = self._vectorizer.fit_transform(dict_list).toarray()
        names = self._vectorizer.get_feature_names()
        return feature, names

    def _dump_feature(self, feature, names, typename):
        fname_feature = 'features/feature_%s' % typename
        fname_names = 'features/names_%s.pkl' % typename
        np.save(fname_feature, feature)
        pickle.dump(names, open(fname_names, 'wb'))

    def _get_feature(self, typename):
        fname_feature = 'features/feature_%s.npy' % typename
        fname_names = 'features/names_%s.pkl' % typename
        if path.isfile(fname_feature) and path.isfile(fname_names):
            feature = np.load(fname_feature)
            names = pickle.load(open(fname_names, 'rb'))
            return feature, names

    def _get_attr_setup(typename='default', load_npz=True):
        def wrap(func):
            def wrap_func(self, *args):
                if self.use_saved_data:
                    data = self._get_feature(typename)
                    if data is not None:
                        return data
                if load_npz and self.npz_data is None:
                    self.load_npz_data()
                return func(self, *args)
            return wrap_func
        return wrap

    def get_labels(self):
        '''
        Decide the labels
        2 for unsecure Jos, 1 for secure Jos, 0 for others
        '''
        #TODO consider labeling different authors for multiclass assignment
        fname = 'features/labels'
        if self.use_saved_data and path.isfile(fname):
            return np.load(fname)
        with open(self._get_path('work_list/Josquin_secure.txt')) as f:
            secure_jos = set(f.read().splitlines())
        labels = []
        for work in self.works:
            label = 0
            if 'Jos' in work:
                if work in secure_jos:
                    label = 1
                else:
                    label = 2
            labels.append(label)
        labels = np.array(labels, dtype=int)
        if self.save_data:
            np.save(fname, labels)
        return labels

    def load_npz_data(self):
        '''
        Load cached numpy npz data for all notes' ps and quarterLength (ql).
        Each woek contains multiple parts as nested list, each part is a
        np.adarray. 1st column is ps (0 for rest) and second column is ql.
        '''
        import music21

        def get_npz(name):
            fname = 'npz_data/%s.npz' % name
            if self.use_saved_data and path.isfile(fname):
                all_data = np.load(fname)
                # Each work contains many parts. Loop through each one.
                return [all_data[i] for i in all_data.files]

            music_file = self._get_path('data/', name + '.krn')
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
            if self.save_data:
                np.savez(fname, *all_arr)
            return all_arr

        if self.save_data and not path.isdir('npz_data'):
            os.mkdir('npz_data')
        self.npz_data = []
        for work in self.works:
            self.npz_data.append(get_npz(work))

    @_get_attr_setup('ps_diff')
    def get_ps_diff(self):
        '''
        Get the ps differnce histogram feature.
        '''
        dict_list = []
        for arrs in self.npz_data:
            new_dict = Counter()
            for arr in arrs:
                is_note = arr[:, 0] != 0
                ps_diff = arr[1:, 0] - arr[:-1, 0]
                # Get rid of rests
                ps_diff = ps_diff[np.logical_and(is_note[1:], is_note[:-1])]
                new_dict.update(Counter(ps_diff))
            dict_list.append(new_dict)
        feature, names = self._vectorize(dict_list)
        if self.save_data:
            self._dump_feature(feature, names, 'ps_diff')
        return feature, names

    @_get_attr_setup('ql_diff')
    def get_ql_diff(self):
        '''
        Get the quarterLength differnce histogram feature (for notes only).
        '''
        dict_list = []
        for arrs in self.npz_data:
            new_dict = Counter()
            for arr in arrs:
                is_note = arr[:, 0] != 0
                ql_diff = arr[1:, 1] - arr[:-1, 1]
                # Get rid of rests
                ql_diff = ql_diff[np.logical_and(is_note[1:], is_note[:-1])]
                new_dict.update(Counter(ql_diff))
            dict_list.append(new_dict)
        feature, names = self._vectorize(dict_list)
        if self.save_data:
            self._dump_feature(feature, names, 'ql_diff')
        return feature, names

    @_get_attr_setup('ps_ql_pair')
    def get_ps_ql_pair(self):
        '''
        Get the ps-ql pair histogram feature.
        '''
        dict_list = []
        for arrs in self.npz_data:
            pairs = Counter()
            for arr in arrs:
                for ps, ql in arr:
                    # Reduce the features by folding the ps into range 1-12.
                    if ps != 0:
                        ps = ps % 12 + 1
                    pairs[(ps, ql)] += 1
            dict_list.append(pairs)
        feature, names = self._vectorize(dict_list)
        if self.save_data:
            self._dump_feature(feature, names, 'ps_ql_pair')
        return feature, names
