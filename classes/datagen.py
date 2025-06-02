import tensorflow as tf
import pandas as pd
import numpy as np
import itertools as it
import re

class TrialGenerator(tf.keras.utils.PyDataset):
    def __init__(self, cue_dict, target_dict, trial_length,
                 trial_blocks=None, batch_size=2,
                 input_noise=0.1, n_blocks=1, n_cues=1):
        ts_time = 3

        self._n_blocks = n_blocks
        if n_cues > 1:
            self._trial_cues = pd.concat(
                [pd.Series(list(b.T), name=a) for a, b in cue_dict.items()],
                           axis=1).stack().apply(pd.Series
                ).unstack(1).reorder_levels((1, 0), axis=1).sort_index(axis=1)
            self._trial_cues = self._trial_cues.reindex(np.arange(trial_length)).fillna(0).T
            self._trial_cues = self._trial_cues.unstack(1)
        else:
            self._trial_cues = pd.concat(
                [pd.Series(b, name=a) for a, b in cue_dict.items()], axis=1
                ).reindex(np.arange(trial_length)).fillna(0).T

        self._response_window = pd.concat(
            [pd.Series(b, name=a) for a, b in target_dict.items()], axis=1
            ).reindex(np.arange(trial_length)).reindex(
                self._trial_cues.index.unique(0), axis=1).fillna(0).T

        if trial_blocks == None:
            trial_blocks = list(self._trial_cues.index.get_level_values(0))
        trial_blocks = trial_blocks * n_blocks

        self._trial_start = \
            [1] * ts_time + [0]*(trial_length - ts_time)

        self._input_noise = input_noise
        self._window_length = trial_length
        self._trial_types = trial_blocks
        self._batch_size = batch_size
        self._n_cues = n_cues

        samples = len(self._response_window.reindex(trial_blocks).stack())
        self._length = (samples - trial_length) // batch_size
        self.prep_data()
        self._count = []

    @classmethod
    def format_validation(cls, data):
        odors = list(filter(re.compile('odor.*').search, data.columns))
        return  list(map(
            np.expand_dims,
            (data[['light'] + odors], data[['response']]),
            it.repeat(0)))

    @property
    def trial_cues(self):
        return self._trial_cues

    @property
    def response_window(self):
        return self._response_window

    def trial_starts(self, n):
        return pd.concat([self._trial_start] * n).reset_index(drop=True)

    def generate_trials(self, n, ts=0.1):
        if ts != 0.1:
            raise Exception('time steps not implemented')

        trial_list = pd.DataFrame(np.tile(self._trial_types, (n, 1))).apply(
            list, axis=1).apply(np.random.permutation).apply(
                pd.Series).stack().reset_index(drop=True)

        trials = trial_list.apply(lambda x: self.trial_cues.loc[x])
        trials.index = pd.MultiIndex.from_frame(trial_list.reset_index())
        trials = trials.stack(level=0, future_stack=True)
        trials.index.names = ['trial', 'type', 'idx']
        if len(trials.shape) == 1:
            X = trials.to_frame(name='cues')
            X['odor'] = X['cues'] + \
                        tf.random.normal((len(X['cues']),), mean=0,
                                         stddev=self._input_noise).numpy()
            cues = ['cues']
            odors = ['odor']
        else:
            cues = [f'cue_{a}' for a in trials.columns]
            trials.columns = cues
            X = trials
            for i, cue in enumerate(cues):
                X[f'odor_{i}'] = X[cue] + \
                                   tf.random.normal((X.shape[0],), mean=0,
                                                    stddev=self._input_noise).numpy()
            odors = [f'odor_{i}' for i, _ in enumerate(cues)]

        X['trial_start'] = np.tile(
            self._trial_start, (n * len(self._trial_types),))
        X['light'] = X['trial_start'] + \
                            tf.random.normal((X.shape[0],), mean=0,
                                             stddev=self._input_noise).numpy()
        X['response'] = trial_list.apply(
            lambda x: self.response_window.loc[x]).stack().values
        X = X.reindex(['trial_start', 'light'] + cues + odors + ['response'], axis=1)
        return X

    def data_splitter(self, features):
        inputs = features[:, slice(0, self._window_length), 0:(1 + self._n_cues)]
        labels = features[:, slice(0, self._window_length), (1 + self._n_cues):(2 + self._n_cues)]
        inputs.set_shape([None, self._window_length, None])
        labels.set_shape([None, self._window_length, None])
        return inputs, labels

    def prep_data(self):
        #trials = self.generate_trials(1)

        if self._n_cues > 1:
            data = np.array(self.generate_trials(1)[
                ['light'] + [f'odor_{a}' for a in range(self._n_cues)] + ['response']], dtype=np.float32)
        else:
            data = np.array(self.generate_trials(1)[
                ['light', 'odor', 'response']], dtype=np.float32)

        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self._window_length,
            sequence_stride=1,
            shuffle=False,
            batch_size=self._batch_size)

        ds = ds.map(self.data_splitter)
        self._ds_iter = iter(ds)
        self._ds = [d for d in ds]

    def on_epoch_end(self):
        self.prep_data()

    def __getitem__(self, index):
        #if False:
        #    if index in self._count:
        #        import pdb; pdb.set_trace()
        #    self._count.append(index)
        #    try:
        #        return next(self._ds_iter)
        #    except:
        #        import pdb; pdb.set_trace()
        return self._ds[index]

    def __len__(self):
        return self._length


class tDNMSGenerator(TrialGenerator):
    def __init__(self, trial_block=('SS', 'SL', 'LS', 'SS'), **kwargs):
        ss = [0] * 30 + [1] * 20 + [0] * 30 + [1] * 20
        sl = [0] * 30 + [1] * 20 + [0] * 30 + [1] * 50
        ls = [0] * 30 + [1] * 50 + [0] * 30 + [1] * 20
        cues = {'SS': ss, 'SL': sl, 'LS': ls}

        response = np.zeros(150)
        response[131:150] = 1
        targets = {'SL': response, 'LS': response}
        super(tDNMSGenerator, self).__init__(
            cues, targets, 200, trial_blocks=trial_block, **kwargs)

class genFactory:
    @classmethod
    def create(cls, task, input_noise, batch_size, n_blocks):
        if task == 'just_short_match' or task == 'potentiate':
            traingen = tDNMSGenerator(
                input_noise=input_noise, batch_size=2, n_blocks=n_blocks)
       else:
            raise Exception('task not found')

       return traingen
