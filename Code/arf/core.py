# -*- coding: utf-8 -*-
"""Audio recognition framework.

Helper tools for DCASE 2017 submission.

"""

import os
import subprocess
import sys
import time

import IPython

import matplotlib
matplotlib.use('Agg')
import librosa
import numpy as np
import pandas as pd
import sklearn as sk
import skimage.measure

from tqdm import *

DATA_PATH = '/volatile/dcase17_1/'
TEST_PATH = '/volatile/dcase17_1_eval/'

os.environ['KERAS_BACKEND'] = 'theano'

import keras
keras.backend.set_image_data_format('channels_first')
from keras import backend as K
from keras.models import Model
from keras.activations import softmax
from keras.layers import Input, concatenate
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Activation, Dense, Dropout, Flatten, Reshape, Lambda, RepeatVector
from keras.callbacks import LearningRateScheduler, Callback, ModelCheckpoint
from keras.layers.normalization import BatchNormalization


L1 = keras.regularizers.l1
L2 = keras.regularizers.l2


class Task:
    def __init__(self, mel_bands):
        self.train = None
        self.holdout = None
        self.validation = None
        self.test = None
        self.bands = mel_bands
        self.label_encoder = sk.preprocessing.LabelEncoder()
        self.n_scenes = None

        self.audio_mean = 0.
        self.audio_std = 1.

    def load_dataset(self, fold):
        def read_fold(filename):
            return pd.read_csv(f'{DATA_PATH}/evaluation_setup/{filename}',
                               sep='\t', names=['file', 'scene'],
                               converters={'file': lambda s: s.replace('audio/', '')})

        assert(fold in [1, 2, 3, 4, 'all'])

        if fold == 'all':
            self.train = read_fold('fold1_train.txt')
            self.validation = read_fold('fold1_evaluate.txt')
            self.train = pd.concat([self.train, self.validation], ignore_index=True)

            self.test = pd.read_csv(f'{TEST_PATH}/evaluation_setup/test.txt',
                                    sep='\t', names=['file'],
                                    converters={'file': lambda s: s.replace('audio/', '')})

            print(f'Loaded all {len(self.train)} segments for training.')
        else:
            self.train = read_fold(f'fold{fold}_train.txt')
            self.validation = read_fold(f'fold{fold}_evaluate.txt')

        self.label_encoder.fit(sorted(pd.unique(self.train['scene'])))
        self.n_scenes = len(self.label_encoder.classes_)

        # Split training into training/holdout
        self.holdout = self.train.sample(400, random_state=20170713)
        self.train = self.train.drop(self.holdout.index)

    def generate_features(self):
        for row in tqdm(self.train.itertuples(), total=len(self.train)):
            self._generate_spec(f'{DATA_PATH}/audio/{row.file}')

        for row in tqdm(self.holdout.itertuples(), total=len(self.holdout)):
            self._generate_spec(f'{DATA_PATH}/audio/{row.file}')

        for row in tqdm(self.validation.itertuples(), total=len(self.validation)):
            self._generate_spec(f'{DATA_PATH}/audio/{row.file}')

        if self.test is not None:
            for row in tqdm(self.test.itertuples(), total=len(self.test)):
                self._generate_spec(f'{TEST_PATH}/audio/{row.file}')

        X, _ = next(self.iterbatches(1000, self.train))

        self.audio_mean = np.mean(X)
        self.audio_std = np.std(X)

        print(f'Input data parameters: mean = {self.audio_mean}, std = {self.audio_std}.')

    def _generate_spec(self, recording):
        spec_file = f'{recording}.spec{self.bands}.npy'

        if os.path.exists(spec_file):
            return

        audio, _ = librosa.core.load(recording, sr=44100, dtype=np.float16, duration=10.0)

        if self.bands == 'stft':
            spec = np.abs(librosa.stft(audio, n_fft=2205, hop_length=882))
        else:
            spec = librosa.feature.melspectrogram(audio, sr=44100, n_fft=2205, hop_length=882,
                                                  n_mels=self.bands, fmax=22050, power=2)
        spec = librosa.power_to_db(spec)

        np.save(spec_file, spec.astype('float16'), allow_pickle=False)

    def _load_spec(self, recording):
        return np.load(f'{recording}.spec{self.bands}.npy').astype('float32')

    @staticmethod
    def _iterrows(dataset):
        while True:
            dataset['mixin'] = ''
            for row in dataset.itertuples():
                # Sample a paired recording of the same class for each example
                other = dataset.loc[dataset['scene'] == row.scene, 'file'].sample().values
                dataset.loc[dataset['file'] == row.file, 'mixin'] = other
            for row in dataset.iloc[np.random.permutation(len(dataset))].itertuples():
                yield row

    def iterbatches(self, batch_size, dataset, augment=False):
        itrain = self._iterrows(dataset)

        while True:
            X, y = [], []

            for i in range(batch_size):
                row = next(itrain)
                spec = self._load_spec(f'{DATA_PATH}/audio/{row.file}')

                if augment:
                    spec_mixin = self._load_spec(f'{DATA_PATH}/audio/{row.mixin}')
                    offset = np.random.randint(np.shape(spec)[1])
                    spec[:, offset:] = spec_mixin[:, offset:]

                    delay = np.random.randint(50)
                    if delay > 0:
                        spec[:, delay:] = spec[:, :-delay]
                        spec[:, :delay] = 0.0

                scene_id = self.label_encoder.transform([row.scene])[0]

                X.append(np.stack([spec]))
                y.append(keras.utils.to_categorical(scene_id, self.n_scenes)[0])

            X = np.stack(X)
            y = np.array(y)

            X -= self.audio_mean
            X /= self.audio_std

            yield X, y

    def save_predictions(self, model, run, fold, mode='validation'):
        if mode == 'validation':
            dataset = self.validation
        elif mode == 'test':
            dataset = self.test

        X, files, predictions = [], [], []

        for row in tqdm(dataset.itertuples()):
            if mode == 'test':
                spec = self._load_spec(f'{TEST_PATH}/audio/{row.file}')
            else:
                spec = self._load_spec(f'{DATA_PATH}/audio/{row.file}')

            X.append(np.stack([spec]))
            files.append(f'audio/{row.file}')

        X = np.stack(X)

        X -= self.audio_mean
        X /= self.audio_std

        predictions = model.predict(X)
        predictions = np.argmax(predictions, axis=1)
        predictions = self.label_encoder.classes_[predictions]

        results = pd.DataFrame({'file': files, 'scene': predictions},
                               columns=['file', 'scene'])
        results = results.sort_values('file')
        if mode != 'test':
            results.to_csv(f'results/run_{run}_{fold}.txt', sep='\t', index=False, header=False)

        predict = K.function([model.input] + [K.learning_phase()], model.layers[-2].output)
        # Thresholded predictions
        for threshold in [0.5]:
            predictions = predict([X, 1.])
            predictions = np.select([predictions >= threshold, predictions < threshold], [1, 0])
            predictions = np.mean(predictions, axis=3)[..., 0]
            predictions = np.argmax(predictions, axis=1)

            predictions = self.label_encoder.classes_[predictions]

            results = pd.DataFrame({'file': files, 'scene': predictions},
                                   columns=['file', 'scene'])
            results = results.sort_values('file')
            if mode == 'test':
                results.to_csv(f'results/run_{run}_test.txt', sep='\t', index=False, header=False)
            else:
                results.to_csv(f'results/run_{run}_th_{np.round(threshold, 2)}_{fold}.txt',
                               sep='\t', index=False, header=False)
