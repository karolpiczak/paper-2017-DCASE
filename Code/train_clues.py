#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The details that matter: Frequency resolution of spectrograms in acoustic scene classification.

Train `dishes` detector on pre-annotated data (`clues.txt`).

"""

import os

os.environ['THEANO_FLAGS'] = ('floatX=float32,'
                              'device=gpu0,'
                              'dnn.conv.algo_bwd_filter=deterministic,'
                              'dnn.conv.algo_bwd_data=deterministic')

from arf.core import *
from arf.generics import describe
import arf.monitor


if __name__ == '__main__':
    for FOLD in [1, 2, 3, 4]:
        RUN = 'd'
        LABEL = 'dishes'
        np.random.seed(20170713)

        task = Task(mel_bands=200)

        task.load_dataset(fold=FOLD)
        task.generate_features()

        clues = pd.read_csv(f'{DATA_PATH}/clues.txt', sep='\t',
                            names=['file', 'start', 'end', 'label'],
                            converters={'file': lambda s: s.replace('audio/', '')})
        clues = clues[clues.label != 'none'].reset_index(drop=True)
        clues = clues[clues.file.isin(task.train.file)]  # limit to training fold

        def get_fragment(file, start, end):
            audio, _ = librosa.core.load(f'{DATA_PATH}/audio/{file}', sr=44100, dtype=np.float16,
                                         duration=10.0)
            audio = audio[int(start * 44100):int(end * 44100)]
            return audio

        def generate_example(label=None):
            audio = None

            while audio is None:
                file = task.train.sample(1).reset_index(drop=True).file[0]
                start = np.random.rand() * 9.0
                end = start + 1.0

                # Check if not overlapping clue event
                c = clues[clues.file == file]

                if not len(c[((c.start > start) & (c.start < end)) | ((c.end > start) & (c.end < end))]):  # noqa
                    audio = get_fragment(file, start, end)

            if label is not None:
                clue = clues[(clues.label == label)].sample(1).reset_index(drop=True)
                file = clue.file[0]
                start = clue.start[0]
                end = clue.end[0]

                if end - start < 1.0:
                    end = start + 1.0
                if end > 10.0:
                    end = 10.0

                overlay = get_fragment(file, start, end)

                overlay_rmse = np.mean(librosa.feature.rmse(overlay))
                audio_rmse = np.mean(librosa.feature.rmse(audio))
                audio *= overlay_rmse / audio_rmse
                audio *= 0.1

                offset = np.random.randint(4410)
                overlay = overlay[offset:offset + 44100].copy()
                overlay.resize(44100)
                audio += 0.9 * overlay

            spec = librosa.feature.melspectrogram(audio, sr=44100, n_fft=2205, hop_length=882,
                                                  n_mels=200, fmax=22050, power=2)
            spec = librosa.power_to_db(spec)
            return spec

        audio_mean = 0.0
        audio_std = 1.0

        def iterbatches(batch_size):
            while True:
                X, y = [], []

                for i in range(batch_size):
                    if np.random.rand() < 0.5:
                        X.append(np.stack([generate_example(LABEL)]))
                        y.append(1.0)
                    else:
                        X.append(np.stack([generate_example(None)]))
                        y.append(0.0)

                X = np.stack(X)
                y = np.array(y)

                X -= audio_mean
                X /= audio_std

                yield X, y

        X, _ = next(iterbatches(100))

        audio_mean = X.mean()
        audio_std = X.std()

        print(audio_mean, audio_std)
        np.savez(f'results/d_{LABEL}_{FOLD}.npz', audio_mean=audio_mean, audio_std=audio_std)

        inputs = Input(shape=(1, 200, 50))

        x = Conv2D(10, kernel_size=(200, 50), kernel_initializer='he_uniform')(inputs)
        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)
        x = Dropout(0.25)(x)

        x = Conv2D(10, kernel_size=(1, 1), kernel_initializer='he_uniform')(x)
        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)
        x = Dropout(0.25)(x)

        x = Conv2D(1, kernel_size=(1, 1), kernel_initializer='he_uniform', activation='sigmoid',
                   name='out')(x)
        x = Flatten()(x)

        model = Model(inputs=inputs, outputs=x)
        model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001),
                      metrics=['accuracy'])

        describe(model)

        model.fit_generator(generator=iterbatches(50), steps_per_epoch=20,
                            epochs=50, max_queue_size=10,
                            callbacks=[ModelCheckpoint(f'results/d_{LABEL}_{FOLD}.h5')])
