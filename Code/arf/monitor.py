# -*- coding: utf-8 -*-
"""Training progress reporting."""

import time
import pdb
import IPython

import keras
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import sklearn.metrics

from arf.generics import to_percentage


sb.set(style="white", palette="muted")


class Monitor(keras.callbacks.Callback):
    def __init__(self, model, holdout_batch, validation_batch, run, fold):
        self.model = model
        self.holdout_batch = holdout_batch
        self.validation_batch = validation_batch
        self.run = str(run)
        self.fold = fold
        self.loss = []
        self.train_score = []
        self.holdout_score_bin = []
        self.validation_score = []
        self.validation_score_bin = []
        self.start_time = None
        self.best_score = 0.0

    def on_train_begin(self, logs={}):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.loss.append(logs.get('loss'))
        self.train_score.append(logs.get('acc'))

        predictions = self.model.predict(self.validation_batch[0])
        predictions = np.argmax(predictions, axis=1)
        ground_truth = np.argmax(self.validation_batch[1], axis=1)

        validation_score = np.sum(ground_truth == predictions) / len(predictions)
        self.validation_score.append(validation_score)

        # Binarization
        # IPython.embed()

        THRESHOLD = 0.5
        predict = K.function([self.model.input] + [K.learning_phase()],
                             self.model.get_layer('softmax').output)

        predictions = predict([self.validation_batch[0], 1.])
        predictions = np.select([predictions >= THRESHOLD, predictions < THRESHOLD], [1, 0])
        predictions = np.mean(predictions, axis=3)[..., 0]
        predictions = np.argmax(predictions, axis=1)
        validation_score_bin = np.sum(ground_truth == predictions) / len(predictions)
        self.validation_score_bin.append(validation_score_bin)

        ground_truth = np.argmax(self.holdout_batch[1], axis=1)
        predictions = predict([self.holdout_batch[0], 1.])
        predictions = np.select([predictions >= THRESHOLD, predictions < THRESHOLD], [1, 0])
        predictions = np.mean(predictions, axis=3)[..., 0]
        predictions = np.argmax(predictions, axis=1)
        holdout_score_bin = np.sum(ground_truth == predictions) / len(predictions)
        self.holdout_score_bin.append(holdout_score_bin)

        time_elapsed = time.time() - self.start_time
        self.start_time = time.time()

        if holdout_score_bin > self.best_score or np.isclose(holdout_score_bin, self.best_score):
            self.best_score = holdout_score_bin
            self.model.save_weights(f'results/run_{self.run}_{self.fold}.h5')

        self.model.save_weights(f'results/run_{self.run}_{self.fold}.last.h5')

        # Save results
        print(f' --- Hld: {np.round(holdout_score_bin, 3)}, '
              f'Val: {np.round(validation_score, 3)}, '
              f'Bin val: {np.round(validation_score_bin, 3)}')

        np.savez(f'results/run_{self.run}_{self.fold}.npz', loss=self.loss,
                 train_score=self.train_score, validation_score=self.validation_score,
                 validation_score_bin=self.validation_score_bin,
                 holdout_score_bin=self.holdout_score_bin)
