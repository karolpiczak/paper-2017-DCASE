#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The details that matter: Frequency resolution of spectrograms in acoustic scene classification.

Train hybrid model with multiple detectors (`detectors`).

"""

import argparse
import os
import sys


if __name__ == '__main__':
    # Parse CLI
    parser = argparse.ArgumentParser()
    parser.add_argument('-D', '--device', help='Theano device used for computations')
    parser.add_argument('--all', dest='all', action='store_true',
                        help='Train on all folds (for final model testing)')
    args = parser.parse_args()

    # Initialize GPU / Keras
    DEVICE = args.device if args.device else 'gpu0'
    THEANO_FLAGS = ('device={},'
                    'floatX=float32,'
                    'dnn.conv.algo_bwd_filter=deterministic,'
                    'dnn.conv.algo_bwd_data=deterministic').format(DEVICE)
    os.environ['THEANO_FLAGS'] = THEANO_FLAGS
    os.environ['KERAS_BACKEND'] = 'theano'

    # Import internals
    sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/..')

    from arf.core import *
    from arf.generics import describe
    import arf.monitor

    # ----------------------------------------------------------------------
    # Train
    BANDS = 200

    folds = [1, 2, 3, 4] if not args.all else ['all']
    for FOLD in folds:
        RUN = 'detectors'
        if args.all:
            RUN += '_all'
        np.random.seed(20170713)

        task = Task(mel_bands=BANDS)

        task.load_dataset(fold=FOLD)
        task.generate_features()

        # Event detectors

        def make_detector(inp):
            fg = Conv2D(10, kernel_size=(BANDS, 50), kernel_initializer='he_uniform')(inp)
            fg = BatchNormalization(axis=1)(fg)
            fg = LeakyReLU()(fg)
            fg = Dropout(0.25)(fg)

            fg = Conv2D(10, kernel_size=(1, 1), kernel_initializer='he_uniform')(fg)
            fg = BatchNormalization(axis=1)(fg)
            fg = LeakyReLU()(fg)
            fg = Dropout(0.25)(fg)

            fg = Conv2D(1, kernel_size=(1, 1), kernel_initializer='he_uniform',
                        activation='sigmoid')(fg)

            fg = GlobalMaxPooling2D()(fg)
            fg = RepeatVector(451)(fg)

            return Reshape((1, 1, 451))(fg)

        # Background

        inputs = Input(shape=(1, BANDS, 500))

        bg = Conv2D(100, kernel_size=(BANDS, 50), kernel_initializer='he_uniform')(inputs)
        bg = BatchNormalization(axis=1)(bg)
        bg = LeakyReLU()(bg)
        bg = Dropout(0.25)(bg)

        bg = Conv2D(100, kernel_size=(1, 1), kernel_initializer='he_uniform')(bg)
        bg = BatchNormalization(axis=1)(bg)
        bg = LeakyReLU()(bg)
        bg = Dropout(0.25)(bg)

        branches = [bg]

        for i in range(15):
            branches.append(make_detector(inputs))

        bg = concatenate(branches, axis=1)

        bg = Conv2D(15, kernel_size=(1, 1), kernel_initializer='he_uniform')(bg)
        bg = Lambda(softmax, arguments={'axis': 1}, name='softmax')(bg)

        bg = GlobalAveragePooling2D()(bg)

        model = Model(inputs=inputs, outputs=bg)
        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001),
                      metrics=['accuracy'])

        describe(model)

        batch_size = 32

        holdout_batch = next(task.iterbatches(len(task.holdout), task.holdout))
        validation_batch = next(task.iterbatches(len(task.validation), task.validation))

        monitor = arf.monitor.Monitor(model, holdout_batch, validation_batch, RUN, FOLD)

        model.fit_generator(generator=task.iterbatches(batch_size, task.train, augment=True),
                            steps_per_epoch=len(task.train) // batch_size,
                            epochs=500,
                            callbacks=[monitor],
                            max_queue_size=10)

        model.load_weights(f'results/run_{RUN}_{FOLD}.h5')
        task.save_predictions(model, run=RUN, fold=FOLD)
