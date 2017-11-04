#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The details that matter: Frequency resolution of spectrograms in acoustic scene classification.

Train ambience (`amb`) only model with different band settings.

"""

import argparse
import os
import sys


if __name__ == '__main__':
    # Parse CLI
    parser = argparse.ArgumentParser()
    parser.add_argument('-D', '--device', help='Theano device used for computations')
    parser.add_argument('-b', '--bands', default=200, help='Number of mel bands or `stft`')
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
    if args.bands == 'stft':
        BANDS = 1103
    else:
        BANDS = int(args.bands)
        assert (BANDS > 0)
        assert (BANDS <= 200)

    folds = [1, 2, 3, 4] if not args.all else ['all']
    for FOLD in folds:
        RUN = str(args.bands)
        if args.all:
            RUN += '_all'
        np.random.seed(20170713)

        task = Task(mel_bands=args.bands)

        task.load_dataset(fold=FOLD)
        task.generate_features()

        inputs = Input(shape=(1, BANDS, 500))

        x = Conv2D(100, kernel_size=(BANDS, 50), kernel_initializer='he_uniform')(inputs)
        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)
        x = Dropout(0.25)(x)

        x = Conv2D(100, kernel_size=(1, 1), kernel_initializer='he_uniform')(x)
        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)
        x = Dropout(0.25)(x)

        x = Conv2D(15, kernel_size=(1, 1), kernel_initializer='he_uniform')(x)
        x = Lambda(softmax, arguments={'axis': 1}, name='softmax')(x)

        x = GlobalAveragePooling2D()(x)

        model = Model(inputs=inputs, outputs=x)
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
