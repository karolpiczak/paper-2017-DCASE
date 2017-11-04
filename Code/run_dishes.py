#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The details that matter: Frequency resolution of spectrograms in acoustic scene classification.

Train hybrid model with a single pretrained detector (`dishes`).

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
        RUN = 'dishes'
        if args.all:
            RUN += '_all'
        np.random.seed(20170713)

        task = Task(mel_bands=BANDS)

        task.load_dataset(fold=FOLD)
        task.generate_features()

        # Event detectors

        def make_detector(inp, name):
            fg = Conv2D(10, kernel_size=(BANDS, 50), kernel_initializer='he_uniform',
                        name=f'{name}_1')(inp)
            fg = BatchNormalization(axis=1, name=f'{name}_2')(fg)
            fg = LeakyReLU(name=f'{name}_3')(fg)
            fg = Dropout(0.25, name=f'{name}_4')(fg)

            fg = Conv2D(10, kernel_size=(1, 1), kernel_initializer='he_uniform',
                        name=f'{name}_5')(fg)
            fg = BatchNormalization(axis=1, name=f'{name}_6')(fg)
            fg = LeakyReLU(name=f'{name}_7')(fg)
            fg = Dropout(0.25, name=f'{name}_8')(fg)

            fg = Conv2D(1, kernel_size=(1, 1), kernel_initializer='he_uniform',
                        activation='sigmoid', name=f'{name}_9')(fg)

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

        for i in range(1):
            branches.append(make_detector(inputs, 'dishes'))

        bg = concatenate(branches, axis=1)

        bg = Conv2D(15, kernel_size=(1, 1), kernel_initializer='he_uniform')(bg)
        bg = Lambda(softmax, arguments={'axis': 1}, name='softmax')(bg)

        bg = GlobalAveragePooling2D()(bg)

        model = Model(inputs=inputs, outputs=bg)
        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001),
                      metrics=['accuracy'])

        describe(model)

        # Load weights from pretrained detector:
        if FOLD == 'all':
            pretrained = keras.models.load_model(f'results/d_dishes_2.h5')
        else:
            pretrained = keras.models.load_model(f'results/d_dishes_{FOLD}.h5')

        for layer_idx in [1, 2, 5, 6]:
            model.get_layer(f'dishes_{layer_idx}').set_weights(
                pretrained.layers[layer_idx].get_weights()
            )

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
