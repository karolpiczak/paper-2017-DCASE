#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Results evaluator.

Creates cross-validation accuracy metrics, misclassifications
and a confusion matrix for a given system.

"""

import argparse
import os

import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.metrics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sb
sb.set(style="white", palette="muted")
matplotlib.rcParams['font.family'] = 'PT Sans'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate system.')
    parser.add_argument('system')
    args = parser.parse_args()

    meta = pd.read_csv('meta.txt', sep='\t', names=['file', 'scene', 'recording'])
    labels = sorted(pd.unique(meta['scene']))
    conf = np.zeros([len(labels), len(labels)])

    N_TRUE = 0
    N_ALL = 0

    for fold in [1, 2, 3, 4]:
        if not os.path.exists(f'{args.system}_{fold}.txt'):
            continue

        results = pd.read_csv(f'{args.system}_{fold}.txt', sep='\t', names=['file', 'scene'])
        results = results.merge(meta, how='left', on='file', suffixes=['_predicted', '_true'])
        results['correct'] = results['scene_predicted'] == results['scene_true']

        n_true = np.sum(results['correct'])
        n_all = len(results['correct'])
        accuracy = n_true / n_all

        conf += sk.metrics.confusion_matrix(results['scene_true'], results['scene_predicted'])
        print(f'Fold {fold}: {n_true} / {n_all} = {np.round(accuracy * 100, 2)}%')

        N_TRUE += n_true
        N_ALL += n_all

        results[results['correct'] == False].to_csv(  # noqa
            f'{args.system}_{fold}_miss.txt',
            columns=['file', 'scene_true', 'scene_predicted'],
            sep='\t',
            index=False,
            header=False
        )

    print(N_TRUE, N_ALL, np.round(N_TRUE / N_ALL, 4))

    plt.imshow(conf / conf.sum(axis=1)[:, np.newaxis], interpolation='nearest',
               cmap=sb.cubehelix_palette(8, as_cmap=True))
    plt.xticks(range(0, len(labels)), [label[:3] for label in labels], fontsize=9)
    plt.yticks(range(0, len(labels)), labels, fontsize=9)
    plt.ylabel('True label', fontsize=10)
    plt.xlabel('Predicted label', fontsize=10)

    for x in range(len(labels)):
        for y in range(len(labels)):
            if x == y:
                plt.text(x, y + 0.15, int(conf[y, x]), fontsize=7, ha='center', color='white')
            else:
                plt.text(x, y + 0.15, int(conf[y, x]), fontsize=7, ha='center')

        plt.text(15, x + 0.15, f'{np.round(conf[x, x] / np.sum(conf, axis=1)[x] * 100, 1)}%',
                 fontsize=7, ha='left')

    plt.title('Overall accuracy: {}%'.format(np.round(N_TRUE / N_ALL * 100, 2)), fontsize=10)

    plt.savefig('{}.pdf'.format(args.system), bbox_inches='tight')
