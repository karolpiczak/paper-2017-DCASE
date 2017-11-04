
###### *Paper replication data for:*

## The details that matter: Frequency resolution of spectrograms in acoustic scene classification

<a href="http://karol.piczak.com/papers/Piczak2017-DCASE.pdf"><img src="https://img.shields.io/badge/paper-PDF-ff69b4.svg" alt="Download paper in PDF format" title="Download paper in PDF format" align="right" /></a>

<a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT licensed" title="MIT licensed" /></a>

## Overview



## Repository content

#### [`/Paper/`](/Paper)

LaTeX source code for the paper.

#### [`/Submission/`](/Submission/)

Actual submission package as delivered for [DCASE 2017](http://www.cs.tut.fi/sgn/arg/dcase2017/challenge/submission).

#### [`/Code`](/Code/)

##### `/Code/*.py`

Source code for experiments.

Run [`run_amb.py`](Code/run_amb.py) to train the ambience model:

```
    $ python run_amb.py -h

    usage: run_amb.py [-h] [-D DEVICE] [-b BANDS] [--all]

    optional arguments:
      -h, --help            show this help message and exit
      -D DEVICE, --device DEVICE
                            Theano device used for computations
      -b BANDS, --bands BANDS
                            Number of mel bands or `stft`
      --all                 Train on all folds (for final model testing)
```

This will generate `run_{BANDS}_{FOLD}.*` files in the [`results/`](Code/results/) directory:

- `run_{BANDS}_{FOLD}.h5` - saved weights (highest holdout score)
- `run_{BANDS}_{FOLD}.last.h5` - saved weights (last epoch)
- `run_{BANDS}_{FOLD}.npz` - training history
- `run_{BANDS}_{FOLD}.txt` - generated predictions

Settings used for training models in the paper:

```python
    BANDS in [40, 60, 100, 200, 'stft']
```

The `--all` switch is used for training the final model (all training folds included).

Issuing:

```shell
    $ python run_detectors.py
```

will train a hybrid model (ambience module + 15 binary event detectors). [`run_dishes.py`](Code/run_dishes.py) is
a streamlined version with only 1 binary detector pre-trained on typical `cafe/restaurant` sounds
(kitchenware, cutlery, crockery etc.). Detector pre-training is done with [`train_clues.py`](Code/train_clues.py) based on [`clues.txt`](Code/clues.txt) annotations.

##### [`/Code/arf/`](/Code/arf/)

Helper/backend code for generating submissions.

##### [`/Code/Figures.ipynb`](/Code/Figures.ipynb) & [`/Code/figures/`](/Code/figures/)

Raw figures with code used for visualization available as a Jupyter notebook ([`Figures.ipynb`](Code/Figures.ipynb)).

##### [`/Code/results/`](/Code/results/)

Training outputs. The [`eval.py`](Code/eval.py) script creates cross-validation accuracy metrics, list of misclassifications
and a confusion matrix for a given system:

```
    $ python eval.py run_200
```

The `_th_0.5` suffixes denote models with prediction thresholding, so:

```
    $ python eval.py run_200_th_0.5
```

will generate a corresponding confusion matrix [`Code/results/run_200_th_0.5.pdf`](Code/results/run_200_th_0.5.pdf).

##### [`/Code/annotator/`](/Code/annotator/)

A very unpolished modification of [CrowdCurio's audio-annotator](https://github.com/CrowdCurio/audio-annotator) JavaScript interface used for creating the `dishes` ([`clues.txt`](Code/clues.txt)) annotation list.

## Reference

If you find this paper useful in some way, you can cite it with the following BibTeX entry:

```bibtex
@inproceedings{piczak2017dcase,
    title={The details that matter: Frequency resolution of spectrograms in acoustic scene classification},
    author={Piczak, Karol J.},
    booktitle={Proceedings of the Detection and Classification of Acoustic Scenes and Events 2017 Workshop},
    year={2017},
    location={Munich, Germany}
}
```
