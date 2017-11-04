""" A very rough modification of https://github.com/CrowdCurio/audio-annotator."""
import json
import glob
import mimetypes
import os

import numpy as np
import pandas as pd

from flask import Flask, Response, request, send_file
from flask import session, redirect, url_for, jsonify, render_template

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/next', methods=['GET'])
def next_task():
    meta = pd.read_csv('/volatile/dcase17_1/meta.txt', sep='\t',
                       names=['file', 'scene', 'recording'])
    clues = pd.read_csv('/volatile/dcase17_1/clues.txt', sep='\t',
                        names=['file', 'start', 'end', 'label'])

    meta = meta.sample(frac=1, random_state=20170713).reset_index(drop=True)

    completed = pd.unique(clues['file'])
    meta = meta[~meta['file'].isin(completed)]

    meta = meta[(meta['scene'] == 'train')].reset_index(drop=True)

    print('Rows: ', len(meta))

    # idx = np.random.randint(len(meta))
    idx = 0

    audio_file = meta.ix[idx, 0].replace('audio/', '')
    audio_scene = meta.ix[idx, 1]
    audio_prediction = meta.ix[idx, 2]

    # audio_file = os.path.basename(np.random.choice(glob.glob('data/audio/*')))
    # audio_scene = meta[meta['file'] == 'audio/' + audio_file].scene.values[0]

    task = dict(feedback="none",
                visualization='spectrogram',
                proximityTag=[],
                annotationTag=['dishes', 'page_flip'],
                url="/static/data/audio/" + audio_file,
                numRecordings='?',
                file=audio_file,
                info='<strong>{}</strong> (pred: {})<br />{}'.format(audio_scene, audio_prediction, audio_file),  # noqa
                tutorialVideoURL="https://www.youtube.com/embed/Bg8-83heFRM",
                alwaysShowTags=True)
    data = json.dumps(dict(task=task))
    # app.logger.debug("Returning:\n{}".format(data))
    resp = Response(data)
    return resp


@app.route('/api/submit', methods=['POST'])
def save_annotation():
    if request.headers['Content-Type'] == 'application/json':
        # app.logger.info("Received Annotation:\n{}".format(json.dumps(request.json, indent=2)))

        clues = pd.read_csv('/volatile/dcase17_1/clues.txt', sep='\t',
                            names=['file', 'start', 'end', 'label'])

        file = 'audio/' + request.json['file']

        for annotation in request.json['annotations']:
            start = np.round(annotation['start'], 2)
            end = np.round(annotation['end'], 2)
            label = annotation['annotation']

            # app.logger.info('{},{},{},{}'.format(file, start, end, label))

            row = pd.DataFrame(columns=('file', 'start', 'end', 'label'))
            row.loc[0] = (file, start, end, label)
            # print(row)

            clues = clues.append(row, ignore_index=True)

        if not len(request.json['annotations']):
            row = pd.DataFrame(columns=('file', 'start', 'end', 'label'))
            row.loc[0] = (file, 0.0, 0.0, 'none')
            clues = clues.append(row, ignore_index=True)

        print(clues.groupby(['label']).aggregate('count'))
        clues.to_csv('/volatile/dcase17_1/clues.txt', sep='\t', header=False, index=False)

        data = json.dumps(dict(message='Success!'))
        status = 200

    resp = Response(data, status=status, mimetype=mimetypes.types_map[".json"])
    return resp
