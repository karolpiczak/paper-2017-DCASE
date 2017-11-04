#!/bin/bash

export FLASK_APP=audiomark.py
export FLASK_DEBUG=1

flask run --port=8027 --host=10.0.0.3
