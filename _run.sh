#!/bin/bash

source /data/group1/${USER}/data/musegan-pytorch/venv/bin/activate
python -u python train.py -c config/train.yml config/base.yml -m model/furo/
