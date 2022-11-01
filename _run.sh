#!/bin/bash

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
source /data/group1/${USER}/data/musegan-pytorch/venv/bin/activate
python -u python train.py -c config/train.yml config/base.yml -m model/furo/
