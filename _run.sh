#!/bin/bash

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
source venv/bin/activate
python train.py -c config/train_con.yml config/base_con.yml -m conditioning_middle_model_64
