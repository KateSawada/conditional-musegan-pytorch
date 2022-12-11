#!/bin/bash

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
source venv/bin/activate
python train.py -c config/base.yml config/train.yml -m tmp_recon_model_64_64_c10
