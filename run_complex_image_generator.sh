#!/bin/bash

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate cs236781-hw
echo "hello from $(python --version) in $(which python)"

python DiffusionGenerator.py