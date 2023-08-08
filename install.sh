#!/bin/sh
set -e
conda install astra-toolbox -c astra-toolbox
conda install pytorch pytorch-cuda -c pytorch -c nvidia
conda install tomosipo=0.5 -c aahendriksen
conda install tomopy xraylib -c conda-forge
conda install numpy scipy scikit-image tqdm dill matplotlib
pip install git+https://github.com/ahendriksen/ts_algorithms
pip install -e .
