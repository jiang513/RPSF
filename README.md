# RPSF: Recovering Permuted Sequential Features for Effective Reinforcement Learning
This is a PyTorch implementation of **svea-RPSF** using Convolution Neural Networks and Vision Transformers.
## Setup
We assume that you have access to a GPU with CUDA >=9.2 support. All dependencies can then be installed with the following commands:
'''
cd ./cnn
conda env create -f ./setup/conda.yaml
conda activate svea-RPSF
sh ./setup/install_envs.sh
'''
## Training & Evaluation
In the 'cnn' or 'transformer' directory, 'scripts' directory contains bash scripts for svea-RPSF. You can run svea-RPSF by 'sh /cnn/scripts/svea-RPSF' or 'sh '
