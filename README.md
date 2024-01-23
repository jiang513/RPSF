# RPSF: Recovering Permuted Sequential Features for Effective Reinforcement Learning
This is a PyTorch implementation of **SVEA-C-RPSF** and **SVEA-T-RPSF** using Convolution Neural Networks and Vision Transformers respectively.
## Setup
We assume that you have access to a GPU with CUDA >=9.2 support. All dependencies can then be installed with the following commands:
```
cd ./cnn
conda env create -f ./setup/conda.yaml
conda activate svea-c-rpsf
sh ./setup/install_envs.sh
```
**SVEA-C-RPSF** and **SVEA-T-RPSF** use the same dependencies.
## Training & Evaluation
In the `cnn` and `transformer` directories, `scripts` directories contain bash scripts for **SVEA-C-RPSF** and **SVEA-T-RPSF**, which can be run by `sh /cnn/scripts/svea-c-rpsf.sh` and `sh /transformer/scripts/svea-t-rpsf.sh` respectively.

Alternatively, you can call the python scripts directly, e.g. for training of **SVEA-C-RPSF** call
```
python3 cnn/src/train.py --seed 0 --algorithm svea --use_aux
```
to run **SVEA-C-RPSF** on the default task, `walker_walk`, and using the default hyperparameters.
