#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH -p t4
#SBATCH --cpus-per-gpu=1
#SBATCH --time=2:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=deepinversion
#SBATCH --output=log_%j.out

. /etc/profile.d/lmod.sh
module use /pkgs/environment-modules/
module load pytorch1.4-cuda10.1-python3.6
pip install wandb --upgrade
cd /h/yanf/DeepInversion
python -m wandb.cli login eb2b6f90693426e29926f54d26564f59e0e3dc5c 
#(while true; do nvidia-smi; top -b -n 1 | head -20; sleep 10; done) &
python cifar10_inversion.py
#wait
