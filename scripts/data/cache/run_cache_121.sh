#!/bin/zsh
source $HOME/.zshrc

export WANDB_DIR=/scratch/cluster/nimit
cd $HOME/Documents/robomaster

conda env list
conda activate /scratch/cluster/nimit/miniconda3/envs/robo

python -c "import consistency.point_dataset; consistency.point_dataset.get_episode('/scratch/cluster/nimit/data/carla/training/route_12_2')"
