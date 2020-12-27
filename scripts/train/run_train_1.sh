#!/bin/zsh
source $HOME/.zshrc

export WANDB_DIR=/scratch/cluster/nimit
cd $HOME/Documents/robomaster

conda env list
conda activate /scratch/cluster/nimit/miniconda3/envs/robo
PYTHONHASHSEED=0 python -m consistency.train \
        --description 12.26-cf-v1 \
        --max_epoch 2000 \
        --checkpoint_dir /scratch/cluster/nimit/checkpoints \
        --dataset_dir /scratch/cluster/nimit/data/carla \
        --hidden_size 32 \
        --num_layers 6 \
        --num_heads 8 \
        --batch_size 1024 \
        --map_size 25 \
        --lr 5e-05 \
        --weight_decay 0.0
