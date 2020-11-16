#!/bin/zsh
source $HOME/.zshrc

export WANDB_DIR=/scratch/cluster/nimit
cd $HOME/Documents/robomaster

conda env list
conda activate /scratch/cluster/nimit/miniconda3/envs/robo
PYTHONHASHSEED=0 python -m consistency.train_il \
    --description 11.29-v1 \
    --max_epoch 500 \
    --checkpoint_dir /scratch/cluster/nimit/checkpoints \
    --dataset_dir /scratch/cluster/nimit/data/carla \
    --hidden_size 32 \
    --num_layers 6 \
    --num_heads 8 \
    --batch_size 1024 \
    --lr 0.0002 \
    --weight_decay 0.0
