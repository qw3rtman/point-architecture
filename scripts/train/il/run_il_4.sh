#!/bin/zsh
source $HOME/.zshrc

export WANDB_DIR=/scratch/cluster/nimit
cd $HOME/Documents/robomaster

conda env list
conda activate /scratch/cluster/nimit/miniconda3/envs/robo
PYTHONHASHSEED=0 python -m consistency.train_il \
    --description 11.07-v8 \
    --max_epoch 500 \
    --checkpoint_dir /scratch/cluster/nimit/checkpoints \
    --dataset_dir /scratch/cluster/bzhou/data/highway_v3 \
    --hidden_size 128 \
    --num_layers 12 \
    --num_heads 8 \
    --batch_size 64 \
    --lr 0.0002 \
    --weight_decay 3.8e-07
