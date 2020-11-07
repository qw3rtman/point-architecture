#!/bin/zsh
source $HOME/.zshrc

export WANDB_DIR=/scratch/cluster/nimit
cd $HOME/Documents/robomaster

conda env list
conda activate /scratch/cluster/nimit/miniconda3/envs/robo
python -m consistency.train_il \
    --description 11.06-v6 \
    --max_epoch 200 \
    --checkpoint_dir /scratch/cluster/nimit/checkpoints \
    --dataset_dir /scratch/cluster/bzhou/data/highway_v3 \
    --hidden_size 128 \
    --num_layers 4 \
    --num_heads 2 \
    --batch_size 128 \
    --lr 0.0002 \
    --weight_decay 3.8e-07
