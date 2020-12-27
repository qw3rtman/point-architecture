#!/bin/zsh
source $HOME/.zshrc

export WANDB_DIR=/scratch/cluster/nimit
cd $HOME/Documents/robomaster/2020_CARLA_challenge

conda env list
conda activate /scratch/cluster/nimit/miniconda3/envs/robo
ROUTES=/u/nimit/Documents/robomaster/2020_CARLA_challenge/leaderboard/data/routes_testing/route_11.xml TEAM_CONFIG=/scratch/cluster/nimit/checkpoints/1024_45_32_6_8_0.0002_0.0_12.26-cf-v1/model_latest.t7 ./run_model