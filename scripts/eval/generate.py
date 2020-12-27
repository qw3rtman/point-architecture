import os
from pathlib import Path
import shutil
import argparse

if 'PYTHONHASHSEED' not in os.environ or os.environ['PYTHONHASHSEED'] != '0':
    print('[!] Set PYTHONHASHSEED=0 first!!\n\n')

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=Path, required=True)
args = parser.parse_args()

wandb_root = Path('/scratch/cluster/nimit/wandb')
wandb_id = str(hash(args.model.parent.name))
print(args.model.parent.stem, wandb_id)
config = list(wandb_root.glob(f'*{wandb_id}'))[0]/'config.yaml' # first is fine, config doesn't change
shutil.copy(config, args.model.parent)

SUBMIT = """Executable = run_{uuid}.sh

+Group="GRAD"
+Project="AI_ROBOTICS"
+ProjectDescription="Training model"
+GPUJob=true

Requirements=(TARGET.GPUSlot)
Rank=memory
Universe=vanilla
Getenv=True
Notification=Complete

Log=/u/nimit/logs/$(ClusterId)_{uuid}.log
Output=/u/nimit/logs/$(ClusterId)_{uuid}.out
Error=/u/nimit/logs/$(ClusterId)_{uuid}.err

Queue 1
"""


TRAIN = """#!/bin/zsh
source $HOME/.zshrc

export WANDB_DIR=/scratch/cluster/nimit
cd $HOME/Documents/robomaster/2020_CARLA_challenge

conda env list
conda activate /scratch/cluster/nimit/miniconda3/envs/robo
"""

from params import get_jobs
for job in get_jobs(args.model):
    experiment = Path('.').resolve().name
    uuid = '{}_{}'.format(experiment, (len(list(Path('.').glob('*'))) - 1) // 2)
    print(uuid)

    Path('submit_%s.sh' % uuid).write_text(SUBMIT.format(uuid=uuid))
    Path('run_%s.sh' % uuid).write_text(TRAIN + job)

    os.chmod('run_%s.sh' % uuid, 509)

    print(uuid)
