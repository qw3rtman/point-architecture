import numpy as np
from pathlib import Path
from datetime import datetime

jobs = list()
unique = datetime.now().strftime("%-m.%d")

for batch_size in [64, 128]:
    for hidden_size in [512]:
        for lr in [5e-3, 2e-4]:
            for weight_decay in [0., 3.8e-7]:
                job = f"""PYTHONHASHSEED=0 python -m consistency.train_il \\
    --description {unique}-v7 \\
    --max_epoch 500 \\
    --checkpoint_dir /scratch/cluster/nimit/checkpoints \\
    --dataset_dir /scratch/cluster/bzhou/data/highway_v3 \\
    --hidden_size {hidden_size} \\
    --num_layers 12 \\
    --num_heads 8 \\
    --batch_size {batch_size} \\
    --lr {lr} \\
    --weight_decay {weight_decay}
"""

                jobs.append(job)
                print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
