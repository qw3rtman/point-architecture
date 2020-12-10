import numpy as np
from pathlib import Path
from datetime import datetime

jobs = list()
unique = datetime.now().strftime("%-m.%d")

for batch_size in [1024]:
    for hidden_size in [32, 64]:
        for lr in [5e-5, 2e-4]: # smaller is better (5e-3 did not work)
            for weight_decay in [0.]:
                job = f"""PYTHONHASHSEED=0 python -m consistency.train_il \\
    --description {unique}-v1 \\
    --max_epoch 2000 \\
    --checkpoint_dir /scratch/cluster/nimit/checkpoints \\
    --dataset_dir /scratch/cluster/nimit/data/carla \\
    --hidden_size {hidden_size} \\
    --num_layers 6 \\
    --num_heads 8 \\
    --batch_size {batch_size} \\
    --lr {lr} \\
    --weight_decay {weight_decay}
"""

                jobs.append(job)
                print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
