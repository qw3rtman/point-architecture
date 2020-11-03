import numpy as np
from pathlib import Path
from datetime import datetime

jobs = list()
unique = datetime.now().strftime("%-m.%d")

for batch_size in [64, 128]:
    for hidden_size in [128, 256]:
        for lr in [2e-4]:
            for weight_decay in [3.8e-7]:
                job = f"""python -m consistency.train_il \\
    --description {unique}-v5 \\
    --max_epoch 200 \\
    --checkpoint_dir /scratch/cluster/nimit/checkpoints \\
    --dataset_dir /scratch/cluster/bzhou/data/highway_v3 \\
    --hidden_size {hidden_size} \\
    --batch_size {batch_size} \\
    --lr {lr} \\
    --weight_decay {weight_decay}
"""

                jobs.append(job)
                print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
