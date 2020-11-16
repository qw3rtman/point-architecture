import numpy as np
from pathlib import Path
from datetime import datetime

jobs = list()
unique = datetime.now().strftime("%-m.%d")

for split_dir in Path('/scratch/cluster/nimit/data/carla').iterdir():
    if not split_dir.is_dir():
        continue
    for dataset_dir in split_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        job = f"""
python -c "import consistency.point_dataset; consistency.point_dataset.get_episode('{dataset_dir}')"
"""

        jobs.append(job)
        print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
