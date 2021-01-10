import pandas as pd
import json
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=Path, required=True)
parser.add_argument('--export', action='store_true')
args = parser.parse_args()

data, indices = [], []
for stats_f in args.model.parent.glob(f'{args.model.stem}-*.txt'):
    with open(stats_f, 'r') as f:
        stats = json.load(f)
    if 'values' not in stats or len(stats['values']) == 0:
        continue
    else:
        labels = stats['labels']

    #route = int(stats_f.stem.split('-')[1].split('_')[1])
    indices.append(stats_f.stem.split('-')[1])
    data.append(dict(zip(stats['labels'], map(float, stats['values']))))

df = pd.DataFrame(data, index=indices).sort_index()
print(df[['Avg. driving score', 'Avg. route completion']])
print()
print(f"{labels[0]}: {df[labels[0]].mean():.02f}")
print(f"{labels[1]}: {df[labels[1]].mean():.02f}")

if args.export:
    csv_f = args.model.parent / f'{args.model.parent.stem}-{args.model.stem}.csv'
    df.to_csv(csv_f)
    print(csv_f)
