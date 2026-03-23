import wandb
import numpy as np

entity = "lichenqi-university-of-regina"
project = "cg"
metric = "train/reward_per_timestep/player_2"   # change this to your exact column name in the Runs table

api = wandb.Api()
runs = api.runs(f"{entity}/{project}")

vals = []
for run in runs:
    if metric in run.summary:
        v = run.summary[metric]
        if isinstance(v, (int, float)):
            vals.append(float(v))

vals = np.array(vals, dtype=float)

print("n =", len(vals))
print("mean =", vals.mean())
print("stddev =", vals.std(ddof=1))   # sample standard deviation