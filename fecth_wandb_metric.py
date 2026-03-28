import wandb
import pandas as pd
import os

entity = "lichenqi-university-of-regina"
project = "ipditm"
group = "welfare-WelfareShaperAtt-vs-PPO_memory-mean_ref"
output_dir = "wandb_exports"

api = wandb.Api()
runs = api.runs(
    f"{entity}/{project}",
    filters={"group": group},
)

print(f"Project: {project}")
print(f"Group: {group}")
print(f"Total runs found: {len(runs)}")

os.makedirs(output_dir, exist_ok=True)

all_runs = []
for run in runs:
    print(f"Fetching {run.name} ({run.id})...")
    history = run.history(samples=10000, pandas=True)
    history["run_name"] = run.name
    history["run_id"] = run.id
    history["seed"] = run.config.get("seed", None)
    all_runs.append(history)

    # Save individual run
    history.to_csv(f"{output_dir}/{run.name}_{run.id}.csv", index=False)

# Save combined CSV
df = pd.concat(all_runs, ignore_index=True)
df.to_csv(f"{output_dir}/all_runs.csv", index=False)

print(f"\nSaved {len(all_runs)} runs to {output_dir}/")
print(f"Combined CSV: {output_dir}/all_runs.csv")
print(f"Columns: {list(df.columns)}")