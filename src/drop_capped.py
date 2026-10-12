from pathlib import Path

import pandas as pd

df = pd.read_csv("results/results.csv")
capped = df[(df.model != "pca") & (df.epochs >= 300)]
print(capped[["model", "latent", "seed", "epochs"]])

for _, r in capped.iterrows():
    for ext in ("npz", "pt"):
        Path(f"results/runs/{r.model}_z{r.latent}_s{r.seed}.{ext}").unlink(missing_ok=True)

df.drop(capped.index).to_csv("results/results.csv", index=False)
print("removed", len(capped), "capped runs")