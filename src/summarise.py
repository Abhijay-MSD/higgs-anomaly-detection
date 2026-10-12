import pandas as pd

df = pd.read_csv("results/results.csv")
cols = ["auc", "eff_at_fpr_0.01", "eff_at_fpr_0.001", "val_recon"]
print(df.groupby(["model", "latent"])[cols].agg(["mean", "std"]).round(4).to_string())