"""Print the derived numbers quoted in the paper text.

Reads only the result tables in results/ (no data or model files needed),
so every number in the Results and Discussion sections can be checked.
Run from the project root:  python src/paper_numbers.py
"""
import pandas as pd
from scipy.stats import spearmanr

R = "results"
r = pd.read_csv(f"{R}/results.csv")
st = pd.read_csv(f"{R}/stratified.csv")
bp = pd.read_csv(f"{R}/bootstrap_pct.csv")
sh = pd.read_csv(f"{R}/shift.csv")


def section(title):
    print(f"\n== {title}")


# 1. Training budget: runs that stopped at the 300-epoch cap
section("Runs that reached the 300-epoch cap")
capped = r[(r.model != "pca") & (r.epochs >= 300)]
print(capped.groupby("model").size().to_dict(), "total", len(capped))

# 2. Reconstruction error vs AUC (Spearman rank correlation)
section("Spearman: validation reconstruction error vs AUC")
pct = st[st.group == "pct_global"][["model", "latent", "seed", "auc"]]
m = r.merge(pct, on=["model", "latent", "seed"], suffixes=("_raw", "_pct"))
for name, d in [("all runs", m), ("AE only", m[m.model == "ae"])]:
    for col in ["auc_raw", "auc_pct"]:
        rho, p = spearmanr(d.val_recon, d[col])
        print(f"{name:9s} {col}: rho={rho:.2f} p={p:.3g} n={len(d)}")

# 3. Raw vs corrected, configuration means
section("Configuration means (raw and corrected)")
raw = r.groupby(["model", "latent"])[["auc", "eff_at_fpr_0.01"]].mean()
cor = pct.groupby(["model", "latent"]).auc.mean()
print("raw AUC range:", raw.auc.min().round(3), "to", raw.auc.max().round(3))
print("raw eff@1% range (%):", (100 * raw["eff_at_fpr_0.01"].min()).round(2),
      "to", (100 * raw["eff_at_fpr_0.01"].max()).round(2))
drop = (raw.auc - cor)
print("AUC drop from correction:", drop.min().round(3), "to", drop.max().round(3))
nn = cor.drop("pca", level="model")
print("AE/VAE corrected mean AUC range:", nn.min().round(3), "to", nn.max().round(3))

# 4. Bootstrap intervals vs chance (corrected score)
section("Bootstrap 95% intervals vs AUC = 0.5")
bp["side"] = "includes 0.5"
bp.loc[bp.auc_hi < 0.5, "side"] = "below"
bp.loc[bp.auc_lo > 0.5, "side"] = "above"
print(bp.groupby(["model", "side"]).size())
below_all = pct[pct.model != "pca"].assign(b=lambda d: d.auc < 0.5) \
    .groupby(["model", "latent"]).b.sum()
print("configurations with all 3 seeds below 0.5:",
      list(below_all[below_all == 3].index))

# 5. Within single jet counts
section("Within-jet AUC ranges (configuration means)")
pj = st[st.group.str.startswith("jets")].groupby(["group", "model", "latent"]).auc.mean()
for g in ["jets0", "jets3"]:
    print(g, pj[g].min().round(3), "to", pj[g].max().round(3))

# 6. Synthetic shift null case
section("Shift study null case (s = 0)")
z = sh[sh["shift"] == 0]
print("AUC mean/min/max:", z.auc.mean().round(3), z.auc.min().round(3), z.auc.max().round(3))
e = 100 * z["eff_at_fpr_0.01"]
print("eff@1% (%) mean/min/max:", e.mean().round(2), e.min().round(2), e.max().round(2))
