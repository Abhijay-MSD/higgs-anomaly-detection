import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from evaluation import evaluate
from models import AE, VAE

D = Path("data/processed")
R = Path("results")

X = torch.tensor(np.load(D / "X_scaled.npy"))
raw = np.load(D / "X_raw.npy")
sp = np.load(D / "splits.npz")
names = json.load(open(D / "feature_names.json"))
jet = raw[:, names.index("PRI_jet_num")].astype(int)

X_train, X_val = X[sp["train"]], X[sp["val"]]
jet_val, jet_bg, jet_sig = jet[sp["val"]], jet[sp["test"]], jet[sp["sig_eval"]]
d = X.shape[1]
runs = pd.read_csv(R / "results.csv")


def val_scores(kind, latent, seed):
    """Anomaly scores for validation background, from the saved model."""
    if kind == "pca":
        pca = PCA(n_components=latent).fit(X_train.numpy())
        A = X_val.numpy()
        return ((A - pca.inverse_transform(pca.transform(A))) ** 2).mean(axis=1)
    model = AE(d, latent) if kind == "ae" else VAE(d, latent)
    model.load_state_dict(torch.load(R / "runs" / f"{kind}_z{latent}_s{seed}.pt"))
    model.eval()
    return model.score(X_val).numpy()


def stratified(s_val, s_bg, s_sig):
    # unstratified reference, same as the main grid
    rows = [{"group": "raw_global", **evaluate(s_bg, s_sig)}]
    pct_bg = np.zeros(len(s_bg))
    pct_sig = np.zeros(len(s_sig))
    for j in range(4):
        v = np.sort(s_val[jet_val == j])          # validation background only
        mb, ms = jet_bg == j, jet_sig == j
        # 1. metrics inside one jet bin
        rows.append({"group": f"jets{j}", **evaluate(s_bg[mb], s_sig[ms])})
        # 2. percentile of each score among normal events with the same jet count
        pct_bg[mb] = np.searchsorted(v, s_bg[mb], side="right") / len(v)
        pct_sig[ms] = np.searchsorted(v, s_sig[ms], side="right") / len(v)
    rows.append({"group": "pct_global", **evaluate(pct_bg, pct_sig)})
    return rows


out = []
for _, r in runs.iterrows():
    kind, latent, seed = r["model"], int(r["latent"]), int(r["seed"])
    scores = np.load(R / "runs" / f"{kind}_z{latent}_s{seed}.npz")
    s_val = val_scores(kind, latent, seed)
    for row in stratified(s_val, scores["s_bg"], scores["s_sig"]):
        out.append({"model": kind, "latent": latent, "seed": seed, **row})
    print("done", kind, latent, seed)

df = pd.DataFrame(out)
df.to_csv(R / "stratified.csv", index=False)

order = ["raw_global", "pct_global", "jets0", "jets1", "jets2", "jets3"]
for metric in ("auc", "eff_at_fpr_0.01"):
    print("\n", metric, "(mean over seeds)")
    tab = df.pivot_table(index=["latent", "model"], columns="group",
                         values=metric, aggfunc="mean")[order]
    print(tab.round(4).to_string())