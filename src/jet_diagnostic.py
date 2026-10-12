import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

TAG = "ae_z8_s0"  # any finished run in results/runs/
names = json.load(open("data/processed/feature_names.json"))
raw = np.load("data/processed/X_raw.npy")
sp = np.load("data/processed/splits.npz")
jet = raw[:, names.index("PRI_jet_num")].astype(int)
r = np.load(f"results/runs/{TAG}.npz")

rows = []
for j in range(4):
    bg = jet[sp["test"]] == j
    sg = jet[sp["sig_eval"]] == j
    y = np.r_[np.zeros(bg.sum()), np.ones(sg.sum())]
    s = np.r_[r["s_bg"][bg], r["s_sig"][sg]]
    rows.append({"jets": j,
                 "bg_share": bg.mean(), "sig_share": sg.mean(),
                 "bg_mean_score": r["s_bg"][bg].mean(),
                 "sig_mean_score": r["s_sig"][sg].mean(),
                 "auc_within_bin": roc_auc_score(y, s)})
print(pd.DataFrame(rows).round(4).to_string(index=False))