"""Diagnostics for the review: (1) background mean score per jet count, full vs masked
reconstruction error; (2) which features signal events are rebuilt better on, within each
jet count (AE k=8 seed 0 and PCA k=16)."""
import json
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.decomposition import PCA
from models import AE
D, R = Path("data/processed"), Path("results")
torch.set_num_threads(1)
X = np.load(D/"X_scaled.npy"); raw = np.load(D/"X_raw.npy"); names = json.load(open(D/"feature_names.json"))
sp = np.load(D/"splits.npz"); jet = raw[:, names.index("PRI_jet_num")].astype(int)
defined = np.c_[~np.isnan(raw), np.zeros((len(raw), 3), bool)]
m = AE(33, 8); m.load_state_dict(torch.load(R/"runs"/"ae_z8_s0.pt")); m.eval()
p = PCA(n_components=16).fit(X[sp["train"]])
def err(kind, A):
    if kind == "ae":
        with torch.no_grad(): return ((m(torch.tensor(A))[0].numpy() - A) ** 2)
    return (p.inverse_transform(p.transform(A)) - A) ** 2
rows, feat = [], []
for kind in ("ae", "pca"):
    for j in range(4):
        b = sp["test"][jet[sp["test"]] == j]; s = sp["sig_eval"][jet[sp["sig_eval"]] == j]
        eb, es = err(kind, X[b]), err(kind, X[s])
        mb, ms = defined[b], defined[s]
        rows.append({"model": kind, "jets": j, "bg_full": eb.mean(1).mean(), "sig_full": es.mean(1).mean(),
                     "bg_masked": ((eb*mb).sum(1)/mb.sum(1)).mean(), "sig_masked": ((es*ms).sum(1)/ms.sum(1)).mean(),
                     "n_defined": mb.sum(1).mean()})
        diff = es.mean(0) - eb.mean(0)
        for i in np.argsort(diff)[:3]:
            feat.append({"model": kind, "jets": j, "feature": names[i], "sig_minus_bg": diff[i]})
print(pd.DataFrame(rows).round(4).to_string()); print(pd.DataFrame(feat).round(4).to_string())
pd.DataFrame(rows).to_csv(R/"revision"/"J_jet_masked_means.csv", index=False)
pd.DataFrame(feat).to_csv(R/"revision"/"J_features_signal_lower.csv", index=False)
