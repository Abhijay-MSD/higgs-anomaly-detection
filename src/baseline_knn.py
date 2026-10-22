"""k-nearest-neighbour baseline (peer-review request): the anomaly score of an event is its
distance to the 10th nearest of 50,000 randomly chosen training-background events
(distance-based outliers, Ramaswamy, Rastogi and Shim 2000). Reported raw and with the
jet-count correction; appends a row to results/revision/D_baselines.csv."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from evaluation import evaluate

D, R = Path("data/processed"), Path("results")
X = np.load(D / "X_scaled.npy").astype(np.float32)
raw = np.load(D / "X_raw.npy")
names = json.load(open(D / "feature_names.json"))
sp = np.load(D / "splits.npz")
jet = raw[:, names.index("PRI_jet_num")].astype(int)

rng = np.random.default_rng(123)
ref = X[rng.choice(sp["train"], 50_000, replace=False)]
nn = NearestNeighbors(n_neighbors=10).fit(ref)
sc = {k: nn.kneighbors(X[sp[k]])[0][:, -1] for k in ("val", "test", "sig_eval")}

jv, jb, js = jet[sp["val"]], jet[sp["test"]], jet[sp["sig_eval"]]
pb, ps = np.zeros(len(sc["test"])), np.zeros(len(sc["sig_eval"]))
for j in range(4):
    v = np.sort(sc["val"][jv == j])
    pb[jb == j] = np.searchsorted(v, sc["test"][jb == j], side="right") / len(v)
    ps[js == j] = np.searchsorted(v, sc["sig_eval"][js == j], side="right") / len(v)
r, c = evaluate(sc["test"], sc["sig_eval"]), evaluate(pb, ps)
row = {"baseline": "distance to 10th nearest neighbour", "raw_auc": r["auc"],
       "raw_eff": r["eff_at_fpr_0.01"], "cor_auc": c["auc"], "cor_eff": c["eff_at_fpr_0.01"]}
f = R / "revision" / "D_baselines.csv"
d = pd.read_csv(f)
d = pd.concat([d[~d.baseline.str.contains("neighbour|kNN")], pd.DataFrame([row])])
d.to_csv(f, index=False)
print(d.to_string())
