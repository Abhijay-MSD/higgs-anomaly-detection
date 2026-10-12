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
d = X.shape[1]
X_train = X[sp["train"]]

# Split held-out background in two random halves: A stays normal, B gets shifted
rng = np.random.default_rng(0)
test_idx = rng.permutation(sp["test"])
half = len(test_idx) // 2
X_a = X[test_idx[:half]]
X_b = X[test_idx[half:2 * half]]

# Only shift original features that are defined for every event (no -999),
# and skip the jet count, which is a category
always = [i for i in range(30)
          if not np.isnan(raw[:, i]).any() and names[i] != "PRI_jet_num"]
print(len(always), "features can be shifted:", [names[i] for i in always])

SHIFTS = [0.0, 0.25, 0.5, 1.0, 2.0]      # in standard deviations; 0.0 is a null check
N_FEATS = [1, 3, 10, len(always)]
N_SUBSETS = 3

# The same random feature subsets are used for every model
sub_rng = np.random.default_rng(1)
subsets = []
for n in N_FEATS:
    reps = 1 if n == len(always) else N_SUBSETS
    for sid in range(reps):
        subsets.append((n, sid, sub_rng.choice(always, size=n, replace=False)))

pca_cache = {}


def get_scorer(kind, latent, seed):
    if kind == "pca":
        if latent not in pca_cache:
            pca_cache[latent] = PCA(n_components=latent).fit(X_train.numpy())
        pca = pca_cache[latent]

        def score(A):
            A = A.numpy()
            return ((A - pca.inverse_transform(pca.transform(A))) ** 2).mean(axis=1)
        return score
    model = AE(d, latent) if kind == "ae" else VAE(d, latent)
    model.load_state_dict(torch.load(R / "runs" / f"{kind}_z{latent}_s{seed}.pt"))
    model.eval()
    return lambda A: model.score(A).numpy()


runs = pd.read_csv(R / "results.csv")
out = []
for _, r in runs.iterrows():
    kind, latent, seed = r["model"], int(r["latent"]), int(r["seed"])
    score = get_scorer(kind, latent, seed)
    s_a = score(X_a)
    for n, sid, cols in subsets:
        for k in SHIFTS:
            X_s = X_b.clone()
            X_s[:, cols] += k
            res = evaluate(s_a, score(X_s))
            out.append({"model": kind, "latent": latent, "seed": seed,
                        "n_feat": n, "subset": sid, "shift": k, **res})
    print("done", kind, latent, seed)

df = pd.DataFrame(out)
df.to_csv(R / "shift.csv", index=False)

for n in N_FEATS:
    print(f"\nAUC, {n} feature(s) shifted (mean over seeds and subsets)")
    print(df[df.n_feat == n].pivot_table(index=["latent", "model"], columns="shift",
                                         values="auc").round(3).to_string())
for n in (3, len(always)):
    print(f"\nSignal caught at 1% false alarms, {n} feature(s) shifted")
    print(df[df.n_feat == n].pivot_table(index=["latent", "model"], columns="shift",
                                         values="eff_at_fpr_0.01").round(4).to_string())