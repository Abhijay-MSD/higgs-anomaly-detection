"""Correlated (energy-scale) shifts, a harder test than independent constant shifts.

Instead of adding a constant to independent features in the scaled space, every
energy-like quantity (transverse momenta, masses, missing and summed transverse energy)
is multiplied by a common factor in physical units, before the median fill and the
quantile transform. Angles, pseudorapidities, ratios, centralities and the jet count are
untouched, so the shifted events keep the correlations of real events and change the way
a miscalibrated energy scale would. Two versions: all energy-like features, and jet
quantities only (a jet energy scale). The held-out background is split in two halves as in
shift_study.py: half A is the unshifted reference, half B is shifted.
Metrics are reported raw and with the jet-count correction (percentile among validation
background events with the same jet count).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer

from evaluation import evaluate
from models import AE, VAE

D, R = Path("data/processed"), Path("results")
OUT = R / "revision"
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)

X = np.load(D / "X_scaled.npy").astype(np.float32)
raw = np.load(D / "X_raw.npy").astype(np.float64)
names = json.load(open(D / "feature_names.json"))
sp = np.load(D / "splits.npz")
jet = raw[:, names.index("PRI_jet_num")].astype(int)
d = X.shape[1]

# rebuild the preprocessing exactly as preprocess.py does, then check it matches
tr = sp["train"]
med = np.nanmedian(raw[tr], axis=0)
qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                         subsample=10**6, random_state=0)
qt.fit(np.where(np.isnan(raw[tr]), med, raw[tr]))


def prep(R_):
    filled = np.where(np.isnan(R_), med, R_)
    ind = np.c_[np.isnan(R_[:, names.index("DER_mass_jet_jet")]),
                np.isnan(R_[:, names.index("PRI_jet_leading_pt")]),
                np.isnan(R_[:, names.index("DER_mass_MMC")])]
    return np.hstack([qt.transform(filled), ind]).astype(np.float32)


chk = sp["test"][:20000]
print("max |rebuilt - saved| on 20k test events:", float(np.abs(prep(raw[chk]) - X[chk]).max()), flush=True)

ENERGY = [n for n in names[:30] if any(t in n for t in ("_pt", "mass", "_met", "sumet", "sum_pt"))
          and n not in ("DER_pt_ratio_lep_tau", "DER_met_phi_centrality")
          and "phi" not in n and "eta" not in n]
JETS = [n for n in ENERGY if "jet" in n]
print("energy-like:", ENERGY, "\njet-only:", JETS, flush=True)

rng = np.random.default_rng(0)
test_idx = rng.permutation(sp["test"])
half = len(test_idx) // 2
ia, ib = test_idx[:half], test_idx[half:2 * half]
FACTORS = [1.0, 1.02, 1.05, 1.10, 1.20]
shifted = {}
for label, cols in (("all energies", ENERGY), ("jets only", JETS)):
    for f in FACTORS:
        Rb = raw[ib].copy()
        for n in cols:
            Rb[:, names.index(n)] *= f
        shifted[(label, f)] = prep(Rb)

pca_cache = {}


def scorer(kind, latent, seed):
    if kind == "pca":
        if latent not in pca_cache:
            pca_cache[latent] = PCA(n_components=latent).fit(X[tr])
        p = pca_cache[latent]
        return lambda A: ((A - p.inverse_transform(p.transform(A))) ** 2).mean(1)
    m = AE(d, latent) if kind == "ae" else VAE(d, latent)
    m.load_state_dict(torch.load(R / "runs" / f"{kind}_z{latent}_s{seed}.pt"))
    m.eval()
    return lambda A: m.score(torch.tensor(A)).numpy()


def corrected(sv, sa, sb, jv, ja, jb):
    pa, pb = np.zeros(len(sa)), np.zeros(len(sb))
    for j in range(4):
        v = np.sort(sv[jv == j])
        pa[ja == j] = np.searchsorted(v, sa[ja == j], side="right") / len(v)
        pb[jb == j] = np.searchsorted(v, sb[jb == j], side="right") / len(v)
    return evaluate(pa, pb)


runs = pd.read_csv(R / "results.csv")
rows = []
jv, ja, jbb = jet[sp["val"]], jet[ia], jet[ib]
for _, r in runs.iterrows():
    kind, latent, seed = r["model"], int(r["latent"]), int(r["seed"])
    sc = scorer(kind, latent, seed)
    sv, sa = sc(X[sp["val"]]), sc(X[ia])
    for (label, f), Xb in shifted.items():
        sb = sc(Xb)
        rm, cm = evaluate(sa, sb), corrected(sv, sa, sb, jv, ja, jbb)
        rows.append({"model": kind, "latent": latent, "seed": seed, "shift": label, "factor": f,
                     "raw_auc": rm["auc"], "raw_eff": rm["eff_at_fpr_0.01"],
                     "cor_auc": cm["auc"], "cor_eff": cm["eff_at_fpr_0.01"]})
    print("done", kind, latent, seed, flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / "I_energy_shift.csv", index=False)
print(df.pivot_table(index=["latent", "model"], columns=["shift", "factor"], values="raw_eff").round(3).to_string())
