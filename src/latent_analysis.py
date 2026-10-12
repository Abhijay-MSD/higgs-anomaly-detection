"""Latent-space analysis: what do the learned codes actually contain?

For every trained model (PCA, AE, VAE) we encode events and ask, with simple
linear probes:
  1. Which physical features can be read out of the code? (ridge regression, R^2)
  2. Can the jet count be read out? (logistic regression, accuracy)
  3. Is the proxy signal linearly visible in the code? (logistic regression)
     This probe is SUPERVISED (it uses sig_clf), so it is a reference, not an
     anomaly detector. Signal never touches PCA/AE/VAE training.
Probes are fit on validation background (plus sig_clf for probe 3) and scored on
test background (plus sig_eval). Also saves small latent samples for the plots.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from evaluation import evaluate
from models import AE, VAE

D, R = Path("data/processed"), Path("results")
LATENTS, SEEDS = [2, 4, 8, 16], [0, 1, 2]
N_PLOT = 5000  # events per class saved for the 2D latent plots

# ---------- load data (same files every other script uses) ----------
X = np.load(D / "X_scaled.npy").astype(np.float32)
raw = np.load(D / "X_raw.npy")
names = json.load(open(D / "feature_names.json"))
sp = np.load(D / "splits.npz")
tr, va, te, sc, se = (sp[k] for k in ("train", "val", "test", "sig_clf", "sig_eval"))
jet = raw[:, names.index("PRI_jet_num")].astype(int)
d = X.shape[1]
X_tr, X_va, X_te, X_sc, X_se = (X[i] for i in (tr, va, te, sc, se))

# probe targets: features that are always defined (no -999), jet count handled apart
defined = ~np.isnan(raw).any(axis=0)
targets = [i for i in range(raw.shape[1]) if defined[i] and names[i] != "PRI_jet_num"]
Y_va, Y_te = X_va[:, targets], X_te[:, targets]   # quantile-normal version of each feature
jet_majority = np.bincount(jet[te]).max() / len(te)


def check_alignment():
    """Scaled column i must be a monotone transform of raw column i."""
    idx = te[:20000]
    for i in targets:
        rho = spearmanr(raw[idx, i], X[idx, i])[0]
        assert rho > 0.95, f"column mismatch for {names[i]} (rho={rho:.2f})"


def get_encoder(kind, latent, seed):
    """Return a function that maps scaled events (numpy) to latent codes (numpy)."""
    if kind == "pca":
        return PCA(n_components=latent).fit(X_tr).transform
    model = AE(d, latent) if kind == "ae" else VAE(d, latent)
    model.load_state_dict(torch.load(R / "runs" / f"{kind}_z{latent}_s{seed}.pt"))
    model.eval()

    @torch.no_grad()
    def encode(a):
        out = []
        for i in range(0, len(a), 65536):
            xb = torch.from_numpy(a[i:i + 65536])
            z = model(xb)[1] if kind == "ae" else model.encode(xb)[0]  # VAE: mean mu
            out.append(z.numpy())
        return np.concatenate(out)
    return encode


def feature_probe(A, B):
    """Ridge R^2 per physical feature: fit on val background, score on test background."""
    ridge = RidgeCV(alphas=(0.1, 1, 10, 100)).fit(A, Y_va)
    return r2_score(Y_te, ridge.predict(B), multioutput="raw_values")


def jet_probe(A, B):
    """Accuracy of reading the jet count (0 to 3) from the code."""
    return LogisticRegression(max_iter=500).fit(A, jet[va]).score(B, jet[te])


def within_jet_auc(s_bg, s_sig):
    """AUC inside each jet bin, averaged with weights = signal events per bin."""
    aucs, ws = [], []
    for j in range(4):
        b, g = jet[te] == j, jet[se] == j
        if b.sum() == 0 or g.sum() == 0:
            continue
        y = np.r_[np.zeros(b.sum()), np.ones(g.sum())]
        aucs.append(roc_auc_score(y, np.r_[s_bg[b], s_sig[g]]))
        ws.append(g.sum())
    return float(np.average(aucs, weights=ws))


def signal_probe(enc):
    """Supervised reference: can a linear model see the proxy signal in the code?"""
    Z_va, Z_sc = enc(X_va), enc(X_sc)
    Z_te, Z_se = enc(X_te), enc(X_se)
    A = np.vstack([Z_va, Z_sc])
    y = np.r_[np.zeros(len(Z_va)), np.ones(len(Z_sc))]
    scaler = StandardScaler().fit(A)
    clf = LogisticRegression(max_iter=500).fit(scaler.transform(A), y)
    s_bg = clf.decision_function(scaler.transform(Z_te))
    s_sig = clf.decision_function(scaler.transform(Z_se))
    out = evaluate(s_bg, s_sig)
    return out["auc"], out["eff_at_fpr_0.01"], within_jet_auc(s_bg, s_sig)


def save_latents(kind, latent, enc):
    """Small fixed sample of codes (test background + eval signal) for the plots."""
    rng = np.random.default_rng(0)
    n = min(N_PLOT, len(te), len(se))
    ib = rng.choice(te, n, replace=False)
    ig = rng.choice(se, n, replace=False)
    z = enc(np.vstack([X[ib], X[ig]]))
    np.savez(R / "latents" / f"{kind}_z{latent}_s0.npz", z=z,
             is_signal=np.r_[np.zeros(n), np.ones(n)],
             jet=np.r_[jet[ib], jet[ig]])


def main():
    check_alignment()
    (R / "latents").mkdir(parents=True, exist_ok=True)
    feat_rows, sum_rows = [], []
    for kind in ("pca", "ae", "vae"):
        for latent in LATENTS:
            for seed in ([0] if kind == "pca" else SEEDS):
                enc = get_encoder(kind, latent, seed)
                scaler = StandardScaler().fit(enc(X_va))
                A, B = scaler.transform(enc(X_va)), scaler.transform(enc(X_te))
                r2 = feature_probe(A, B)
                auc, eff, auc_jet = signal_probe(enc)
                # VAE only: a dimension is "active" if its mean code varies across events
                active = int((enc(X_te).var(axis=0) > 0.01).sum()) if kind == "vae" else np.nan
                sum_rows.append(dict(model=kind, latent=latent, seed=seed, mean_r2=r2.mean(),
                                     jet_acc=jet_probe(A, B), jet_majority=jet_majority,
                                     sig_auc=auc, sig_eff_1pct=eff, sig_auc_within_jet=auc_jet,
                                     active_dims=active))
                feat_rows += [dict(model=kind, latent=latent, seed=seed, feature=names[i], r2=v)
                              for i, v in zip(targets, r2)]
                if seed == 0:
                    save_latents(kind, latent, enc)
                print("done", kind, latent, seed, flush=True)

    # reference: the same signal probe on the raw 33 inputs (no compression at all)
    auc, eff, auc_jet = signal_probe(lambda a: a)
    sum_rows.append(dict(model="input", latent=d, seed=0, sig_auc=auc,
                         sig_eff_1pct=eff, sig_auc_within_jet=auc_jet))

    summ, feats = pd.DataFrame(sum_rows), pd.DataFrame(feat_rows)
    summ.to_csv(R / "latent_summary.csv", index=False)
    feats.to_csv(R / "latent_features.csv", index=False)

    cols = ["mean_r2", "jet_acc", "sig_auc", "sig_auc_within_jet", "sig_eff_1pct"]
    print("\nMean over seeds:")
    print(summ.groupby(["latent", "model"])[cols].mean().round(3).to_string())
    print("\njet_majority (always guess the commonest jet count):", round(jet_majority, 3))
    print("\nVAE active dimensions (out of latent):")
    print(summ[summ.model == "vae"].groupby("latent").active_dims.mean().round(1).to_string())
    print("\nR^2 per feature, latent 8, mean over seeds:")
    f8 = feats[feats.latent == 8].groupby(["feature", "model"]).r2.mean().unstack()
    print(f8.round(2).to_string())


if __name__ == "__main__":
    main()