"""Retrain without direct jet-count inputs (peer-review request).

Drops PRI_jet_num and the two jet indicators (ind_lt2_jets, ind_0_jets), leaving 30
inputs (29 transformed features + the DER_mass_MMC indicator). PCA at every latent size,
and the AE and VAE at latent size 8 with seeds 0, 1 and 2, are trained exactly as in
run_grid.py (Adam, lr 1e-3, batch 512, patience 15, cap 300 epochs). The jet count is
still used, from the raw data, for the jet-count correction and the diagnostics.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from evaluation import evaluate
from models import AE, VAE, ae_loss, make_vae_loss, set_seed, train

D, R = Path("data/processed"), Path("results")
OUT = R / "revision"
(OUT / "runs_nojet").mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)

names = json.load(open(D / "feature_names.json"))
drop = [names.index(n) for n in ("PRI_jet_num", "ind_lt2_jets", "ind_0_jets")]
keep = [i for i in range(len(names)) if i not in drop]
X = torch.tensor(np.load(D / "X_scaled.npy")[:, keep])
raw = np.load(D / "X_raw.npy")
sp = np.load(D / "splits.npz")
jet = raw[:, names.index("PRI_jet_num")].astype(int)
d = X.shape[1]
S = {k: sp[k] for k in ("train", "val", "test", "sig_clf", "sig_eval")}
Xs = {k: X[v] for k, v in S.items()}
jv, jb, js = jet[S["val"]], jet[S["test"]], jet[S["sig_eval"]]


def corrected(sv, sb, ss):
    pb, ps = np.zeros(len(sb)), np.zeros(len(ss))
    for j in range(4):
        v = np.sort(sv[jv == j])
        pb[jb == j] = np.searchsorted(v, sb[jb == j], side="right") / len(v)
        ps[js == j] = np.searchsorted(v, ss[js == j], side="right") / len(v)
    return evaluate(pb, ps)


def report(kind, latent, seed, score, encode, epochs, val_recon):
    sv, sb, ss = (score(Xs[k]) for k in ("val", "test", "sig_eval"))
    raw_m, cor_m = evaluate(sb, ss), corrected(sv, sb, ss)
    row = {"model": kind, "latent": latent, "seed": seed, "epochs": epochs, "val_recon": val_recon,
           "raw_auc": raw_m["auc"], "raw_eff": raw_m["eff_at_fpr_0.01"],
           "cor_auc": cor_m["auc"], "cor_eff": cor_m["eff_at_fpr_0.01"]}
    for j in range(4):
        row[f"bg_mean_score_j{j}"] = float(sb[jb == j].mean())
        row[f"auc_j{j}"] = evaluate(sb[jb == j], ss[js == j])["auc"]
    # can the jet count still be read linearly from the code? (fit on val, score on test)
    clf = LogisticRegression(max_iter=2000).fit(encode(Xs["val"]), jv)
    row["jet_probe_acc"] = float((clf.predict(encode(Xs["test"])) == jb).mean())
    return row


rows = []
for latent in (2, 4, 8, 16):
    pca = PCA(n_components=latent).fit(Xs["train"].numpy())
    score = lambda A: ((A.numpy() - pca.inverse_transform(pca.transform(A.numpy()))) ** 2).mean(1)
    rows.append(report("pca", latent, 0, score, lambda A: pca.transform(A.numpy()), 0,
                       float(score(Xs["val"]).mean() * d)))
    print(rows[-1], flush=True)

for kind in ("ae", "vae"):
    for seed in (0, 1, 2):
        set_seed(seed)
        model = AE(d, 8) if kind == "ae" else VAE(d, 8)
        loss = ae_loss if kind == "ae" else make_vae_loss(1.0)
        model, hist = train(model, loss, Xs["train"], Xs["val"], max_epochs=300, patience=15, verbose=False)
        model.eval()
        torch.save(model.state_dict(), OUT / "runs_nojet" / f"{kind}_z8_s{seed}.pt")
        score = lambda A: model.score(A).numpy()
        encode = ((lambda A: model.enc(A).detach().numpy()) if kind == "ae"
                  else (lambda A: model.encode(A)[0].detach().numpy()))
        rows.append(report(kind, 8, seed, score, encode, len(hist["val"]),
                           float(model.score(Xs["val"]).mean().item() * d)))
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(OUT / "H_nojet.csv", index=False)

pd.DataFrame(rows).to_csv(OUT / "H_nojet.csv", index=False)
print("done")
