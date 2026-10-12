import csv
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

from evaluation import evaluate
from models import AE, VAE, ae_loss, make_vae_loss, set_seed, train

D = Path("data/processed")
R = Path("results")
(R / "runs").mkdir(parents=True, exist_ok=True)

X = torch.tensor(np.load(D / "X_scaled.npy"))
sp = np.load(D / "splits.npz")
X_train, X_val, X_test = X[sp["train"]], X[sp["val"]], X[sp["test"]]
X_sig = X[sp["sig_eval"]]
d = X.shape[1]

LATENTS = [2, 4, 8, 16]
SEEDS = [0, 1, 2]
BETA = 1.0
CSV = R / "results.csv"
FIELDS = ["model", "latent", "seed", "beta", "epochs", "val_recon",
          "auc", "eff_at_fpr_0.01", "eff_at_fpr_0.001", "seconds"]

# Resume support: remember which runs are already in the CSV
done = set()
if CSV.exists():
    with open(CSV) as f:
        for row in csv.DictReader(f):
            done.add((row["model"], int(row["latent"]), int(row["seed"])))
else:
    with open(CSV, "w", newline="") as f:
        csv.DictWriter(f, FIELDS).writeheader()


def log(row):
    with open(CSV, "a", newline="") as f:
        csv.DictWriter(f, FIELDS).writerow(row)


def run_pca(latent):
    t0 = time.time()
    pca = PCA(n_components=latent).fit(X_train.numpy())

    def score(A):
        A = A.numpy()
        return ((A - pca.inverse_transform(pca.transform(A))) ** 2).mean(axis=1)

    s_bg, s_sig = score(X_test), score(X_sig)
    res = evaluate(s_bg, s_sig)
    np.savez(R / "runs" / f"pca_z{latent}_s0.npz", s_bg=s_bg, s_sig=s_sig)
    return {"model": "pca", "latent": latent, "seed": 0, "beta": "", "epochs": 0,
            "val_recon": float(score(X_val).mean() * d), **res,
            "seconds": round(time.time() - t0, 1)}


def run_neural(kind, latent, seed):
    set_seed(seed)
    if kind == "ae":
        model, loss_fn = AE(d, latent), ae_loss
    else:
        model, loss_fn = VAE(d, latent), make_vae_loss(BETA)
    t0 = time.time()
    model, hist = train(model, loss_fn, X_train, X_val,
                        max_epochs=600, patience=15, verbose=False)
    model.eval()
    s_bg, s_sig = model.score(X_test).numpy(), model.score(X_sig).numpy()
    res = evaluate(s_bg, s_sig)
    tag = f"{kind}_z{latent}_s{seed}"
    np.savez(R / "runs" / f"{tag}.npz", s_bg=s_bg, s_sig=s_sig,
             train=hist["train"], val=hist["val"])
    torch.save(model.state_dict(), R / "runs" / f"{tag}.pt")
    return {"model": kind, "latent": latent, "seed": seed,
            "beta": BETA if kind == "vae" else "",
            "epochs": len(hist["val"]),
            "val_recon": float(model.score(X_val).mean().item() * d), **res,
            "seconds": round(time.time() - t0, 1)}


for latent in LATENTS:
    if ("pca", latent, 0) not in done:
        row = run_pca(latent)
        log(row)
        print(row)
    for seed in SEEDS:
        for kind in ("ae", "vae"):
            if (kind, latent, seed) in done:
                continue
            row = run_neural(kind, latent, seed)
            log(row)
            print(row)

print("grid finished")