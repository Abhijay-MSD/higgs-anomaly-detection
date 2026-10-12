from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from evaluation import evaluate
from models import AE, ae_loss, set_seed, train

D = Path("data/processed")
Path("figures").mkdir(exist_ok=True)

X = torch.tensor(np.load(D / "X_scaled.npy"))
sp = np.load(D / "splits.npz")
X_train, X_val, X_test = X[sp["train"]], X[sp["val"]], X[sp["test"]]
X_sig = X[sp["sig_eval"]]
d = X.shape[1]
LATENT = 8
print("input size:", d, "| latent size:", LATENT)

# --- Vanilla autoencoder ---
set_seed(0)
model, hist = train(AE(d, LATENT), ae_loss, X_train, X_val, max_epochs=300, patience=15)
model.eval()
s_bg = model.score(X_test).numpy()
s_sig = model.score(X_sig).numpy()
ae_res = evaluate(s_bg, s_sig)

# --- PCA baseline with the same number of components ---
pca = PCA(n_components=LATENT).fit(X_train.numpy())
def pca_score(A):
    A = A.numpy()
    return ((A - pca.inverse_transform(pca.transform(A))) ** 2).mean(axis=1)
pca_res = evaluate(pca_score(X_test), pca_score(X_sig))

print("\nAE :", ae_res)
print("PCA:", pca_res)

# --- Figures ---
plt.figure(figsize=(5, 3.5))
plt.plot(hist["train"], label="train")
plt.plot(hist["val"], label="validation")
plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
plt.savefig("figures/first_run_loss.png", dpi=150)

plt.figure(figsize=(5, 3.5))
bins = np.linspace(np.log10(min(s_bg.min(), s_sig.min()) + 1e-12),
                   np.log10(max(s_bg.max(), s_sig.max())), 80)
plt.hist(np.log10(s_bg + 1e-12), bins=bins, density=True, alpha=0.6, label="background")
plt.hist(np.log10(s_sig + 1e-12), bins=bins, density=True, alpha=0.6, label="signal (proxy)")
plt.xlabel("log10 anomaly score"); plt.ylabel("density"); plt.legend(); plt.tight_layout()
plt.savefig("figures/first_run_scores.png", dpi=150)
print("\nsaved figures/first_run_loss.png and figures/first_run_scores.png")