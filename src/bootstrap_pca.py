"""Bootstrap confidence intervals for the jet-corrected (percentile) results.

For every run we rebuild the percentile score used in stratified_eval.py (each event's
score ranked among validation background events with the same jet count), then resample
the test background and eval signal events with replacement 1000 times. The spread of
the metric over those resamples gives a 95% interval, plus a wider interval that allows
for looking at 4 PCA sizes at once. This covers test-sample noise only. PCA is one fit,
so training-sample noise is not included. Signal is used for evaluation only.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from models import AE, VAE

D, R = Path("data/processed"), Path("results")
N_BOOT = 1000
WIDE = 100 * 0.05 / 4 / 2          # 0.625: two-sided interval for 4 looks at 5% overall

X = torch.tensor(np.load(D / "X_scaled.npy"))
raw = np.load(D / "X_raw.npy")
sp = np.load(D / "splits.npz")
names = json.load(open(D / "feature_names.json"))
jet = raw[:, names.index("PRI_jet_num")].astype(int)
X_train, X_val = X[sp["train"]], X[sp["val"]]
jet_val, jet_bg, jet_sig = jet[sp["val"]], jet[sp["test"]], jet[sp["sig_eval"]]
d = X.shape[1]


def val_scores(kind, latent, seed):
    """Anomaly scores for validation background, from the saved model (as in stratified_eval)."""
    if kind == "pca":
        pca = PCA(n_components=latent).fit(X_train.numpy())
        A = X_val.numpy()
        return ((A - pca.inverse_transform(pca.transform(A))) ** 2).mean(axis=1)
    model = AE(d, latent) if kind == "ae" else VAE(d, latent)
    model.load_state_dict(torch.load(R / "runs" / f"{kind}_z{latent}_s{seed}.pt"))
    model.eval()
    return model.score(X_val).numpy()


def percentile_scores(s_val, s_bg, s_sig):
    """Rank each score among validation background events with the same jet count."""
    pct_bg, pct_sig = np.zeros(len(s_bg)), np.zeros(len(s_sig))
    for j in range(4):
        v = np.sort(s_val[jet_val == j])
        mb, ms = jet_bg == j, jet_sig == j
        pct_bg[mb] = np.searchsorted(v, s_bg[mb], side="right") / len(v)
        pct_sig[ms] = np.searchsorted(v, s_sig[ms], side="right") / len(v)
    return pct_bg, pct_sig


def metrics(wb, ws):
    """AUC and signal efficiency at 1% FPR from per-value event counts (ties count half)."""
    nb, ns = wb.sum(), ws.sum()
    below = np.cumsum(wb) - wb                       # background strictly below each value
    auc = (ws * (below + 0.5 * wb)).sum() / (nb * ns)
    fpr = np.r_[0, np.cumsum(wb[::-1]) / nb]         # sweep the threshold from the top down
    tpr = np.r_[0, np.cumsum(ws[::-1]) / ns]
    return auc, np.interp(0.01, fpr, tpr)


def bootstrap(pct_bg, pct_sig, rng):
    """Point estimate and N_BOOT resampled (auc, eff) pairs. Resampling is done on counts."""
    vals = np.unique(np.r_[pct_bg, pct_sig])
    cb = np.bincount(np.searchsorted(vals, pct_bg), minlength=len(vals))
    cs = np.bincount(np.searchsorted(vals, pct_sig), minlength=len(vals))
    pb, ps = cb / cb.sum(), cs / cs.sum()
    draws = np.array([metrics(rng.multinomial(cb.sum(), pb), rng.multinomial(cs.sum(), ps))
                      for _ in range(N_BOOT)])
    return metrics(cb, cs), draws


def run_row(kind, latent, seed, rng):
    sc = np.load(R / "runs" / f"{kind}_z{latent}_s{seed}.npz")
    pct_bg, pct_sig = percentile_scores(val_scores(kind, latent, seed), sc["s_bg"], sc["s_sig"])
    point, draws = bootstrap(pct_bg, pct_sig, rng)
    row = {"model": kind, "latent": latent, "seed": seed}
    for k, name in enumerate(("auc", "eff")):
        row[name] = point[k]
        row[f"{name}_lo"], row[f"{name}_hi"] = np.percentile(draws[:, k], [2.5, 97.5])
        row[f"{name}_wlo"], row[f"{name}_whi"] = np.percentile(draws[:, k], [WIDE, 100 - WIDE])
    return row


def main():
    runs = pd.read_csv(R / "results.csv")
    rng = np.random.default_rng(0)
    rows = []
    for _, r in runs.iterrows():
        rows.append(run_row(r["model"], int(r["latent"]), int(r["seed"]), rng))
        print("done", r["model"], r["latent"], r["seed"], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(R / "bootstrap_pct.csv", index=False)

    # sanity check: point estimates must match the earlier stratified results
    ref = pd.read_csv(R / "stratified.csv").query("group == 'pct_global'")
    m = df.merge(ref, on=["model", "latent", "seed"], suffixes=("", "_ref"))
    print("\nmax |AUC - stratified.csv| :", float((m.auc - m.auc_ref).abs().max()))
    print("max |eff - stratified.csv| :", float((m.eff - m["eff_at_fpr_0.01"]).abs().max()))

    pca = df[df.model == "pca"].set_index("latent")
    print("\nPCA, jet-corrected (chance: AUC 0.5, eff 0.01)")
    print(pca[["auc", "auc_lo", "auc_hi", "auc_wlo", "auc_whi",
               "eff", "eff_lo", "eff_hi"]].round(4).to_string())

    nn = df[df.model != "pca"].copy()
    nn["auc_above_chance"] = nn.auc_lo > 0.5
    nn["auc_below_chance"] = nn.auc_hi < 0.5
    g = nn.groupby(["model", "latent"]).agg(auc=("auc", "mean"), eff=("eff", "mean"),
                                            runs_above=("auc_above_chance", "sum"),
                                            runs_below=("auc_below_chance", "sum"))
    print("\nAE and VAE: mean over seeds, and how many of the 3 seeds have a 95% interval")
    print("entirely above (or below) chance")
    print(g.round(4).to_string())


if __name__ == "__main__":
    main()