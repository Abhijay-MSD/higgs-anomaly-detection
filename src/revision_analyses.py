"""Extra analyses requested in peer review. Uses the saved models only (no retraining).

  A. Confirmation run: the jet-corrected evaluation repeated on sig_clf, the half of the
     signal that the detectors never saw and that was not used to choose the correction.
  B. Rank correlations at the configuration level (seeds averaged), PCA and neural apart.
  C. Raw efficiencies at 1% FPR per configuration.
  D. Simple baselines: jet count as the score and an isolation forest (the k-nearest-
     neighbour baseline is in baseline_knn.py).
  E. Unsupervised scores on the latent codes: Mahalanobis distance, Gaussian-mixture
     negative log-likelihood, and (VAE) the negative ELBO. PCA rows get bootstrap intervals.
  F. Below-chance check: are signal events closer to the centre of the scaled feature
     space than background events with the same jet count?
  G. Masked score: reconstruction error averaged only over features that are defined for
     the event (no median-filled jet features, no indicators).

Signal labels are used only to compute metrics. Every fit (PCA, GMM, isolation forest,
kNN reference set, Mahalanobis mean and covariance) uses training background only.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture

from evaluation import evaluate
from models import AE, VAE

D, R = Path("data/processed"), Path("results")
OUT = R / "revision"
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)

X = np.load(D / "X_scaled.npy").astype(np.float32)
raw = np.load(D / "X_raw.npy")
names = json.load(open(D / "feature_names.json"))
sp = np.load(D / "splits.npz")
jet = raw[:, names.index("PRI_jet_num")].astype(int)
d = X.shape[1]
SETS = {k: sp[k] for k in ("train", "val", "test", "sig_clf", "sig_eval")}
defined = np.c_[~np.isnan(raw), np.zeros((len(raw), 3), bool)]   # indicators never count
runs = pd.read_csv(R / "results.csv")
rng = np.random.default_rng(0)
rng_boot = np.random.default_rng(1)   # separate stream for the latent-score intervals


# ---------------------------------------------------------------- helpers
def pct_correct(s_val, s_bg, s_sig, jv, jb, js):
    """Rank each score among validation background events with the same jet count."""
    pb, ps = np.zeros(len(s_bg)), np.zeros(len(s_sig))
    for j in range(4):
        v = np.sort(s_val[jv == j])
        pb[jb == j] = np.searchsorted(v, s_bg[jb == j], side="right") / len(v)
        ps[js == j] = np.searchsorted(v, s_sig[js == j], side="right") / len(v)
    return pb, ps


def count_metrics(wb, ws):
    nb, ns = wb.sum(), ws.sum()
    below = np.cumsum(wb) - wb
    auc = (ws * (below + 0.5 * wb)).sum() / (nb * ns)
    fpr = np.r_[0, np.cumsum(wb[::-1]) / nb]
    tpr = np.r_[0, np.cumsum(ws[::-1]) / ns]
    return auc, np.interp(0.01, fpr, tpr)


def bootstrap(pb, ps, n=1000, r=None):
    r = rng if r is None else r
    """Same multinomial bootstrap over score values as bootstrap_pca.py."""
    vals = np.unique(np.r_[pb, ps])
    cb = np.bincount(np.searchsorted(vals, pb), minlength=len(vals))
    cs = np.bincount(np.searchsorted(vals, ps), minlength=len(vals))
    qb, qs = cb / cb.sum(), cs / cs.sum()
    draws = np.array([count_metrics(r.multinomial(cb.sum(), qb),
                                    r.multinomial(cs.sum(), qs)) for _ in range(n)])
    point = count_metrics(cb, cs)
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    return {"auc": point[0], "auc_lo": lo[0], "auc_hi": hi[0],
            "eff": point[1], "eff_lo": lo[1], "eff_hi": hi[1]}


def full_eval(scores, sig="sig_eval", boot=False, r=None):
    """Raw and jet-corrected metrics for a dict of scores keyed by split name."""
    sb, ss, sv = scores["test"], scores[sig], scores["val"]
    raw_m = evaluate(sb, ss)
    pb, ps = pct_correct(sv, sb, ss, jet[SETS["val"]], jet[SETS["test"]], jet[SETS[sig]])
    row = {"raw_auc": raw_m["auc"], "raw_eff": raw_m["eff_at_fpr_0.01"]}
    if boot:
        b = bootstrap(pb, ps, r=r)
        row.update({f"cor_{k}": v for k, v in b.items()})
    else:
        c = evaluate(pb, ps)
        row.update({"cor_auc": c["auc"], "cor_eff": c["eff_at_fpr_0.01"]})
    return row


pca_cache = {}


def load_model(kind, latent, seed):
    if kind == "pca":
        if latent not in pca_cache:
            pca_cache[latent] = PCA(n_components=latent).fit(X[SETS["train"]])
        return pca_cache[latent]
    m = AE(d, latent) if kind == "ae" else VAE(d, latent)
    m.load_state_dict(torch.load(R / "runs" / f"{kind}_z{latent}_s{seed}.pt"))
    m.eval()
    return m


@torch.no_grad()
def recon_and_code(model, kind, A):
    """Reconstruction, latent code (mean code for the VAE) and VAE log-variance."""
    if kind == "pca":
        z = model.transform(A)
        return model.inverse_transform(z), z, None
    t = torch.tensor(A)
    if kind == "ae":
        z = model.enc(t)
        return model.dec(z).numpy(), z.numpy(), None
    mu, logvar = model.encode(t)
    return model.dec(mu).numpy(), mu.numpy(), logvar.numpy()


# ---------------------------------------------------------------- A, E, G
rows_a, rows_e, rows_g = [], [], []
for _, r in runs.iterrows():
    kind, latent, seed = r["model"], int(r["latent"]), int(r["seed"])
    model = load_model(kind, latent, seed)
    mse, masked, maha, gmm_s, elbo = {}, {}, {}, {}, {}
    sub = rng.choice(SETS["train"], 100_000, replace=False)
    _, z_tr, _ = recon_and_code(model, kind, X[sub])
    mean, cov = z_tr.mean(0), np.cov(z_tr, rowvar=False).reshape(latent, latent)
    icov = np.linalg.pinv(cov)
    gmm = GaussianMixture(n_components=8, covariance_type="full", random_state=0).fit(z_tr)
    for name in ("val", "test", "sig_clf", "sig_eval"):
        A = X[SETS[name]]
        rec, z, logvar = recon_and_code(model, kind, A)
        err = (rec - A) ** 2
        mse[name] = err.mean(1)
        m = defined[SETS[name]]
        masked[name] = (err * m).sum(1) / m.sum(1)
        dz = z - mean
        maha[name] = np.einsum("ij,jk,ik->i", dz, icov, dz)
        gmm_s[name] = -gmm.score_samples(z)
        if kind == "vae":
            kl = -0.5 * (1 + logvar - z ** 2 - np.exp(logvar)).sum(1)
            elbo[name] = err.sum(1) + kl
    base = {"model": kind, "latent": latent, "seed": seed}
    # A: confirmation on sig_clf, with bootstrap intervals
    rows_a.append({**base, **full_eval(mse, "sig_clf", boot=True)})
    # E: latent-code scores, on sig_eval
    for sname, sc in (("mahalanobis", maha), ("gmm_nll", gmm_s), ("neg_elbo", elbo)):
        if sc:
            # PCA is a single fit, so its latent-score rows also get bootstrap intervals
            rows_e.append({**base, "score": sname, **full_eval(sc, boot=(kind == "pca"), r=rng_boot)})
    # G: masked reconstruction error
    rows_g.append({**base, **full_eval(masked)})
    print("A/E/G done", kind, latent, seed, flush=True)

pd.DataFrame(rows_a).to_csv(OUT / "A_sigclf_confirmation.csv", index=False)
pd.DataFrame(rows_e).to_csv(OUT / "E_latent_scores.csv", index=False)
pd.DataFrame(rows_g).to_csv(OUT / "G_masked_score.csv", index=False)

# ---------------------------------------------------------------- B
strat = pd.read_csv(R / "stratified.csv")
pct = strat[strat.group == "pct_global"][["model", "latent", "seed", "auc"]].rename(columns={"auc": "cor_auc"})
m = runs.merge(pct, on=["model", "latent", "seed"])
cfg = m.groupby(["model", "latent"], as_index=False)[["val_recon", "auc", "cor_auc"]].mean()
rows_b = []
for label, df in (("all runs", m), ("all configs", cfg),
                  ("neural runs", m[m.model != "pca"]), ("neural configs", cfg[cfg.model != "pca"]),
                  ("pca", cfg[cfg.model == "pca"]),
                  ("ae configs", cfg[cfg.model == "ae"]), ("vae configs", cfg[cfg.model == "vae"]),
                  ("ae runs", m[m.model == "ae"])):
    for col in ("auc", "cor_auc"):
        rho, p = spearmanr(df.val_recon, df[col])
        rows_b.append({"set": label, "n": len(df), "metric": col, "rho": rho, "p": p})
pd.DataFrame(rows_b).to_csv(OUT / "B_correlations.csv", index=False)

# ---------------------------------------------------------------- C
c = runs.groupby(["model", "latent"]).agg(raw_auc=("auc", "mean"), raw_eff=("eff_at_fpr_0.01", "mean"),
                                         raw_eff_sd=("eff_at_fpr_0.01", "std"))
c.to_csv(OUT / "C_raw_efficiency.csv")

# ---------------------------------------------------------------- D
rows_d = []
jscore = {k: jet[v].astype(float) for k, v in SETS.items()}
jm = evaluate(jscore["test"], jscore["sig_eval"])
rows_d.append({"baseline": "jet count", "raw_auc": jm["auc"], "raw_eff": jm["eff_at_fpr_0.01"],
               "cor_auc": 0.5, "cor_eff": 0.01})
iso = IsolationForest(n_estimators=200, random_state=0).fit(X[SETS["train"]])
rows_d.append({"baseline": "isolation forest",
               **full_eval({k: -iso.score_samples(X[SETS[k]]) for k in ("val", "test", "sig_eval")})})
# the k-nearest-neighbour baseline is in baseline_knn.py
pd.DataFrame(rows_d).to_csv(OUT / "D_baselines.csv", index=False)
print("D done", flush=True)

# ---------------------------------------------------------------- F
rows_f = []
for j in range(4):
    b = SETS["test"][jet[SETS["test"]] == j]
    s = SETS["sig_eval"][jet[SETS["sig_eval"]] == j]
    def radius(idx):
        mk = defined[idx][:, :30].copy()
        mk[:, names.index("PRI_jet_num")] = False
        return ((X[idx][:, :30] ** 2) * mk).sum(1) / mk.sum(1)
    rb, rs = radius(b), radius(s)
    rows_f.append({"jets": j, "bg_mean_r2": rb.mean(), "sig_mean_r2": rs.mean(),
                   "auc_radius": evaluate(rb, rs)["auc"]})
pd.DataFrame(rows_f).to_csv(OUT / "F_centrality.csv", index=False)
print("all done")
