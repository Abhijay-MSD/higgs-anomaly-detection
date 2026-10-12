# PRD: Autoencoder Anomaly Detection Model

Oct 3, 2026 · @Moneyplotter

## 1. Overview and goal

This document specifies the models and experiments you will build for the paper, and how to build them step by step. The system trains neural networks on ordinary collision events only, then measures how well they flag a signal they have never seen.

**Goal:** compare a vanilla autoencoder, a variational autoencoder (VAE) and a PCA baseline on how well each detects a hidden proxy signal in the ATLAS Higgs dataset, and measure how that changes with latent size and signal difficulty.

**Core question:** which design choices make an unsupervised anomaly detector sensitive to unseen signals?

**Who it is for:** you (Plotter), as the sole builder, working in Google Colab or a local machine with PyTorch.

## 2. Scope

**In scope:**

- Vanilla autoencoder and VAE written in PyTorch
- PCA reconstruction-error baseline (scikit-learn)
- A supervised classifier as an upper bound, for comparison only
- One dataset: the ATLAS Higgs challenge data, with the Higgs events as the proxy signal
- Latent sizes, seeds, signal injection and a latent-space analysis

**Out of scope for the first version (extras if time allows):**

- Denoising and deeper autoencoders
- A second dataset
- CWoLa and other weakly supervised methods
- Real detector data and systematic uncertainties
- Any claim of discovering new physics

## 3. Success criteria

The model work is done when all of these are true:

- Both autoencoders train without errors and their validation loss on background flattens out
- Every result is the mean and standard deviation over at least 3 seeds
- The primary metric (signal efficiency at 1% false positive rate) is computed for every model and latent size
- The PCA baseline and the supervised upper bound are reported next to the autoencoders
- No signal event was used in training, validation, early stopping or tuning, and the code proves it
- One script regenerates all splits and results from the raw data
- A fixed set of figures and tables exists (listed in section 8)

**Not a success criterion:** the autoencoders beating PCA. A tie or a loss is still a valid, reportable result.

## 4. Data requirements

**Source:** the ATLAS Higgs ML challenge dataset on the CERN Open Data Portal (simulated collisions, about 800,000 events, about 30 features, a signal/background label and an event weight). Confirm the exact file name, columns and counts on the portal page before you rely on them.

**Rules the data pipeline must follow:**

- Features go to the model. Labels and weights are kept separately and never enter training.
- The missing-value marker (-999 in this dataset) must be handled by one documented choice: drop the column, impute, or add a missing-indicator column.
- The jet count feature (PRI\_jet\_num) controls which other features are defined. Check this when you explore, and decide whether to treat it as a category.
- Background events are split into train (about 70%), validation (about 15%) and test (about 15%). Fix the random seed for the split and save the indices.
- Signal events form a separate pool used only at evaluation time.
- Scalers (standardisation or quantile transform) are fitted on training background only.
- Heavy-tailed features (energies, momenta) get a log or quantile transform, checked with before and after histograms.
- Processed arrays are saved to disk so every model reads identical data.

**Check:** print the shape of every split and the signal fraction in each. If signal appears in a training split, stop and fix it.

## 5. Model specifications

The numbers below are sensible starting values, not fixed rules. Keep them identical across models unless a row says otherwise. Here, d is the number of input features after preprocessing.

| Model | Type | Architecture | Loss | Anomaly score |
| --- | --- | --- | --- | --- |
| PCA baseline | Linear, no training loop | Keep k components, k equal to each latent size | None (variance based) | Squared error between input and PCA reconstruction |
| Vanilla AE | Neural network | d, 64, 32, latent, 32, 64, d with ReLU activations | Mean squared error | Per-event mean squared reconstruction error |
| VAE | Neural network, probabilistic | Encoder d, 64, 32 then two heads (mean, log variance) of size latent, decoder mirrors the vanilla AE | Reconstruction error plus beta times KL divergence | Reconstruction error using the latent mean (no random sampling at test time) |
| Supervised classifier (upper bound) | Neural network | d, 64, 32, 1 with sigmoid | Binary cross entropy | Predicted signal probability |
|  |  |  |  |  |

**Latent sizes to test:** 2, 4, 8, 16 (add 32 if time allows). Each applies to PCA, the vanilla AE and the VAE.

**VAE beta:** start at 1, then try one smaller value such as 0.1. Report which you used.

**Matching rule:** the vanilla AE and VAE share hidden layer sizes so differences come from the method, not capacity.

**Supervised classifier:** trained on labelled signal and background, used only as a ceiling. Do not let it influence any choice for the unsupervised models.

## 6. Training requirements

| Setting | Starting value | Rule |
| --- | --- | --- |
| Optimiser | Adam | Same for all neural models |
| Learning rate | 0.001 | Change only after a documented validation-loss reason |
| Batch size | 512 | Same for all neural models |
| Max epochs | 100 | Stop earlier with early stopping |
| Early stopping | Patience of 10 epochs on validation loss | Validation loss uses background only |
| Seeds | 3 minimum (5 if time allows) | Report mean and standard deviation |
| Hardware | Free Colab GPU or CPU | The networks are small, so CPU is fine |

**Tuning rule:** hyperparameters are chosen using validation reconstruction loss on background only. Signal performance is never used to pick a setting.

**Save for every run:** the config, seed, final weights, training and validation loss curves, and training time.

## 7. Evaluation requirements

**Test set construction:** take the held-out background test split and mix in signal events from the separate signal pool at fixed fractions (0.1%, 1% and 10%). Score every event with every model. Signal efficiency at a fixed false positive rate does not change with the signal fraction, so use precision and significance when you vary it.

**Primary metric:** signal efficiency at a 1% false positive rate. Write this down before you see any result.

**Other metrics to report:**

- ROC curve and AUC
- Signal efficiency at a 0.1% false positive rate
- Histograms of anomaly scores for background and signal
- Per-feature reconstruction error, to check that one feature is not dominating the score

**Thresholds:** chosen only from validation background, for example the 99th percentile of its scores. Never from signal.

**Signal difficulty study:** repeat the evaluation on signal subsets (split by a physical variable such as invariant mass) and on synthetic perturbations where you shift selected features by 0.25, 0.5, 1 and 2 standard deviations. Label perturbed data clearly as synthetic.

**Latent analysis:** project encoded events with PCA and UMAP, fit a linear probe to predict signal from latent codes, and regress physical variables from them.

## 8. Experiment matrix and required outputs

**Main grid:** 3 models (PCA, vanilla AE, VAE) across 4 latent sizes. The two neural models run with 3 seeds each, so the main grid is 2 models by 4 sizes by 3 seeds, which gives 24 neural runs. PCA is deterministic and needs only one run per size. Add the supervised classifier at 3 seeds.

**Outputs to produce:**

1. Table of primary metric for every model and latent size, with standard deviations
2. Score histograms, background vs. signal, per model
3. ROC curves at one chosen latent size
4. Primary metric vs. latent dimension, one line per model, with error bars
5. Detectability plots: precision and significance vs. signal fraction, and signal efficiency vs. feature shift size
6. Latent projections (PCA and UMAP) coloured by class and by a physical variable
7. Table of probe AUC and variance explained
8. Training curves in the appendix
9. A results folder of saved scores so figures can be remade without retraining

## 9. Risks and constraints

| Risk | Why it matters | Mitigation |
| --- | --- | --- |
| Data leakage | Makes results look better than they are | Fit scalers on train background only, keep the signal pool separate, assert no overlap in code |
| Signal is only a proxy | Higgs is a Standard Model signal, not new physics | Call it a proxy signal everywhere, state it in the limitations |
| One seed looks good by luck | False ranking of models | Always report at least 3 seeds with standard deviations |
| Score dominated by one feature | The anomaly score ends up measuring one variable | Check per-feature error and compare against distance from the mean |
| Missing values (-999) fed in raw | Corrupts scaling and training | Handle explicitly before scaling, and plot to confirm |
| Posterior collapse in the VAE | Latent space carries no information | Watch the KL term, lower beta if it goes to zero |
| Scope creep | Project does not finish | Finish the 2-model version first and treat extras as optional |
| Wrong dataset facts | Errors in the paper | Confirm counts, columns and links on the CERN page |

## 10. Step-by-step, part 1: setup and data

The code in this and the next two sections is a starting sketch. Type it out yourself, run it in small pieces and fix what breaks. That is how you learn it well enough to defend it.

**Step 1. Rehearse on MNIST first.** Before the real data, train a plain autoencoder on nine MNIST digits, hold one digit out, score it by reconstruction error and plot a ROC curve. This tests your whole pipeline on easy data. Do not skip it.

**Step 2. Set up your workspace.**

1. Open Google Colab and create a notebook. Switch to a GPU runtime if you like, though CPU is enough here.
2. Make a GitHub repository with folders `src/`, `configs/`, `results/` and `figures/`.
3. Install what you need: PyTorch, scikit-learn, pandas, numpy, matplotlib and umap-learn.
4. Fix a global random seed function so you can reproduce runs.

**Step 3. Download the data.** Go to the CERN Open Data Portal, find the ATLAS Higgs ML challenge 2014 dataset, and download the CSV file. Read its description page and note the column list, the label meaning and the number of events.

**Step 4. Load and explore.**

```python
import pandas as pd, numpy as np

df = pd.read_csv("atlas-higgs-challenge-2014-v2.csv.gz")  # check the real file name
print(df.shape)
print(df.columns.tolist())
print(df["Label"].value_counts())   # s = signal, b = background

feature_cols = [c for c in df.columns if c.startswith(("DER_", "PRI_"))]
print(len(feature_cols))
print((df[feature_cols] == -999.0).mean().sort_values(ascending=False).head(10))
```

Then plot a histogram of five or six features for signal and background. Write one sentence on what you notice, for example which features separate the classes and which overlap.

**Step 5. Preprocess carefully.**

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

X = df[feature_cols].replace(-999.0, np.nan)
is_sig = (df["Label"] == "s").values

# 1. missing-value indicators (keeps the information that a value was undefined)
missing_cols = [c for c in feature_cols if X[c].isna().any()]
for c in missing_cols:
    X[c + "_missing"] = X[c].isna().astype(float)

# 2. split background only, keep signal in its own pool
bg_idx = np.where(~is_sig)[0]
sig_idx = np.where(is_sig)[0]
train_idx, rest = train_test_split(bg_idx, test_size=0.30, random_state=0)
val_idx, test_idx = train_test_split(rest, test_size=0.50, random_state=0)

# 3. fill missing values using TRAIN background medians only
medians = X.iloc[train_idx][feature_cols].median()
X[feature_cols] = X[feature_cols].fillna(medians)

# 4. fit the scaler on TRAIN background only, then apply to everything
qt = QuantileTransformer(output_distribution="normal", random_state=0)
qt.fit(X.iloc[train_idx])
X_scaled = qt.transform(X)

# 5. save everything so every model uses identical data
np.save("X_scaled.npy", X_scaled)
np.savez("splits.npz", train=train_idx, val=val_idx, test=test_idx, sig=sig_idx)
```

The indicator columns for the missing values are also run through the quantile transform here. That is fine, but check they still look sensible.

**Check before moving on:**

- No signal index appears in `train`, `val` or `test`
- Scaled training features look roughly standard normal
- Running the script twice gives identical splits

## 11. Step-by-step, part 2: build and train the models

**Step 6. Load the processed data into PyTorch.**

```python
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

X = torch.tensor(np.load("X_scaled.npy"), dtype=torch.float32)
sp = np.load("splits.npz")
X_train, X_val, X_test = X[sp["train"]], X[sp["val"]], X[sp["test"]]
X_sig = X[sp["sig"]]
d = X.shape[1]
```

Split the signal pool in half now. One half is used only to train the supervised upper bound. The other half is the evaluation signal that every model is tested on.

**Step 7. Write the two neural models.**

```python
class AE(nn.Module):
    def __init__(self, d, d_lat):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU(),
                                 nn.Linear(32, d_lat))
        self.dec = nn.Sequential(nn.Linear(d_lat, 32), nn.ReLU(),
                                 nn.Linear(32, 64), nn.ReLU(),
                                 nn.Linear(64, d))
    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z
    @torch.no_grad()
    def score(self, x):
        recon, _ = self(x)
        return ((recon - x) ** 2).mean(dim=1)

class VAE(nn.Module):
    def __init__(self, d, d_lat):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, 64), nn.ReLU(),
                                  nn.Linear(64, 32), nn.ReLU())
        self.mu = nn.Linear(32, d_lat)
        self.logvar = nn.Linear(32, d_lat)
        self.dec = nn.Sequential(nn.Linear(d_lat, 32), nn.ReLU(),
                                 nn.Linear(32, 64), nn.ReLU(),
                                 nn.Linear(64, d))
    def encode(self, x):
        h = self.body(x)
        return self.mu(h), self.logvar(h)
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.dec(z), mu, logvar
    @torch.no_grad()
    def score(self, x):
        mu, _ = self.encode(x)          # no random sampling at test time
        return ((self.dec(mu) - x) ** 2).mean(dim=1)
```

**Step 8. Write the losses and one shared training function.**

```python
def ae_loss(model, x):
    recon, _ = model(x)
    return ((recon - x) ** 2).sum(dim=1).mean()

def make_vae_loss(beta):
    def vae_loss(model, x):
        recon, mu, logvar = model(x)
        rec = ((recon - x) ** 2).sum(dim=1).mean()
        kl = (-0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(dim=1)).mean()
        return rec + beta * kl
    return vae_loss

def train(model, loss_fn, X_tr, X_va, lr=1e-3, bs=512, max_epochs=100, patience=10):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(X_tr), batch_size=bs, shuffle=True)
    best, best_state, wait, history = float("inf"), None, 0, []
    for epoch in range(max_epochs):
        model.train()
        for (xb,) in loader:
            opt.zero_grad()
            loss_fn(model, xb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = loss_fn(model, X_va).item()
        history.append(val)
        if val < best:
            best, wait = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model, history
```

The validation set here is background only. That is what keeps tuning honest.

**Step 9. First end-to-end run (one model, one latent size).**

```python
torch.manual_seed(0)
model, hist = train(AE(d, 8), ae_loss, X_train, X_val)
model.eval()
s_bg  = model.score(X_test).numpy()
s_sig = model.score(X_sig_eval).numpy()   # X_sig_eval = the evaluation half of the signal pool
```

Plot the validation-loss curve and a score histogram. If the loss does not fall, or the signal scores look the same as background, debug here before building anything else. A weak first result is normal, so check the preprocessing and learning rate first.

**Step 10. Add the PCA baseline.**

```python
from sklearn.decomposition import PCA

def pca_scores(k, X_tr, X_eval):
    pca = PCA(n_components=k).fit(X_tr.numpy())
    rec = pca.inverse_transform(pca.transform(X_eval.numpy()))
    return ((X_eval.numpy() - rec) ** 2).mean(axis=1)
```

**Step 11. Train the VAE.** Use `make_vae_loss(1.0)` and watch the KL term. If it drops to nearly zero, the latent space is not being used, so lower beta.

**Step 12. Train the supervised upper bound.** Build a small network (d, 64, 32, 1) with a sigmoid output and binary cross entropy. Train it on background train data plus the training half of the signal pool. Never use it to make any decision about the unsupervised models.

**Step 13. Run the full grid.** Loop over latent sizes 2, 4, 8, 16, models (vanilla AE, VAE) and seeds (0, 1, 2). Save each run's weights, loss history, config and test scores to the results folder as it finishes, so a crash does not cost you everything.

## 12. Step-by-step, part 3: evaluate, analyse, log

**Step 14. Compute the metrics for every run.**

```python
from sklearn.metrics import roc_auc_score, roc_curve

def evaluate(s_bg, s_sig, fpr_targets=(0.01, 0.001)):
    y = np.r_[np.zeros(len(s_bg)), np.ones(len(s_sig))]
    s = np.r_[s_bg, s_sig]
    fpr, tpr, _ = roc_curve(y, s)
    out = {"auc": roc_auc_score(y, s)}
    for f in fpr_targets:
        out[f"eff_at_fpr_{f}"] = float(np.interp(f, fpr, tpr))
    return out
```

Append one row per run (model, latent size, seed, metrics) to a CSV file. Later, a pandas `groupby` on model and latent size gives your mean and standard deviation.

**A point that is easy to get wrong:** signal efficiency at a fixed false positive rate does not change with the signal fraction, because the ROC curve does not depend on class balance. Signal fraction matters for precision and for significance, so measure those when you vary it.

**Step 15. Run the hidden-signal studies.**

1. **Signal fraction.** Mix evaluation signal into held-out background at 0.1%, 1% and 10%. Number of signal events is `n_sig = f * n_bg / (1 - f)`. Report precision and significance (signal divided by the square root of background above the threshold) at a threshold set from validation background, for example its 99th percentile.
2. **Signal subsets.** Split the evaluation signal by a physical variable, such as bins of a mass feature, and compute efficiency at fixed false positive rate per bin.
3. **Synthetic perturbations.** Copy held-out background events and shift chosen features by 0.25, 0.5, 1 and 2 standard deviations. Because the features were quantile-transformed to roughly a standard normal, adding the shift directly to the scaled values works. Plot efficiency against shift size. Label these events as synthetic in every figure.

```python
def make_synthetic(X_bg, cols, shift):
    X_syn = X_bg.clone()
    X_syn[:, cols] += shift
    return X_syn
```

**Step 16. Analyse the latent space.**

```python
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

with torch.no_grad():
    z_bg  = model.enc(X_test).numpy()          # for the VAE use model.encode(x)[0]
    z_sig = model.enc(X_sig_eval).numpy()

Z = np.r_[z_bg, z_sig]
y = np.r_[np.zeros(len(z_bg)), np.ones(len(z_sig))]
Z_tr, Z_te, y_tr, y_te = train_test_split(Z, y, test_size=0.5, random_state=0, stratify=y)

probe = LogisticRegression(max_iter=1000).fit(Z_tr, y_tr)
print("probe AUC:", roc_auc_score(y_te, probe.predict_proba(Z_te)[:, 1]))
```

For the physics-meaning test, fit a `Ridge` regression from the latent codes to an unscaled physical variable such as a transverse mass feature, and report the R-squared on held-out events. For pictures, take about 20,000 events and run PCA first, then UMAP (`umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=0)`), coloured by class and by a physical variable. State the UMAP settings in the paper.

**Step 17. Aggregate and plot.** Read your results CSV, group by model and latent size, and make the figures from section 8. Use error bars from the seed spread. Keep plotting code in a script so you can regenerate everything quickly.

**Step 18. Log and reproduce.**

1. Record the config, seed and library versions for each run.
2. Commit your code to GitHub regularly with clear messages.
3. Rerun one full experiment from the raw data on a fresh Colab runtime. If the numbers match, your pipeline is reproducible.

**Step 19. Stop and write.** Once the main grid, the hidden-signal studies and the latent analysis are done, move to writing. Methods and Results come first, straight from the code and figures you now have. Extras like denoising autoencoders, ablations or a second dataset come after a finished draft exists.

## 13. Done checklist

- [ ] MNIST rehearsal works and gives a sensible ROC curve
- [ ] Dataset downloaded, columns and counts confirmed on the CERN page
- [ ] Preprocessing script run, splits saved, no signal in train, validation or test background
- [ ] Primary metric and hypotheses written down before seeing results
- [ ] Vanilla AE trains end to end on one latent size
- [ ] PCA baseline added
- [ ] VAE trains and its KL term is not collapsed
- [ ] Supervised upper bound trained on its own half of the signal pool
- [ ] Full grid run: 2 neural models, 4 latent sizes, 3 seeds
- [ ] Metrics saved to a results CSV, one row per run
- [ ] Signal fraction, signal subset and synthetic perturbation studies done
- [ ] Latent probes, regressions and PCA/UMAP projections done
- [ ] All figures regenerate from saved results with one script
- [ ] One experiment reproduced from scratch on a fresh runtime
- [ ] Code committed to GitHub with a README
- [ ] Ready to start writing Methods and Results
