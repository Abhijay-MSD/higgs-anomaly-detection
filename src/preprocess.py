import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

SEED = 0
RAW = "data/raw/atlas-higgs-challenge-2014-v2.csv.gz"
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW)
feature_cols = [c for c in df.columns if c.startswith(("DER_", "PRI_"))]
assert len(feature_cols) == 30

X = df[feature_cols].replace(-999.0, np.nan)
is_sig = (df["Label"] == "s").to_numpy()
jet_num = df["PRI_jet_num"].to_numpy()

# 1. Indicator columns, one per missingness pattern
ind = pd.DataFrame({
    "ind_lt2_jets": X["DER_mass_jet_jet"].isna(),
    "ind_0_jets": X["PRI_jet_leading_pt"].isna(),
    "ind_mmc_missing": X["DER_mass_MMC"].isna(),
}).astype(np.float32)

# Check our understanding of the missing values against the data
assert (ind["ind_lt2_jets"].to_numpy() == (jet_num < 2)).all()
assert (ind["ind_0_jets"].to_numpy() == (jet_num == 0)).all()

# 2. Splits: background only for train/val/test, signal kept in its own pool
bg_idx = np.where(~is_sig)[0]
sig_idx = np.where(is_sig)[0]
train_idx, rest = train_test_split(bg_idx, test_size=0.30, random_state=SEED)
val_idx, test_idx = train_test_split(rest, test_size=0.50, random_state=SEED)
sig_clf_idx, sig_eval_idx = train_test_split(sig_idx, test_size=0.50, random_state=SEED)

# 3. Fill missing values using TRAIN background medians only
medians = X.iloc[train_idx].median()
X_filled = X.fillna(medians)

# 4. Fit the transform on TRAIN background only, then apply to everything
qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                         subsample=10**6, random_state=SEED)
qt.fit(X_filled.iloc[train_idx])
X_scaled = qt.transform(X_filled)

# 5. Add the raw 0/1 indicators after the transform
X_final = np.hstack([X_scaled, ind.to_numpy()]).astype(np.float32)
feature_names = feature_cols + list(ind.columns)

# 6. Safety checks
assert X_final.shape == (len(df), 33)
assert np.isfinite(X_final).all()
groups = {"train": train_idx, "val": val_idx, "test": test_idx,
          "sig_clf": sig_clf_idx, "sig_eval": sig_eval_idx}
names = list(groups)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        assert len(np.intersect1d(groups[names[i]], groups[names[j]])) == 0, (names[i], names[j])
for n in ("train", "val", "test"):
    assert not is_sig[groups[n]].any(), f"signal found in {n}"

# 7. Save everything so every model uses identical data
np.save(OUT / "X_scaled.npy", X_final)
np.save(OUT / "X_raw.npy", X.to_numpy(dtype=np.float32))   # unscaled, with NaN, for physics checks
np.save(OUT / "is_signal.npy", is_sig)
np.save(OUT / "weights.npy", df["Weight"].to_numpy(dtype=np.float32))
np.savez(OUT / "splits.npz", **groups)
(OUT / "feature_names.json").write_text(json.dumps(feature_names))

print("input shape:", X_final.shape)
for n, idx in groups.items():
    print(f"{n:9s} {len(idx):8d} events")
print("first 5 train indices:", train_idx[:5])
print("all checks passed")