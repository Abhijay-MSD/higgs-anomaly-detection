from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from evaluation import evaluate
from models import set_seed

D = Path("data/processed")
X = torch.tensor(np.load(D / "X_scaled.npy"))
sp = np.load(D / "splits.npz")
d = X.shape[1]
sig_tr, sig_va = train_test_split(sp["sig_clf"], test_size=0.2, random_state=0)


def make(bg_idx, sig_idx):
    idx = np.r_[bg_idx, sig_idx]
    y = np.r_[np.zeros(len(bg_idx)), np.ones(len(sig_idx))]
    return X[idx], torch.tensor(y, dtype=torch.float32)


set_seed(0)
clf = nn.Sequential(nn.Linear(d, 64), nn.ReLU(),
                    nn.Linear(64, 32), nn.ReLU(),
                    nn.Linear(32, 1))
X_tr, y_tr = make(sp["train"], sig_tr)
X_va, y_va = make(sp["val"], sig_va)
loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=512, shuffle=True)
opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
lossf = nn.BCEWithLogitsLoss()

best, wait, best_state = float("inf"), 0, None
for epoch in range(100):
    clf.train()
    for xb, yb in loader:
        opt.zero_grad()
        lossf(clf(xb).squeeze(1), yb).backward()
        opt.step()
    clf.eval()
    with torch.no_grad():
        val = lossf(clf(X_va).squeeze(1), y_va).item()
    if epoch % 5 == 0:
        print(f"epoch {epoch:3d}  val loss {val:.4f}")
    if val < best:
        best, wait = val, 0
        best_state = {k: v.clone() for k, v in clf.state_dict().items()}
    else:
        wait += 1
        if wait >= 10:
            print(f"early stop at epoch {epoch}")
            break

clf.load_state_dict(best_state)
clf.eval()
with torch.no_grad():
    s_bg = clf(X[sp["test"]]).squeeze(1).numpy()
    s_sig = clf(X[sp["sig_eval"]]).squeeze(1).numpy()
print("supervised:", evaluate(s_bg, s_sig))