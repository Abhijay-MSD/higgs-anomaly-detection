import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
        mu, _ = self.encode(x)  # no random sampling at test time
        return ((self.dec(mu) - x) ** 2).mean(dim=1)


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


def train(model, loss_fn, X_tr, X_va, lr=1e-3, bs=512,
          max_epochs=100, patience=10, verbose=True):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(X_tr), batch_size=bs, shuffle=True)
    best, best_state, wait = float("inf"), None, 0
    train_hist, val_hist = [], []
    for epoch in range(max_epochs):
        model.train()
        total, n = 0.0, 0
        for (xb,) in loader:
            opt.zero_grad()
            loss = loss_fn(model, xb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
            n += len(xb)
        model.eval()
        with torch.no_grad():
            val = loss_fn(model, X_va).item()
        train_hist.append(total / n)
        val_hist.append(val)
        if verbose and epoch % 5 == 0:
            print(f"epoch {epoch:3d}  train {train_hist[-1]:.4f}  val {val:.4f}")
        if val < best:
            best, wait = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                print(f"early stop at epoch {epoch}, best val {best:.4f}")
                break
    model.load_state_dict(best_state)
    return model, {"train": train_hist, "val": val_hist}