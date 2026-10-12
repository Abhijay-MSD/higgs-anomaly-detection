import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def evaluate(s_bg, s_sig, fpr_targets=(0.01, 0.001)):
    """Higher score = more anomalous. Background is class 0, signal is class 1."""
    y = np.r_[np.zeros(len(s_bg)), np.ones(len(s_sig))]
    s = np.r_[s_bg, s_sig]
    fpr, tpr, _ = roc_curve(y, s)
    out = {"auc": float(roc_auc_score(y, s))}
    for f in fpr_targets:
        out[f"eff_at_fpr_{f}"] = float(np.interp(f, fpr, tpr))
    return out