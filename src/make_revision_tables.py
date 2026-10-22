"""Build the two appendix tables for the revised paper (LaTeX) from results/revision/*.csv."""
from pathlib import Path

import pandas as pd

R = Path("results/revision")
A = pd.read_csv(R / "A_sigclf_confirmation.csv")
Dd = pd.read_csv(R / "D_baselines.csv")
E = pd.read_csv(R / "E_latent_scores.csv")
G = pd.read_csv(R / "G_masked_score.csv")
H = pd.read_csv(R / "H_nojet.csv")
I = pd.read_csv(R / "I_energy_shift.csv")
NAME = {"pca": "PCA", "ae": "AE", "vae": "VAE"}


def f3(x): return f"{x:.3f}"
def pc(x): return f"{100 * x:.2f}"
def rng3(s): return f"{s.min():.3f}--{s.max():.3f}"
def rngp(s): return f"{100 * s.min():.2f}--{100 * s.max():.2f}"


def ms(s, pct=False):
    """mean +- sd over seeds"""
    k = 100 if pct else 1
    d = 2 if pct else 3
    return f"{k * s.mean():.{d}f} $\\pm$ {k * s.std():.{d}f}"


rows = []
def add(check, model, raw, cor, eff): rows.append(f"    {check} & {model} & {raw} & {cor} & {eff} \\\\")

# confirmation on sig_clf
for k in (2, 4, 8, 16):
    r = A[(A.model == "pca") & (A.latent == k)].iloc[0]
    add("Confirmation (\\texttt{sig\\_clf})" if k == 2 else "", f"PCA, {k}", f3(r.raw_auc),
        f"{f3(r.cor_auc)} ({f3(r.cor_auc_lo)}--{f3(r.cor_auc_hi)})", f"{pc(r.cor_eff)} ({pc(r.cor_eff_lo)}--{pc(r.cor_eff_hi)})")
nn = A[A.model != "pca"].groupby(["model", "latent"])[["raw_auc", "cor_auc", "cor_eff"]].mean()
add("", "AE, VAE (all $k$)", rng3(nn.raw_auc), rng3(nn.cor_auc), rngp(nn.cor_eff))
rows.append("    \\addlinespace")
# baselines
for i, r in Dd.iterrows():
    name = {"jet count": "Jet count as the score", "isolation forest": "Isolation forest"}.get(r.baseline, "Distance to 10th neighbour")
    add("Baseline" if i == 0 else "", name, f3(r.raw_auc), f3(r.cor_auc), pc(r.cor_eff))
rows.append("    \\addlinespace")
# latent-code scores: best k (by corrected AUC) per score and model
LAB = {"mahalanobis": "Mahalanobis distance", "gmm_nll": "Gaussian-mixture NLL", "neg_elbo": "Negative ELBO"}
first = True
for sc in ("mahalanobis", "gmm_nll", "neg_elbo"):
    g = E[E.score == sc].groupby(["model", "latent"])
    m = g[["raw_auc", "cor_auc", "cor_eff"]].mean()
    for mod in ("pca", "ae", "vae"):
        if mod not in m.index.get_level_values(0):
            continue
        sub = m.loc[mod]
        k = sub.cor_auc.idxmax()
        runs = E[(E.score == sc) & (E.model == mod) & (E.latent == k)]
        if mod == "pca":
            add("Latent-code score" if first else "", f"{LAB[sc]}: PCA, {k}", f3(runs.raw_auc.iloc[0]),
                f3(runs.cor_auc.iloc[0]), pc(runs.cor_eff.iloc[0]))
        else:
            add("Latent-code score" if first else "", f"{LAB[sc]}: {NAME[mod]}, {k}", f3(runs.raw_auc.mean()),
                ms(runs.cor_auc), ms(runs.cor_eff, True))
        first = False
rows.append("    \\addlinespace")
# masked score
r = G[(G.model == "pca") & (G.latent == 4)].iloc[0]
add("Masked score", "PCA, 4", f3(r.raw_auc), f3(r.cor_auc), pc(r.cor_eff))
nm = G[G.model != "pca"].groupby(["model", "latent"])[["raw_auc", "cor_auc", "cor_eff"]].mean()
add("", "AE, VAE (all $k$)", rng3(nm.raw_auc), rng3(nm.cor_auc), rngp(nm.cor_eff))
rows.append("    \\addlinespace")
# no direct jet inputs
for k in (2, 4, 8, 16):
    r = H[(H.model == "pca") & (H.latent == k)].iloc[0]
    add("No jet inputs" if k == 2 else "", f"PCA, {k}", f3(r.raw_auc), f3(r.cor_auc), pc(r.cor_eff))
for mod in ("ae", "vae"):
    h = H[H.model == mod]
    if len(h):
        add("", f"{NAME[mod]}, 8 ({len(h)} seeds)", ms(h.raw_auc), ms(h.cor_auc), ms(h.cor_eff, True))

checks = r"""\begin{table}[tbp]
  \centering
  \footnotesize
  \caption[Further checks of the jet-count finding]{\textbf{Further checks of the jet-count finding}\newline Baselines, latent-code scores, a masked score and retraining without jet inputs all give corrected results near chance; only PCA keeps a small edge.}
    \label{tab:checks}
  \setlength{\tabcolsep}{4pt}
  \begin{tabular}{llccc}
    \toprule
    Check & Score or model, $k$ & Raw AUC & Corrected AUC & Corrected eff.\ (\%) \\
    \midrule
""" + "\n".join(rows) + r"""
    \bottomrule
  \end{tabular}
  \par\vspace{3pt}{\footnotesize\raggedright\textit{Note.} Evaluated on \texttt{sig\_eval} except the confirmation rows, which use \texttt{sig\_clf} (95\% bootstrap intervals in brackets). Corrected values use the percentile within jet count; eff., signal efficiency at a 1\% false-positive rate (chance 1.00\%). Latent-code rows show the latent size with the highest corrected AUC for each score and model. Ranges are over configuration means; $\pm$ values are standard deviations over seeds. NLL, negative log-likelihood; ELBO, evidence lower bound. Computed with Python 3.11, PyTorch 2.14.1 and scikit-learn 1.9.1.\par}
\end{table}
"""

# energy-scale table
erows = []
for shift in ("all energies", "jets only"):
    for f in (1.02, 1.05, 1.10, 1.20):
        x = I[(I["shift"] == shift) & (I.factor == f)]
        cells = []
        for mod in ("pca", "ae", "vae"):
            cells.append(f3(x[(x.model == mod) & (x.latent == 8)].raw_auc.mean()))
        label = ("All energy-like features" if shift == "all energies" else "Jet features only") if f == 1.02 else ""
        erows.append(f"    {label} & {f:.2f} & "
                     + " & ".join(cells) + f" & {rng3(x.raw_auc)} & {pc(x.raw_eff.max())} & {pc(x.cor_eff.max())} \\\\")
    if shift == "all energies":
        erows.append("    \\addlinespace")
energy = r"""\begin{table}[tbp]
  \centering
  \footnotesize
  \caption[Correlated energy-scale shifts]{\textbf{Correlated energy-scale shifts}\newline Scaling energies by up to 20\% while keeping correlations leaves every detector near chance.}
    \label{tab:energy}
  \setlength{\tabcolsep}{4pt}
  \begin{tabular}{lcccccc}
    \toprule
    Features scaled & Factor & PCA, 8 & AE, 8 & VAE, 8 & All runs & Max.\ eff.\ raw / corrected (\%) \\
    \midrule
""" + "\n".join(erows).replace(" & {", " & ") + r"""
    \bottomrule
  \end{tabular}
  \par\vspace{3pt}{\footnotesize\raggedright\textit{Note.} Entries are raw AUC (area under the receiver operating characteristic curve) for shifted against unshifted halves of the test background; the latent-size-8 columns are means over seeds, and ``All runs'' gives the range over all 28 runs. The last two columns give the highest share of shifted events caught at a 1\% false-positive rate over all runs, raw and after the jet-count correction. Energy-like features are the 14 transverse momenta, masses and energy sums; jet features are the four among them that describe jets. Computed with Python 3.11, PyTorch 2.14.1 and scikit-learn 1.9.1.\par}
\end{table}
"""
energy = energy.replace("Max.\\ eff.\\ raw / corrected (\\%) \\\\", "Max.\\ eff.\\ raw (\\%) & Max.\\ eff.\\ corrected (\\%) \\\\").replace("{lcccccc}", "{lccccccc}")
Path(R / "tables.tex").write_text(checks + "\n" + energy)
print(checks + "\n" + energy)
