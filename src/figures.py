"""Paper figures. Reads results/*.csv and results/latents/*.npz, writes figures/*.png and *.pdf.

Colour rules: one fixed colour AND marker per model in every figure (validated palette,
slots 1 to 3), so colour is never the only cue. Jet count (ordered) uses one blue ramp.
Reference lines (chance etc.) are dashed grey, explained in the legend or labelled.
"""
import traceback
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

R, F = Path("results"), Path("figures")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ORDER = ["pca", "ae", "vae"]
COLOR = {"ae": "#2a78d6", "pca": "#eb6834", "vae": "#1baf7a"}
MARK = {"ae": "o", "pca": "s", "vae": "^"}
LABEL = {"ae": "Autoencoder", "pca": "PCA", "vae": "VAE"}
LAT = [2, 4, 8, 16]
JET_COLORS = ["#86b6ef", "#5598e7", "#256abf", "#104281"]   # blue ramp, light to dark
BG_GREY, SIGNAL = "#c5c4bd", "#6250d6"


# ---------- shared helpers ----------
def style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.size": 9, "text.color": INK, "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "axes.titlesize": 9.5, "axes.titlelocation": "left",
    })


def save(fig, name):
    F.mkdir(exist_ok=True)
    fig.savefig(F / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(F / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def add_legend(fig, handles=None, chance=False):
    """One shared legend above the panels (always present for 2+ series)."""
    handles = handles or [Line2D([0], [0], color=COLOR[m], marker=MARK[m], lw=2, ms=6,
                                 mec="white", label=LABEL[m]) for m in ORDER]
    if chance:                                   # dashed reference line explained once here
        handles = handles + [Line2D([0], [0], color=MUTED, lw=1, ls="--", label="Chance")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, 1.0))


def mean_std(df, x, y):
    """Mean and spread over seeds. PCA is one deterministic fit, so its spread is 0."""
    g = df.groupby(["model", x])[y].agg(["mean", "std"]).reset_index()
    g["std"] = g["std"].fillna(0)
    return g


def draw_lines(ax, df, x, y):
    """One line per model: mean with error bars, 2px line, marker with a white ring."""
    g = mean_std(df, x, y)
    for m in ORDER:
        s = g[g.model == m]
        if len(s):
            ax.errorbar(s[x], s["mean"], yerr=s["std"], color=COLOR[m], marker=MARK[m],
                        ms=6, lw=2, capsize=2, elinewidth=1, mec="white", mew=1, zorder=3)


def ref_line(ax, y, text, below=False):
    """Dashed grey reference line; a small label (above or below it) if text is given."""
    ax.axhline(y, color=MUTED, lw=1, ls="--", zorder=2)
    if text:
        ax.annotate(text, xy=(0.99, y), xycoords=ax.get_yaxis_transform(),
                    xytext=(0, -3 if below else 3), textcoords="offset points",
                    ha="right", va="top" if below else "bottom", fontsize=7.5, color=MUTED)


def log_x(ax):
    """Plain-number ticks (0.2, 0.5, 1) on a log axis, no crowded 10^-1 labels."""
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def latent_axis(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(LAT)
    ax.set_xticklabels(LAT)
    ax.minorticks_off()
    ax.set_xlabel("Latent size")


# ---------- figure 1: better rebuild is not better detection ----------
def fig_recon_vs_detection():
    res = pd.read_csv(R / "results.csv")
    pct = pd.read_csv(R / "stratified.csv").query("group == 'pct_global'")
    pct = pct[["model", "latent", "seed", "auc"]].rename(columns={"auc": "auc_jet"})
    df = res.merge(pct, on=["model", "latent", "seed"]).dropna(subset=["val_recon"])
    df = df[df.val_recon > 0]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharex=True, sharey=True)
    for ax, col, title in zip(axes, ["auc", "auc_jet"], ["a  Raw score", "b  Jet count removed"]):
        for m in ORDER:
            s = df[df.model == m]
            ax.scatter(s.val_recon, s[col], color=COLOR[m], marker=MARK[m], s=38,
                       edgecolor="white", linewidth=1, zorder=3)
        ref_line(ax, 0.5, None)
        log_x(ax)
        ax.set_title(title)
    axes[0].set_ylabel("AUC on proxy signal")
    fig.supxlabel("Validation reconstruction error (lower = better rebuild)", fontsize=9,
                  color=MUTED, y=-0.04)
    add_legend(fig, chance=True)
    save(fig, "fig1_recon_vs_detection")


# ---------- figure 2: the jet-count confound ----------
def fig_jet_confound():
    s = pd.read_csv(R / "stratified.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    for ax, grp, title in zip(axes, ["raw_global", "pct_global"],
                              ["a  Raw score", "b  Score as percentile within jet count"]):
        draw_lines(ax, s[s.group == grp], "latent", "auc")
        ref_line(ax, 0.5, None)
        latent_axis(ax)
        ax.set_title(title)
    axes[0].set_ylabel("AUC on proxy signal")
    add_legend(fig, chance=True)
    save(fig, "fig2_jet_confound")


# ---------- figure 3: synthetic shift detectability ----------
def fig_shift_curves():
    s = pd.read_csv(R / "shift.csv").query("latent == 8")
    counts = [3, 10, int(s.n_feat.max())]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), sharey=True)
    for ax, n in zip(axes, counts):
        draw_lines(ax, s[s.n_feat == n], "shift", "auc")
        ref_line(ax, 0.5, None)
        ax.set_title(f"{n} features shifted")
        ax.set_xlabel("Shift size (std. dev.)")
    axes[0].set_ylabel("AUC (latent size 8)")
    add_legend(fig, chance=True)
    save(fig, "fig3_shift_curves")


# ---------- figure 4: does latent size help detect shifts? ----------
def fig_shift_by_latent():
    s = pd.read_csv(R / "shift.csv")
    s["caught"] = s["eff_at_fpr_0.01"] * 100
    n_all = int(s.n_feat.max())
    cases = [(3, 2.0, "a  3 features, shift 2.0"), (n_all, 1.0, f"b  {n_all} features, shift 1.0")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for ax, (n, k, title) in zip(axes, cases):
        draw_lines(ax, s[(s.n_feat == n) & (s["shift"] == k)], "latent", "caught")
        ref_line(ax, 1.0, None)
        latent_axis(ax)
        ax.set_title(title)
    axes[0].set_ylabel("Shifted events caught at 1% false alarms (%)")
    add_legend(fig, chance=True)
    save(fig, "fig4_shift_by_latent")


# ---------- figure 5: what the codes contain ----------
def fig_probes():
    s = pd.read_csv(R / "latent_summary.csv")
    nets, ref = s[s.model != "input"], s[s.model == "input"].iloc[0]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9))
    draw_lines(axes[0], nets, "latent", "sig_auc_within_jet")
    ref_line(axes[0], 0.5, "chance")
    ref_line(axes[0], ref.sig_auc_within_jet, "raw inputs")
    axes[0].set_title("a  Signal, linear probe")
    axes[0].set_ylabel("AUC within jet count")
    draw_lines(axes[1], nets, "latent", "jet_acc")
    ref_line(axes[1], nets.jet_majority.iloc[0], "always guess commonest")
    axes[1].set_title("b  Jet count readout")
    axes[1].set_ylabel("Accuracy")
    draw_lines(axes[2], nets[nets.model == "vae"], "latent", "active_dims")
    axes[2].plot(LAT, LAT, color=MUTED, lw=1, ls="--", zorder=2)
    axes[2].text(16, 14.2, "all used", ha="right", fontsize=7.5, color=MUTED)
    axes[2].set_title("c  VAE dimensions in use")
    axes[2].set_ylabel("Active dimensions")
    for ax in axes:
        latent_axis(ax)
    fig.tight_layout()
    add_legend(fig)
    save(fig, "fig5_probes")


# ---------- figure 6: which physics features survive in the code ----------
def fig_feature_r2():
    f = pd.read_csv(R / "latent_features.csv").query("latent == 8")
    t = f.groupby(["feature", "model"]).r2.mean().unstack()[ORDER]
    t = t.loc[t.mean(axis=1).sort_values(ascending=False).index]
    cmap = LinearSegmentedColormap.from_list(
        "blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    fig, ax = plt.subplots(figsize=(4.4, 0.27 * len(t) + 1.2))
    im = ax.imshow(t.values.clip(0, 1), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ORDER)), [LABEL[m] for m in ORDER])
    ax.set_yticks(range(len(t)), t.index)
    ax.grid(False)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            v = t.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if v > 0.55 else INK)
    fig.colorbar(im, ax=ax, label="R² of linear readout (latent size 8)", shrink=0.6)
    save(fig, "fig6_feature_r2")


# ---------- figures 7 and 8: 2D maps of the latent codes ----------
@lru_cache(maxsize=None)
def projection(model, how):
    """2D map of a saved latent sample. UMAP is slow, so each one is computed once."""
    z = np.load(R / "latents" / f"{model}_z8_s0.npz")["z"]
    z = StandardScaler().fit_transform(z)
    if how == "PCA":
        return PCA(n_components=2).fit_transform(z)
    import umap
    return umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0).fit_transform(z)


def latent_grid(paint, handles, name):
    """3 rows (models) by 2 columns (PCA map, UMAP map); `paint` draws one panel."""
    fig, axes = plt.subplots(3, 2, figsize=(6.2, 8.4))
    fig.subplots_adjust(top=0.94, hspace=0.12, wspace=0.08)
    for r, m in enumerate(ORDER):
        d = np.load(R / "latents" / f"{m}_z8_s0.npz")
        for c, how in enumerate(["PCA", "UMAP"]):
            ax = axes[r, c]
            paint(ax, projection(m, how), d)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if r == 0:
                ax.set_title(f"{how} map of the code")
        axes[r, 0].set_ylabel(LABEL[m])
    add_legend(fig, handles)
    save(fig, name)


def fig_latent_class():
    def paint(ax, xy, d):
        bg = d["is_signal"] == 0
        for mask, col in ((bg, BG_GREY), (~bg, SIGNAL)):          # signal drawn on top
            ax.scatter(xy[mask, 0], xy[mask, 1], s=3, color=col, alpha=0.5,
                       linewidths=0, rasterized=True)
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=BG_GREY, mec="none",
                      ms=7, label="Background"),
               Line2D([0], [0], marker="o", color="none", markerfacecolor=SIGNAL, mec="none",
                      ms=7, label="Proxy signal")]
    latent_grid(paint, handles, "fig7_latent_class")


def fig_latent_jets():
    def paint(ax, xy, d):
        bg = np.where(d["is_signal"] == 0)[0]
        bg = np.random.default_rng(0).permutation(bg)               # no jet bin drawn last
        cols = np.array(JET_COLORS)[d["jet"][bg].astype(int)]
        ax.scatter(xy[bg, 0], xy[bg, 1], s=3, color=cols, alpha=0.6,
                   linewidths=0, rasterized=True)
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=c, mec="none", ms=7,
                      label=f"{j} jet" + ("" if j == 1 else "s"))
               for j, c in enumerate(JET_COLORS)]
    latent_grid(paint, handles, "fig8_latent_jets")


FIGS = [fig_recon_vs_detection, fig_jet_confound, fig_shift_curves, fig_shift_by_latent,
        fig_probes, fig_feature_r2, fig_latent_class, fig_latent_jets]

if __name__ == "__main__":
    style()
    for fn in FIGS:
        try:
            fn()
        except Exception:                      # one bad figure should not stop the rest
            print("FAILED", fn.__name__)
            traceback.print_exc()