"""Render paper figures from per-backbone exports written by the notebooks."""

# Standard Library
from pathlib import Path

# Third party
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path("paper_data")
OUT_DIR  = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

BACKBONES   = ["DeMo", "Polaris", "QCNet"]
AGENT_TYPES = ["vehicle", "bus", "cyclist", "motorcyclist", "pedestrian"]

GREEN, RED  = "#1B7A3A", "#B22B2B"
TRAIN_COL, VAL_COL = "#1f4e79", "#2a9d4a"

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})

def load(bb):
    with open(DATA_DIR / f"{bb}.pkl", "rb") as f:
        return pickle.load(f)

data = {bb: load(bb) for bb in BACKBONES}


# -------- Fig: learning curves (1 x 3) --------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True, layout="constrained")
for ax, bb in zip(axes, BACKBONES):
    d = data[bb]
    tr, va = d["train_losses"], d["val_losses"]
    best = d["best_epoch"]
    ax.plot(tr, label="Train", color=TRAIN_COL, lw=1.4)
    ax.plot(va, label="Validation",   color=VAL_COL,   lw=1.4)
    ax.axvline(best, color="gray", ls="--", lw=0.8, label=f"Best (ep. {best})")
    ax.set_title(bb)
    ax.set_xlabel("Epoch")
    ax.set_ylim(0.35, 0.66)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
axes[0].set_ylabel("BCE Loss")
fig.savefig(OUT_DIR / "fig_learning_curves.pdf")
plt.close(fig)


# -------- Fig: tail-percentile per backbone (1 x 3) -------------------------
from statsmodels.nonparametric.smoothers_lowess import lowess

fig, axes = plt.subplots(1, 3, figsize=(9, 2), sharey=True, layout="constrained")
for ax, bb in zip(axes, BACKBONES):
    res = data[bb]["results"]
    b = res["backbone_score"].to_numpy()
    delta = res["delta"].to_numpy()
    # per-scenario backbone-error percentile (0..100), no binning
    pct = (b.argsort().argsort() / (len(b) - 1)) * 100.0
    # LOWESS = conditional mean of delta given percentile, smoothed
    sm = lowess(delta, pct, frac=0.125, it=0, return_sorted=True)
    xs, ys = sm[:, 0], sm[:, 1]
    ax.fill_between(xs, ys, 0, where=ys <= 0, color=GREEN, alpha=0.25, interpolate=True)
    ax.fill_between(xs, ys, 0, where=ys >= 0, color=RED,   alpha=0.25, interpolate=True)
    ax.plot(xs, ys, color="#1a1a1a", lw=1.6)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlim(0, 100)
    ax.set_xlabel(f"{bb} Error Percentile")
    ax.set_ylim(-0.1, 0.02)
axes[0].set_ylabel(r"Mean $\Delta$ bMinADE$_1$")
fig.savefig(OUT_DIR / "percentile_difference.pdf")
plt.close(fig)


# -------- Fig: per-class relative-error distribution (3 x 5) ----------------
# One density panel per (backbone, agent_type). Restricted to scenarios where the
# gate actually fired (w > 0) so the plot is "what does fusion *do*, when it acts."
from scipy.stats import gaussian_kde

x_range = np.linspace(-0.5, 0.5, 401)
fig, axes = plt.subplots(len(BACKBONES), len(AGENT_TYPES),
                         figsize=(9, 3), sharex=True, sharey="row",
                         layout="constrained")
for i, bb in enumerate(BACKBONES):
    res = data[bb]["results"]
    res = res[res["weight"] > 0]   # gate fired
    for j, ag in enumerate(AGENT_TYPES):
        sub = res[res["agent_type"] == ag]
        ax = axes[i, j]
        if len(sub) < 5 or sub["fusion_score"].std() < 1e-6:
            ax.set_visible(False); continue

        rel = ((sub["fusion_score"] - sub["backbone_score"])
               / sub["backbone_score"].clip(lower=1e-6)).to_numpy()
        rel = np.clip(rel, -0.5, 0.5)

        kde = gaussian_kde(rel, bw_method=0.1)   # tune for visual sharpness
        ys = kde(x_range)
        ax.fill_between(x_range, ys, 0, where=x_range <= 0,
                        color=GREEN, alpha=0.55, interpolate=True)
        ax.fill_between(x_range, ys, 0, where=x_range >= 0,
                        color=RED,   alpha=0.55, interpolate=True)
        ax.plot(x_range, ys, color="#222", lw=1.0)
        # ax.axvline(0, color="k", lw=0.7)

        med = float(np.median(rel))
        ax.text(0.97, 0.92,
                f"n={len(sub)}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=10, color="#444")
        if i == 0:                       ax.set_title(ag.capitalize(), fontsize=11)
        if j == 0:                       ax.set_ylabel(bb, fontsize=11)
        if i == len(BACKBONES) - 1:
            ax.set_xticks([-0.5, 0, 0.5])
            ax.set_xticklabels([r"$-50\%$", "0", r"$+50\%$"])
            ax.set_xlabel(r"Relative $\Delta$ bMinADE$_1$")
        ax.set_yticks([])    # density units aren't paper-meaningful; shape is
fig.savefig(OUT_DIR / "fig_per_type_distribution.pdf")
plt.close(fig)

# -------- Fig: gating sweep per backbone (1 x 3) -----------------------------
ULO = np.log(1/6)

def _brier_minade1(log_bb, log_tr, ade_mat, w):
    fused = log_bb + w[:, None] * (log_tr - ULO)
    fused = np.exp(fused - fused.max(1, keepdims=True)); fused /= fused.sum(1, keepdims=True)
    idx = fused.argmax(1)
    sel = np.take_along_axis(fused, idx[:, None], 1).squeeze(1)
    sel_ade = np.take_along_axis(ade_mat, idx[:, None], 1).squeeze(1)
    return sel_ade + (1 - sel) ** 2

ts = np.round(np.arange(0.0, 0.96, 0.05), 2)

fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.4), sharey=True, layout="constrained")
for ax, bb in zip(axes, BACKBONES):
    d = data[bb]
    w_raw   = d["w_raw"]
    ade_mat = d["ade_mat"]
    log_bb  = np.log(d["bb_mat"] + 1e-10)
    log_tr  = np.log(d["tr_mat"] + 1e-10)
    agent   = d["agent_arr"]
    vtypes  = sorted(set(agent))

    bb_score = _brier_minade1(log_bb, log_tr, ade_mat, np.zeros(len(w_raw)))
    ov, bal  = [], []
    for t in ts:
        delta = _brier_minade1(log_bb, log_tr, ade_mat, np.where(w_raw >= t, 1.0, 0.0)) - bb_score
        ov.append(delta.mean())
        bal.append(np.mean([delta[agent == v].mean() for v in vtypes]))
    ov, bal = np.asarray(ov), np.asarray(bal)

    ax.axhline(0, color="grey", lw=0.8)
    ax.fill_between(ts, np.minimum(0, np.minimum(ov, bal)), 0, color=GREEN, alpha=0.10)
    ax.plot(ts, ov,  "-o", ms=3.5, color=TRAIN_COL, label="Overall")
    ax.plot(ts, bal, "-s", ms=3.5, color=GREEN,     label="Balanced")
    # ax.axvline(d["threshold"], color="k", ls="--", lw=0.9,
    #            label=fr"$\theta = {d['threshold']:.2f}$")
    ax.set_title(bb)
    ax.set_xlabel(r"Gate Threshold $\theta$")
    ax.set_xticks(np.arange(0, 1.1, 0.25))
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
axes[0].set_ylabel(r"$\Delta$ bMinADE$_1$")
fig.savefig(OUT_DIR / "fig_gating_sweep.pdf")
plt.close(fig)

print("wrote:", *(p.name for p in sorted(OUT_DIR.glob("fig_*.pdf"))))