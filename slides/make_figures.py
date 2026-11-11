"""Generate the figures of pca_slides.tex from the real data with the package's own compute_pca.

Run from the repository root:  CUDA_VISIBLE_DEVICES= .venv/bin/python slides/make_figures.py
Reads /mnt/ceph_drive/processed read-only; writes slides/figures/*.pdf and slides/figures/numbers.txt.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")     # CPU only: leave the GPU to the other services

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from tof_sims_driver.figures import ellipse_params      # noqa: E402
from tof_sims_driver.pca import compute_pca              # noqa: E402
from tof_sims_driver.store import Store                  # noqa: E402

DATA = "/mnt/ceph_drive/processed"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
GROUPS = ["6PPD", "6PPD-Q", "Si_control", "cyclohexylmethylamine", "phenylurea"]
COLORS = dict(zip(GROUPS, ["#c0392b", "#2c6fbb", "#2e8b57", "#c47f17", "#6f42c1"]))
BIN = 1.0
NUMBERS = []

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "figure.dpi": 150})


def note(s):
    print(s)
    NUMBERS.append(s)


store = Store(DATA)
names, labels = [], []
for g in GROUPS:
    for n in sorted(store.samples):
        if re.fullmatch(re.escape(g) + r"_pos\d+", n):
            names.append(n)
            labels.append(g)
note(f"subsamples: {len(names)} ({', '.join(f'{g}={labels.count(g)}' for g in GROUPS)}), positive polarity")

# ------------------------------------------------------------------ All-peak PCA
res_all = compute_pca(store, names, labels, 5, "tic", True, mode="all", bin_width=BIN, figures=False)
note(f"all-peak: {res_all['n_features']} features kept of {res_all['steps'][0]['detail']}")
st = res_all["steps"]
note("all-peak EV% = " + ", ".join(f"{100 * v:.1f}" for v in res_all["explained"]))

ratio = np.array([np.inf if r is None else r for r in res_all["peak_ratio"]])
frac = np.array(res_all["peak_frac"])
passing = (ratio >= res_all["ratio_min"]) & (frac >= res_all["frac_min"])
mz = np.array(res_all["peak_mz"])
note(f"passing the quality filter (ratio >= {res_all['ratio_min']:g}, share >= {100 * res_all['frac_min']:g}%): "
     f"{int(passing.sum())} of {len(mz)} bins at {BIN:g} u")

# Selected peaks = the app's default pre-selection: the TOP_PRESELECT bins with the largest max_k |loading|
from tof_sims_driver.pca import TOP_PRESELECT           # noqa: E402
influence = np.abs(np.array(res_all["comps"])).max(0)
pick = np.sort(np.argsort(-influence)[:TOP_PRESELECT])
peaks = [dict(name="", lo=float(mz[i] - BIN / 2), hi=float(mz[i] + BIN / 2)) for i in pick]
res_sel = compute_pca(store, names, labels, 5, "tic", True, mode="mine", peaks=peaks, figures=False)
note(f"selected-peak: {res_sel['n_features']} peaks (top {TOP_PRESELECT} by max |loading|)")
note("selected-peak EV% = " + ", ".join(f"{100 * v:.1f}" for v in res_sel["explained"]))
share_sel = float(np.sum(frac[pick]))
note(f"ion share carried by the selected bins: {100 * share_sel:.1f}%")
note("selected m/z: " + ", ".join(f"{mz[i]:.0f}" for i in pick))


def scores_plot(res, path, title):
    sc = np.array(res["scores"])
    ev = res["explained"]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for g in GROUPS:
        idx = [i for i, l in enumerate(res["labels"]) if l == g]
        ax.scatter(sc[idx, 0], sc[idx, 1], s=26, color=COLORS[g], label=g, zorder=3)
        e = ellipse_params(sc[idx, 0], sc[idx, 1])
        if e:
            ax.add_patch(Ellipse((e["cx"], e["cy"]), 2 * e["semi_major"], 2 * e["semi_minor"], angle=e["angle_deg"],
                                 facecolor=COLORS[g], alpha=0.13, edgecolor=COLORS[g], lw=1))
    ax.axhline(0, color="#999", lw=0.6)
    ax.axvline(0, color="#999", lw=0.6)
    ax.set_xlabel(f"PC1 ({100 * ev[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({100 * ev[1]:.1f}%)")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


scores_plot(res_all, OUT / "scores_all.pdf", f"All-peak PCA ({res_all['n_features']} bins)")
scores_plot(res_sel, OUT / "scores_sel.pdf", f"Selected-peak PCA ({res_sel['n_features']} peaks)")


def loadings_plot(res, path, title, k=10):
    c = np.array(res["comps"][0])
    m = np.array(res["peak_mz"])
    idx = np.argsort(-np.abs(c))[:k][::-1]
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.barh([f"{m[i]:.1f}" for i in idx], c[idx], color=["#2e8b57" if c[i] > 0 else "#c0392b" for i in idx])
    ax.axvline(0, color="#555", lw=0.7)
    ax.set_xlabel("PC1 loading")
    ax.set_ylabel("m/z")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


loadings_plot(res_all, OUT / "loadings_all.pdf", "All-peak: top 10 |PC1| loadings")
loadings_plot(res_sel, OUT / "loadings_sel.pdf", "Selected-peak: top 10 |PC1| loadings")


def scree_plot(path):
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    w = 0.38
    k = np.arange(1, 6)
    ax.bar(k - w / 2, 100 * np.array(res_all["explained"]), w, label=f"All-peak ({res_all['n_features']} bins)", color="#7f8c9a")
    ax.bar(k + w / 2, 100 * np.array(res_sel["explained"]), w, label=f"Selected-peak ({res_sel['n_features']} peaks)", color="#1f4e79")
    ax.set_xticks(k)
    ax.set_xticklabels([f"PC{i}" for i in k])
    ax.set_ylabel("explained variance (%)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


scree_plot(OUT / "scree.pdf")


# ------------------------------------------------------------------ preprocessing effect on real spectra
X = np.vstack([store.coarse_vector(n, BIN, 900.0) for n in names]).astype(float)
keep = X.sum(0) > 0
Xtic = X / X.sum(1, keepdims=True)
Xsq = np.sqrt(Xtic)


def within_group_mean_var(M):
    """mean and variance of each bin over the replicates of a group, averaged over groups (replicate spread)."""
    mus, vs = [], []
    for g in GROUPS:
        idx = [i for i, l in enumerate(labels) if l == g]
        mus.append(M[idx].mean(0))
        vs.append(M[idx].var(0, ddof=1))
    return np.mean(mus, 0), np.mean(vs, 0)


mu_t, v_t = within_group_mean_var(Xtic)
mu_s, v_s = within_group_mean_var(Xsq)
ok = (mu_t > 0) & (v_t > 0)
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5))
axes[0].loglog(mu_t[ok], v_t[ok], ".", ms=3, color="#7f8c9a")
axes[0].set_xlabel("mean of a bin (TIC-normalised)")
axes[0].set_ylabel("variance between replicates")
axes[0].set_title("after normalisation only", fontsize=11)
ok2 = (mu_s > 0) & (v_s > 0)
axes[1].loglog(mu_s[ok2], v_s[ok2], ".", ms=3, color="#1f4e79")
axes[1].set_xlabel(r"mean of a bin ($\sqrt{\ }$ of TIC-normalised)")
axes[1].set_title("after normalisation + square root", fontsize=11)
for a in axes:
    a.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(OUT / "variance_sqrt.pdf")
plt.close(fig)
span = lambda v: np.log10(np.percentile(v[v > 0], 99) / np.percentile(v[v > 0], 1))
note(f"replicate variance spread (99th/1st percentile, decades): TIC only {span(v_t[ok]):.1f}, TIC + sqrt {span(v_s[ok2]):.1f}")

# one spectrum through the three transforms: the largest bins dominate until the root is taken
j = int(np.argmax(res_all["peak_counts"]))
s = int(names.index("6PPD_pos1")) if "6PPD_pos1" in names else 0
fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.9), sharex=True)
xm = (np.arange(X.shape[1]) + 0.5) * BIN
for a, (y, t) in zip(axes, [(X[s], "raw counts $x_{ij}$"), (Xtic[s], r"TIC-normalised $\tilde{x}_{ij}$"),
                            (Xsq[s], r"square root $y_{ij}=\sqrt{\tilde{x}_{ij}}$")]):
    a.vlines(xm, 0, y, lw=0.6, color="#1f4e79")
    a.set_title(t, fontsize=11)
    a.set_xlabel("m/z (u)")
axes[0].set_ylabel(names[s])
fig.tight_layout()
fig.savefig(OUT / "spectrum_steps.pdf")
plt.close(fig)
top5 = np.sort(Xtic[s])[::-1][:5].sum()
top5s = np.sort(Xsq[s])[::-1][:5].sum() / Xsq[s].sum()
note(f"{names[s]}: top-5 bins hold {100 * top5:.0f}% of the ions (after TIC) but {100 * top5s:.0f}% of the summed root intensity")

# per-step ion statistics used on the slides
tot = [store.total_ions(n) for n in names]
note(f"ions per subsample: min {min(tot):,}, max {max(tot):,} (ratio {max(tot) / min(tot):.1f}x) -> why TIC normalisation")
(OUT / "numbers.txt").write_text("\n".join(NUMBERS) + "\n")
print("done")
