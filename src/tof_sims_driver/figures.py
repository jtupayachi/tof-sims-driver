"""Matplotlib figures for a PCA result (see :func:`tof_sims_driver.pca.compute_pca`).

Each figure is returned as ``{id, title, caption, formula, formula_png, png}``: ``png`` / ``formula_png``
are data URIs, ``formula`` the list of mathtext (LaTeX) lines behind the plot.
"""
import base64
import io
import math
import re
import textwrap
import threading

import numpy as np

PAL = ["#2b6cb0", "#c53030", "#2f855a", "#b7791f", "#6b46c1", "#319795", "#d53f8c", "#718096"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
_FIG_LOCK = threading.Lock()


def _fmt(name):
    """C_13H_18N_2O- -> mathtext subscripts / charge superscript; plain text if it does not parse."""
    from matplotlib.mathtext import MathTextParser
    plain = str(name or "").replace("$", "")
    s = re.sub(r"_(\d+)", r"$_{\1}$", plain)
    s = re.sub(r"([+-])$", r"$^{\1}$", s)
    try:
        MathTextParser("path").parse(s)
        return s
    except Exception:
        return plain


def _grid_size(k, base=2.6, total=22.0, lo=1.5):
    """Cell size (inches) of a k x k panel matrix, shrinking so the figure stays <= `total` inches."""
    return max(lo, min(base, total / max(k, 1)))


def _dpi_for(fig, cap_px=9000, hi=120):
    """Highest dpi (<= hi) that keeps the longest side of the PNG under cap_px pixels."""
    w, h = fig.get_size_inches()
    return int(max(60, min(hi, cap_px / max(w, h))))


def _tex_png(lines):
    """Render mathtext (LaTeX subset) lines to a PNG data URI; offline, no LaTeX install needed."""
    from matplotlib.figure import Figure
    from matplotlib.mathtext import MathTextParser
    n = len(lines)
    fig = Figure(figsize=(14, 0.55 * n + 0.2), facecolor="#f7f8fb")
    for i, s in enumerate(lines):
        try:
            MathTextParser("path").parse(s)
        except Exception:
            s = re.sub(r"[$\\{}]", "", s)
        fig.text(0.0, 1 - (i + 0.5) / n, s, fontsize=16, va="center", ha="left")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight", pad_inches=0.1)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# (dx, dy) offsets in points tried in order; dy is flipped for labels that go below their point
_CANDS = [(0, 5), (6, 5), (-6, 5), (0, 15), (12, 15), (-12, 15), (0, 26), (20, 26), (-20, 26), (0, 38),
          (30, 38), (-30, 38), (34, 0), (-34, 0), (0, 52), (40, 52), (-40, 52)]


class _Packer:
    """Greedy label placement: a label is kept only where it overlaps no earlier label and stays inside
    its axes (or the figure). Labels that do not fit are counted in ``skipped``."""

    def __init__(self, fig):
        fig.draw_without_rendering()                      # resolve the constrained layout first
        self.fig, self.r, self.taken, self.skipped = fig, fig._get_renderer(), [], 0

    def put(self, ax, xy, s, cands=None, fs=8, rot=0, color="#222", within="ax", down=False):
        from matplotlib.transforms import offset_copy
        box = ax.get_window_extent(self.r) if within == "ax" else self.fig.bbox
        for dx, dy in (cands or _CANDS):
            dy = -dy if down else dy
            ha = "left" if dx > 1 else "right" if dx < -1 else "center"
            va = "bottom" if dy > 1 else "top" if dy < -1 else "center"
            t = ax.text(xy[0], xy[1], s, fontsize=fs, rotation=rot, color=color, ha=ha, va=va, zorder=5,
                        clip_on=False, transform=offset_copy(ax.transData, fig=self.fig, x=dx, y=dy, units="points"))
            bb = t.get_window_extent(self.r).expanded(1.05, 1.15)
            if (bb.x0 < box.x0 or bb.x1 > box.x1 or bb.y0 < box.y0 or bb.y1 > box.y1
                    or any(bb.overlaps(o) for o in self.taken)):
                t.remove()
                continue
            self.taken.append(bb)
            if np.hypot(dx, dy) > 16:
                ax.annotate("", xy, xytext=(dx, dy), textcoords="offset points", zorder=4,
                            arrowprops=dict(arrowstyle="-", lw=0.5, color="#888"))
            return True
        self.skipped += 1
        return False


CHI2_90 = -2.0 * math.log(0.10)           # chi-square quantile (2 degrees of freedom, 90%) = 4.605


def ellipse_params(x, y):
    """90% ellipse of a group of points: the covariance ellipse scaled by sqrt(CHI2_90).
    Returns ``dict(n, cx, cy, semi_major, semi_minor, angle_deg)`` (angle of the major axis from the x axis, degrees),
    or None for fewer than 3 points or points on a line. For groups of 6 or fewer points every point lies inside by construction
    (a point's squared distance from the group centre is at most (n-1)^2/n, which is below CHI2_90 up to n = 6)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return None
    cov = np.cov(x, y)
    if not np.all(np.isfinite(cov)):
        return None
    vals, vecs = np.linalg.eigh(cov)                      # ascending
    if vals[0] <= 1e-12 * max(vals[1], 1e-300):
        return None
    k = math.sqrt(CHI2_90)
    return dict(n=int(len(x)), cx=float(x.mean()), cy=float(y.mean()), semi_major=k * math.sqrt(vals[1]),
                semi_minor=k * math.sqrt(vals[0]), angle_deg=math.degrees(math.atan2(vecs[1, 1], vecs[0, 1])))


def label_loadings(res):
    """Which features and PCs characterise each label group.

    The scores are mean-centred, so a group's mean score vector ``tbar_g`` places the group relative to the average file, and
    ``d_gj = sum_k tbar_gk * p_kj`` is how much feature j of that group's average spectrum lies above (+) or below (-) the
    all-file average in the PCA's units (the PCA's own reconstruction of the group). Term k of the sum is the part PC k carries.
    Returns None when there is no labelled group to contrast with the rest, otherwise a dict with ``groups``, ``n`` (files per
    group), ``tbar`` (groups x PCs), ``z`` (``tbar`` in units of each PC's score SD), ``share`` (groups x PCs, share of
    ``|tbar_g|^2`` per PC, rows sum to 1), ``d`` (groups x features) and ``parts`` (groups x PCs x features)."""
    scores = np.asarray(res["scores"], float)
    comps = np.asarray(res["comps"], float)
    n = len(res["names"])
    lab = [str(v).strip() for v in res.get("labels", [])] + [""] * n
    lab = lab[:n]
    groups = [g for g in dict.fromkeys(lab) if g]
    if not groups or (len(groups) == 1 and lab.count(groups[0]) == n):
        return None
    tbar = np.array([scores[[i for i in range(n) if lab[i] == g]].mean(0) for g in groups])
    sd = scores.std(0, ddof=1) if n > 1 else np.ones(scores.shape[1])
    sd = np.where(sd > 0, sd, 1.0)
    norm = (tbar ** 2).sum(1, keepdims=True)
    share = np.where(norm > 0, tbar ** 2 / np.where(norm > 0, norm, 1.0), 0.0)
    parts = tbar[:, :, None] * comps[None, :, :]
    return dict(groups=groups, n=[lab.count(g) for g in groups], tbar=tbar, z=tbar / sd, share=share,
                d=parts.sum(1), parts=parts)


def make_figures(res):
    """Render every PCA figure with matplotlib. Returns [{id, title, caption, formula, png}]."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import MaxNLocator
    nc = res["n_components"]
    n_lab = max(0, int(res.get("n_labels", 12)))
    ev = [100.0 * e for e in res["explained"]]
    scores = np.asarray(res["scores"], float)
    comps = np.asarray(res["comps"], float)
    mz = np.asarray(res["peak_mz"], float)
    pn = res["peak_names"]
    names = res["names"]
    n = len(names)
    glabels = [(str(lb).strip() or "(unlabeled)") for lb in res.get("labels", [])]
    glabels += ["(unlabeled)"] * (n - len(glabels))
    groups = list(dict.fromkeys(glabels))
    multi = len(groups) > 1
    by_shape = 1 < len(groups) <= len(MARKERS)               # a different marker per label only while it stays readable
    gmark = {g: (MARKERS[k] if by_shape else "o") for k, g in enumerate(groups)}
    gcol = {g: PAL[k % len(PAL)] for k, g in enumerate(groups)}      # colours repeat beyond 8 labels

    def fmz(x):                                           # integers without trailing ".0"
        x = float(x)
        return f"{x:.0f}" if abs(x - round(x)) < 1e-9 else f"{x:.2f}"

    def png(fig, hi=120):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=_dpi_for(fig, hi=hi), bbox_inches="tight")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    def slab(i):                                          # short label: name + m/z
        return f"{_fmt(pn[i])} {fmz(mz[i])}" if pn[i] else fmz(mz[i])

    def left_out(pk):
        return (f" {pk.skipped} label(s) were left out because they would overlap." if pk.skipped else "")

    def span(j):                                          # m/z range of a feature: "22-23" for a bin, "218.14" for a narrow window
        lo, hi = float(res["peak_lo"][j]), float(res["peak_hi"][j])
        return f"{lo:g}-{hi:g}" if hi - lo >= 0.4 else f"{(lo + hi) / 2:.2f}"

    out = []

    def add(fid, title, caption, formula, fig, hi=120):
        out.append(dict(id=fid, title=title, caption=f"Figure {len(out) + 1}. {caption}",
                        formula=formula, formula_png=_tex_png(formula), png=png(fig, hi)))

    def ellipse90(ax, x, y, color, lw=1.4, zorder=2):
        """Outline the 90% ellipse of a group of points (see :func:`ellipse_params`); False when it cannot be drawn.
        Drawn as a line (no fill) so that the axes autoscale to include the whole ellipse."""
        e = ellipse_params(x, y)
        if e is None:
            return False
        t_ = np.linspace(0, 2 * np.pi, 181)
        th = math.radians(e["angle_deg"])
        u, w = e["semi_major"] * np.cos(t_), e["semi_minor"] * np.sin(t_)
        ax.plot(e["cx"] + u * math.cos(th) - w * math.sin(th), e["cy"] + u * math.sin(th) + w * math.cos(th),
                color=color, lw=lw, zorder=zorder)
        return True

    def sparse(ax, nx=5, ny=5):
        """Few, well-separated ticks on both axes."""
        ax.xaxis.set_major_locator(MaxNLocator(nbins=nx))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=ny))

    def stick_panel(ax, c, fs=10):
        """One PC's loadings as a stick spectrum (positive up, negative down) in a framed panel with a zero line."""
        w = comps[c]
        ax.vlines(mz, 0, w, colors="#333", lw=0.9 if len(mz) > 120 else 1.5)
        ax.axhline(0, color="#888", lw=0.8)
        lim = float(np.abs(w).max()) or 1.0
        ax.set_ylim(-1.45 * lim, 1.45 * lim)
        ax.set_xlim(max(0.0, float(mz.min()) - 0.02 * float(mz.max())), float(mz.max()) * 1.04)
        sparse(ax, 8, 5)
        ax.tick_params(labelsize=fs - 1)
        ax.set_xlabel("m/z", fontsize=fs)
        ax.set_ylabel(f"PC{c + 1} Loadings ({ev[c]:.1f} %)", fontsize=fs)

    def label_sticks(pk, ax, c, fs=8):
        for i in np.argsort(-np.abs(comps[c]))[:n_lab]:
            pk.put(ax, (mz[i], comps[c][i]), slab(i), fs=fs, rot=90,
                   cands=[(0, 3), (5, 3), (-5, 3), (0, 36), (6, 36), (-6, 36), (0, 70), (6, 70), (-6, 70)],
                   down=bool(comps[c][i] < 0))

    LL = label_loadings(res)

    with _FIG_LOCK:
        # 1 - scree: explained variance of every PC and, per label, how its separation is shared between the PCs
        t = "Scree plot"
        two = LL is not None
        fig = Figure(figsize=(16 if two else 9, 5.6), layout="constrained")
        axs = fig.subplots(1, 2 if two else 1, squeeze=False)[0]
        ax = axs[0]
        xs = np.arange(1, nc + 1)
        step = 1 if nc <= 12 else 2
        ax.plot(xs, ev, color=PAL[0], lw=1, marker="o", ms=9, label="per component")
        for x_, v_ in zip(xs, ev):
            ax.annotate(f"{v_:.1f}%", (x_, v_), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
        ax.plot(xs, np.cumsum(ev), color=PAL[1], lw=1, marker="o", ms=6, label="cumulative")
        ax.set_xticks(xs[::step], [f"PC{i}" for i in xs[::step]])
        ax.set_xlabel("Principal component", fontsize=11)
        ax.set_ylabel("Explained variance (%)", fontsize=11)
        ax.set_ylim(0, 112)
        ax.grid(alpha=0.3)
        ax.set_title("All samples", fontsize=11)
        ax.legend(loc="center right")
        if two:
            ax = axs[1]
            for q, g in enumerate(LL["groups"]):
                ax.plot(xs, 100 * LL["share"][q], color=gcol[g], lw=1, marker=gmark[g], ms=8, label=g)
            ax.set_xticks(xs[::step], [f"PC{i}" for i in xs[::step]])
            ax.set_xlabel("Principal component", fontsize=11)
            ax.set_ylabel("Share of the label's separation (%)", fontsize=11)
            ax.set_ylim(0, 105)
            ax.grid(alpha=0.3)
            ax.set_title("Per label", fontsize=11)
            if len(LL["groups"]) <= 14:
                ax.legend(title="label", fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
        fig.suptitle(t, fontweight="bold", fontsize=13)
        add("scree", t, "Left: percentage of the total variance captured by each component (large dots) and their running "
            "total (small dots); components after the elbow carry mostly noise. "
            + ("Right: for each label (group), how its distance from the average file is shared between the PCs; "
               "a label whose dots peak at PCk is separated from the others mainly by PCk." if two else
               "Give the files labels to also see, per label, which PCs separate it."),
            [r"$\lambda_k = s_k^2\,/\,(n-1)$   ($s_k$ = $k$-th singular value of the mean-centred matrix $Z$)",
             r"$\mathrm{EV}_k = 100\,\lambda_k\,/\,\sum_l \lambda_l$  (%);   cumulative$_k = \sum_{l \leq k}\ \mathrm{EV}_l$",
             r"per label: $100\,\bar t_{gk}^2\ /\ \sum_l \bar t_{gl}^2$  ($\bar t_{gk}$ = mean score of label $g$ on PC $k$)"], fig)

        # 2 - scores of every pair of PCs, all labels coloured, one 90% ellipse per label
        drawn = 0
        if nc >= 2:
            t = "Scores by PC pair"
            kk = min(nc, 10)
            pairs = [(a, b) for a in range(kk) for b in range(a + 1, kk)]
            ncol = 3
            nrow = -(-len(pairs) // ncol)
            fig = Figure(figsize=(7.0 * ncol + 1.8, 6.0 * nrow + 0.8), layout="constrained")
            axs = fig.subplots(nrow, ncol, squeeze=False)
            for q in range(nrow * ncol):
                ax = axs[q // ncol][q % ncol]
                if q >= len(pairs):
                    ax.set_visible(False)
                    continue
                a, b = pairs[q]
                for g in groups:
                    ii = [i for i in range(n) if glabels[i] == g]
                    if g != "(unlabeled)" and ellipse90(ax, scores[ii, a], scores[ii, b], gcol[g]):
                        drawn += 1
                    ax.scatter(scores[ii, a], scores[ii, b], s=38, color=gcol[g], marker=gmark[g], edgecolor="white", lw=0.4, zorder=3)
                ax.axhline(0, color="#222", lw=0.8); ax.axvline(0, color="#222", lw=0.8)
                sparse(ax, 5, 5)
                ax.tick_params(labelsize=9)
                ax.set_xlabel(f"PC{a + 1} Scores ({ev[a]:.1f} %)", fontsize=10)
                ax.set_ylabel(f"PC{b + 1} Scores ({ev[b]:.1f} %)", fontsize=10)
            if multi and len(groups) <= 14:
                fig.legend(handles=[Line2D([], [], marker=gmark[g], ls="", color=gcol[g], label=g) for g in groups],
                           title="label", loc="outside right upper", fontsize=9)
            fig.suptitle(t, fontweight="bold", fontsize=14)
            pk3 = _Packer(fig)                              # the label (group) name at the centre of its files in every panel
            for q, (a, b) in enumerate(pairs):
                for g in groups:
                    if g != "(unlabeled)":
                        ii = [i for i in range(n) if glabels[i] == g]
                        pk3.put(axs[q // ncol][q % ncol], (scores[ii, a].mean(), scores[ii, b].mean()), g, fs=8, color=gcol[g],
                                cands=[(0, 7), (0, -7), (9, 0), (-9, 0)] + _CANDS)
            add("scores", t, f"One panel per pair of PCs ({'the first %d of %d PCs' % (kk, nc) if kk < nc else 'all PCs'}); x = the lower PC, y = the higher PC. "
                "One dot per file, coloured by its label, and every label is named at the centre of its files in every panel. "
                "The outlined ellipse around each label group is its 90% confidence ellipse, drawn from that group's own spread. "
                "An ellipse needs at least 3 files. With 6 or fewer files in a group every one of "
                "them always lies inside, so there the ellipse shows where the group sits and whether groups overlap, not outliers. "
                + ("No labels were given, so no ellipse is drawn: type a label (group) for each file to get one. " if all(g == "(unlabeled)" for g in glabels)
                   else "No labelled group has 3 or more files here, so no ellipse is drawn. " if not drawn
                   else "Files without a label have no ellipse. " if "(unlabeled)" in glabels else "")
                + ("Colours repeat when there are more than 8 labels. " if len(groups) > len(PAL) else "")
                + "Names that would overlap another name in a panel are left out of that panel.",
                [r"$t_i = (t_{i1},\ldots,t_{iK}) = z_i V$  (scores)",
                 r"ellipse of a group: $(t - \bar t)^T S^{-1} (t - \bar t) \leq \chi^2_{2,\,0.90}$",
                 r"$\bar t$, $S$ = mean and covariance of the group's two scores;  $\chi^2_{2,\,0.90} = -2\ln 0.10 \approx 4.61$"], fig)

        # 3 - for every PC: scores of all files as a dot plot (left) next to its loadings as a stick spectrum (right)
        t = "Scores and loadings per PC"
        pos = {i: k_ + 1 for k_, i in enumerate(sorted(range(n), key=lambda i: groups.index(glabels[i])))}   # files grouped by label
        fig = Figure(figsize=(40, 5.0 * nc + 0.8), layout="constrained")
        gs = fig.add_gridspec(nc, 2, width_ratios=[1, 3.2])
        sides, corner = [], []
        centres = [(np.mean([pos[i] for i in range(n) if glabels[i] == g]), g) for g in groups]
        for c in range(nc):
            a, b = fig.add_subplot(gs[c, 0]), fig.add_subplot(gs[c, 1])
            for g in groups:
                ii = [i for i in range(n) if glabels[i] == g]
                xs_ = [pos[i] for i in ii]
                a.axvspan(min(xs_) - 0.5, max(xs_) + 0.5, color=gcol[g], alpha=0.07, lw=0)
                a.scatter(xs_, scores[ii, c], s=48, color=gcol[g], marker=gmark[g], edgecolor="white", lw=0.4, zorder=3)
            a.axhline(0, color="#888", lw=1.0)
            a.set_xticks([x_ for x_, _ in centres], [g for _, g in centres], rotation=30 if len(groups) > 4 else 0,
                         ha="right" if len(groups) > 4 else "center")
            a.yaxis.set_major_locator(MaxNLocator(nbins=5))
            a.set_xlim(0.3, n + 0.7)
            a.tick_params(labelsize=10)
            a.set_xlabel("Label", fontsize=11)
            a.set_ylabel(f"PC{c + 1} Scores ({ev[c]:.1f} %)", fontsize=11)
            stick_panel(b, c)
            if LL is not None:                              # labels on the positive side of this PC top left, negative side bottom left
                tb, z = LL["tbar"][:, c], LL["z"][:, c]
                up = [q for q in np.argsort(-tb) if tb[q] > 0 and abs(z[q]) >= 0.5][:6]
                dn = [q for q in np.argsort(tb) if tb[q] < 0 and abs(z[q]) >= 0.5][:6]
                for j, q in enumerate(up):
                    corner.append(b.text(0.012, 0.97 - 0.075 * j, LL["groups"][q], transform=b.transAxes, ha="left", va="top", fontsize=10,
                                         color=gcol[LL["groups"][q]], fontweight="bold"))
                for j, q in enumerate(dn):
                    corner.append(b.text(0.012, 0.03 + 0.075 * j, LL["groups"][q], transform=b.transAxes, ha="left", va="bottom", fontsize=10,
                                         color=gcol[LL["groups"][q]], fontweight="bold"))
            sides.append(b)
        if multi and len(groups) <= 14:
            fig.legend(handles=[Line2D([], [], marker=gmark[g], ls="", color=gcol[g], label=g) for g in groups],
                       title="label", loc="outside right upper", fontsize=9)
        fig.suptitle(t, fontweight="bold", fontsize=14)
        pk = _Packer(fig)
        pk.taken.extend(tx.get_window_extent(pk.r) for tx in corner)       # keep the m/z labels clear of the label names
        for c, b in enumerate(sides):
            label_sticks(pk, b, c)
        add("pcloadings", t, "One row per PC, all PCs. Left: the PC score of every file (one dot each, files grouped under their label, "
            "shaded bands mark the labels); dots above the zero line are on the positive side of the PC, dots below on the "
            "negative side. Right: the loadings of that PC, one stick per m/z; sticks up are higher in files on the positive side, sticks down in files on the "
            "negative side. "
            + ("The label names in the corners of the right panel say which labels are on that side: labels named at the top sit on the positive side "
               "(rich in the upward m/z), labels named at the bottom on the negative side (rich in the downward m/z); only labels whose mean score is at least "
               "half a standard deviation from zero are named. " if LL is not None else "")
            + f"The {n_lab} largest |loadings| of each PC are labelled with their m/z." + left_out(pk),
            [r"left: points $(i,\ t_{ik})$ for every file $i$ (files ordered by label)",
             r"right: sticks $(m_j,\ p_{kj})$ for every feature $j$,  $\|p_k\| = 1$",
             r"score $t_{ik} = \sum_j z_{ij}\,p_{kj}$, so $p_{kj} > 0$ raises the score of files rich in $j$"], fig, hi=160)

        # 4 - driving peaks across all PCs (dot plot)
        t = "Driving peaks"
        drivers = list(dict.fromkeys(int(j) for c in range(nc) for j in np.argsort(-np.abs(comps[c]))[:3]))
        cmap = matplotlib.colormaps["tab20"]
        xs = np.arange(1, nc + 1)
        nb = max(1, len(drivers))
        bw = 0.8 / nb
        fig = Figure(figsize=(min(40.0, max(18.0, 3.6 * nc + 4)), 10))
        ax = fig.add_subplot()
        for k_, i in enumerate(drivers):
            ax.scatter(xs + (k_ - (nb - 1) / 2) * bw, comps[:, i], s=46, color=cmap(k_ % 20), edgecolor="white", lw=0.5, zorder=3)
        for c in range(nc):                               # a band per PC so the columns of dots are clearly separated
            if c % 2 == 0:
                ax.axvspan(c + 0.5, c + 1.5, color="#8a94a6", alpha=0.10, lw=0)
            ax.axvline(c + 1.5, color="#555", lw=1.0, ls="--")
        ax.set_xlim(0.5, nc + 0.5)
        ax.axhline(0, color="#999", lw=0.8)
        ax.grid(axis="y", alpha=0.3)
        ax.set_xticks(xs, [f"PC{i}\n({ev[i - 1]:.1f}%)" for i in xs], fontsize=11)
        lim = float(np.abs(comps[:, drivers]).max()) or 1.0
        ax.set_ylim(-1.7 * lim, 1.7 * lim)
        ax.set_xlabel("Principal component (explained variance)\nDots: the 3 peaks with the largest |loading| on each PC, drawn on every PC", fontsize=11)
        ax.set_ylabel("Loading (eigenvector weight)", fontsize=12)
        ax.set_title(t, fontweight="bold", fontsize=14)
        fig.tight_layout()
        pk = _Packer(fig)
        for c in range(nc):                               # every dot is labelled with its own m/z (no legend)
            for k_ in np.argsort(-np.abs(comps[c, drivers]))[:n_lab]:
                h = comps[c, drivers[k_]]
                pk.put(ax, (c + 1 + (k_ - (nb - 1) / 2) * bw, h), slab(drivers[k_]), cands=[(0, 5)], fs=7,
                       rot=90, within="fig", down=bool(h < 0))
        add("drivers", t, f"How it is built: for every PC the 3 features with the largest absolute loading are taken "
            f"and pooled into one set of {len(drivers)} driving peaks (each colour = one peak, no legend: every dot "
            "carries the m/z of its peak). For each PC "
            "(x-axis) a column of dots shows the loading of every driving peak on that PC. A dot far from zero at PCk means "
            "the peak is a main driver of PCk; the sign shows the direction (positive = richer in subsamples with "
            "high PCk scores). A peak with dots far from zero on several PCs contributes to all of them. "
            "Loadings are weights, not percentages: squaring a loading gives the fraction of that PC carried by the peak (all squares of a PC add up to 1). "
            "The dashed lines and grey bands separate the PCs. "
            f"The {n_lab} largest dots of every PC are labelled." + left_out(pk),
            [r"$D = \bigcup_{k=1}^{K}\ \mathrm{top}_3\,|p_{kj}|$   (over $j$)",
             r"dot $(k,\ j)$ sits at height $p_{kj}$,  $j \in D$",
             r"$\sum_j p_{kj}^2 = 1$, so $p_{kj}^2$ is the share of PC$k$ carried by peak $j$"], fig)

        if LL is not None:
            G = len(LL["groups"])
            top_n = n_lab or 10
            pcc = [PAL[c % len(PAL)] for c in range(nc)]
            norm_on, sqrt_on = str(res.get("normalize")) == "tic", bool(res.get("sqrt", True))
            unit_txt = ("\u221a(share of total ions)" if norm_on and sqrt_on else "share of total ions" if norm_on
                        else "\u221a(counts)" if sqrt_on else "counts")           # what one unit of d means for these settings

            # 5 - label x PC heat map: which PCs separate each label from the average file
            t = "Label by PC heat map"
            fig = Figure(figsize=(max(8.0, 1.3 * nc + 4.5), max(4.0, 0.7 * G + 2.4)), layout="constrained")
            ax = fig.add_subplot()
            zmax = float(np.abs(LL["z"]).max()) or 1.0
            im = ax.imshow(LL["z"], cmap="RdBu_r", vmin=-zmax, vmax=zmax, aspect="auto")
            ax.set_xticks(range(nc), [f"PC{c + 1}\n{ev[c]:.1f}%" for c in range(nc)], fontsize=10)
            ax.set_yticks(range(G), [f"{g} (n={m})" for g, m in zip(LL["groups"], LL["n"])], fontsize=10)
            ax.set_xlabel("Principal component (explained variance)", fontsize=11)
            ax.set_ylabel("Label", fontsize=11)
            if G * nc <= 150:
                for r in range(G):
                    for c in range(nc):
                        ax.text(c, r, f"{LL['z'][r, c]:+.1f}", ha="center", va="center", fontsize=9,
                                color="white" if abs(LL["z"][r, c]) > 0.6 * zmax else "#222")
            fig.colorbar(im, ax=ax, label="mean score / SD of the PC")
            ax.set_title(t, fontweight="bold", fontsize=12)
            add("labelpc", t, "One row per label, one column per PC. The number is the average score of that label's files on the PC, "
                "in units of the standard deviation of all scores on that PC. It says where the label sits along the PC, not how much of "
                "the PC 'explains' it. Red: the label's files sit on the positive side of the PC; "
                "blue: on the negative side; white: no different from the average file. A strongly coloured cell means this PC "
                "separates the label from the others. How to read it: a red cell means the label is rich in the m/z with positive loadings on that PC "
                "and poor in those with negative loadings; a blue cell is the reverse. Look those m/z up in the loadings of that PC "
                "(scores and loadings figure). "
                "Replicates of one label should agree: if they are spread over both signs of a PC, that PC is not what defines the label.",
                [r"$\bar t_{gk} = \frac{1}{n_g}\sum_{i \in g} t_{ik}$   (mean score of label $g$ on PC $k$)",
                 r"cell $= \bar t_{gk}\ /\ \mathrm{SD}(t_{\cdot k})$"], fig)

            # 6 - per label, the features that define it (dot plot of the magnitude), coloured by the PC that carries them
            t = "Top loadings per label"
            ncol = min(G, 2)
            nrow = -(-G // ncol)
            fig = Figure(figsize=(max(7.5, 0.6 * top_n + 3.0) * ncol + 2.0, 6.0 * nrow + 1.0), layout="constrained")
            axs = fig.subplots(nrow, ncol, squeeze=False)
            used = set()
            for q in range(nrow * ncol):
                ax = axs[q // ncol][q % ncol]
                if q >= G:
                    ax.set_visible(False)
                    continue
                d, parts = LL["d"][q], LL["parts"][q]
                sel = np.argsort(-np.abs(d))[:top_n]                # largest first, left to right
                main = np.argmax(np.abs(parts[:, sel]), axis=0)     # PC that carries most of each feature
                for sgn, mk in ((1, "^"), (-1, "v")):
                    m_ = (d[sel] >= 0) if sgn > 0 else (d[sel] < 0)
                    if m_.any():
                        ax.scatter(np.flatnonzero(m_), np.abs(d[sel])[m_], s=70, marker=mk, c=[pcc[c] for c in main[m_]],
                                   edgecolor="white", lw=0.6, zorder=3)
                used.update(int(c) for c in main)
                for x_, j in enumerate(sel):                        # the plotted value on top of every dot
                    ax.annotate(f"{abs(d[j]):.3g}", (x_, abs(d[j])), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
                ax.set_xticks(range(len(sel)), [slab(j) for j in sel], rotation=90, fontsize=10)
                ax.set_xlim(-0.6, len(sel) - 0.4)
                ax.set_ylim(0, 1.2 * (float(np.abs(d[sel]).max()) or 1.0))
                ax.grid(alpha=0.3)
                top_pc = np.argsort(-LL["share"][q])[:2]
                ax.set_title(f"{LL['groups'][q]} (n={LL['n'][q]}):  " + ", ".join(f"PC{c + 1} {100 * LL['share'][q, c]:.0f}%" for c in top_pc),
                             fontsize=11, fontweight="bold")
                ax.set_xlabel("m/z of the peak", fontsize=10)
                ax.set_ylabel(f"|label minus average file|  ({unit_txt})", fontsize=10)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
                ax.tick_params(axis="y", labelsize=10)
            fig.legend(handles=[Line2D([], [], marker="^", ls="", color="#555", label="richer than average"),
                                Line2D([], [], marker="v", ls="", color="#555", label="poorer than average")]
                       + [Line2D([], [], marker="o", ls="", color=pcc[c], label=f"mostly PC{c + 1}") for c in sorted(used)],
                       title="dot", loc="outside right upper", fontsize=8)
            # a worked example from the data itself: the strongest peak of the first label, term by term
            g0, d0 = LL["groups"][0], LL["d"][0]
            j0 = int(np.argmax(np.abs(d0)))
            tb0, part0 = LL["tbar"][0], LL["parts"][0][:, j0]
            big = [int(k) for k in np.argsort(-np.abs(part0))[:3]]
            rest = float(part0.sum() - part0[big].sum())
            ex_txt = " + ".join(f"PC{k + 1}: ({tb0[k]:+.3f} x {comps[k][j0]:+.3f})" for k in big) \
                + (f" + ({rest:+.4f}) from the other PCs" if nc > 3 else "") + f" = {d0[j0]:+.4f}"
            ex_tex = r"$d = " + " + ".join(rf"({tb0[k]:+.3f}\times{comps[k][j0]:+.3f})" for k in big) \
                + (rf" + ({rest:+.4f})" if nc > 3 else "") + rf" = {d0[j0]:+.4f}$"
            fig.suptitle(t, fontweight="bold", fontsize=13)
            fig.supxlabel(r"dot height $= |d_{gj}|$,   $d_{gj} = \sum_{k}\ \bar t_{gk}\, p_{kj}$   "
                          r"($\bar t_{gk}$ = mean score of label $g$ on PC $k$,  $p_{kj}$ = loading of m/z $j$ on PC $k$,  sum over all PCs)"
                          f"\nunit of d: {unit_txt}   |   example, {g0}, m/z {fmz(mz[j0])}:  {ex_txt}",
                          fontsize=10)
            add("labelload", t, f"One panel per label. Each dot is one of the {top_n} features that most distinguish the label's average spectrum "
                "from the average of all files, computed for each m/z separately; the number above a dot is its height. The height is the size (absolute value) of that difference, the highest dot is the most "
                "characteristic peak. The triangle gives the sign: pointing up = richer in this label, pointing down = poorer. "
                "The colour is the PC that carries most of the feature; a feature that sits on a PC that also separates the label "
                "(see the heat map) is a real marker of it. The title gives the two PCs that carry most of the label's separation "
                "(share of its squared distance from the average file). "
                f"Unit: d is the label's average minus the all-file average of the pre-processed value of that m/z, i.e. in {unit_txt}"
                + (" (the share of the file's ions that fall in that m/z bucket, square-rooted); " if norm_on and sqrt_on else "; ")
                + "it has no other scale, so compare heights within a figure, not across runs with different settings. "
                f"Worked example from your data, label '{g0}', m/z {fmz(mz[j0])}: {ex_txt}. "
                "Because the model keeps the first PCs only, d is the PCA's own reconstruction of that difference; with every PC kept it equals "
                "the plain difference of the two averages.",
                [r"1. counts $c_{ij}$ of file $i$, feature $j$" + (r";  share $x_{ij} = c_{ij}\,/\,\sum_{j'} c_{ij'}$" if norm_on else "")
                 + (r";  $x_{ij} \leftarrow \sqrt{x_{ij}}$" if sqrt_on else ""),
                 r"2. mean-centre: $z_{ij} = x_{ij} - \bar x_j$,   $\bar x_j = \frac{1}{n}\sum_i x_{ij}$",
                 r"3. PCA of $Z$: loadings $p_{kj}$ (unit length), scores $t_{ik} = \sum_j z_{ij}\,p_{kj}$",
                 r"4. mean score of label $g$ ($n_g$ files): $\bar t_{gk} = \frac{1}{n_g}\sum_{i \in g} t_{ik}$",
                 r"5. $d_{gj} = \sum_{k=1}^{K} \bar t_{gk}\, p_{kj}\ \approx\ \frac{1}{n_g}\sum_{i \in g} x_{ij} - \bar x_j$   (label average minus all-file average)",
                 r"6. dot height $= |d_{gj}|$, triangle = sign of $d_{gj}$, colour = PC $k$ with the largest $|\bar t_{gk}\, p_{kj}|$",
                 r"7. example: " + ex_tex,
                 r"share of PC $k$ $= \bar t_{gk}^2\ /\ \sum_l \bar t_{gl}^2$"], fig)
        # 7 - nearest centroid: which reference label is every file closest to (leave-one-out)
        NC = res.get("centroid")
        if NC and nc >= 2 and len(NC.get("pred", [])) == n:
            t = "Nearest centroid"
            refs, pl = NC["refs"], NC["plane"]
            Dm = np.array([[np.inf if v is None else v for v in r] for r in NC["dist"]], float)
            cen = np.asarray(NC["centroids"], float)
            order = sorted(range(n), key=lambda i: (glabels[i] not in refs, glabels[i], i))
            hit = lambda i: NC["pred"][i] == glabels[i]
            ev_ = [i for i in range(n) if NC["evaluated"][i]]
            unk = [i for i in range(n) if not NC["is_ref"][i]]
            fig = Figure(figsize=(21, max(6.0, 0.28 * n + 2.5)), layout="constrained")
            ax1, ax2, ax3 = fig.subplots(1, 3, gridspec_kw={"width_ratios": [1.15, 1.0, 0.7]})
            for g in groups:                                  # files: reference files filled, unknown files hollow
                ii = [i for i in range(n) if glabels[i] == g]
                solid = g in refs
                ax1.scatter(scores[ii, 0], scores[ii, 1], s=70, marker=gmark[g], zorder=3,
                            c=gcol[g] if solid else "none", edgecolor=gcol[g], lw=1.4, label=f"{g} (n={len(ii)})" + ("" if solid else ", unknown"))
            for q, g in enumerate(refs):
                ax1.scatter(cen[q, 0], cen[q, 1], s=420, marker="*", c=gcol[g], edgecolor="black", lw=1.2, zorder=5)
                ax1.annotate(f"centroid {g}", (cen[q, 0], cen[q, 1]), textcoords="offset points", xytext=(8, 8), fontsize=10, fontweight="bold")
            for i in range(n):                                # a line to the assigned centroid: red when a reference file is wrong
                q = refs.index(NC["pred"][i])
                bad = NC["evaluated"][i] and not hit(i)
                if bad or not NC["is_ref"][i]:
                    ax1.plot([scores[i, 0], cen[q, 0]], [scores[i, 1], cen[q, 1]], ls="--", lw=1.0, zorder=2,
                             color="#c53030" if bad else "#888")
            ax1.axhline(0, color="#999", lw=0.6)
            ax1.axvline(0, color="#999", lw=0.6)
            ax1.set_xlabel(f"PC1 ({ev[0]:.1f}%)", fontsize=11)
            ax1.set_ylabel(f"PC2 ({ev[1]:.1f}%)", fontsize=11)
            ax1.grid(alpha=0.3)
            ax1.legend(fontsize=8, loc="best")
            ax1.set_title("Scores, centroids (stars) and assignments", fontsize=11, fontweight="bold")

            Dp = Dm[order]
            fin = Dp[np.isfinite(Dp)]
            im = ax2.imshow(np.where(np.isfinite(Dp), Dp, np.nan), cmap="viridis_r", aspect="auto",
                            vmin=0, vmax=float(fin.max()) if fin.size else 1.0)
            ax2.set_xticks(range(len(refs)), [f"{g}" for g in refs], fontsize=10, rotation=20, ha="right")
            ax2.set_yticks(range(n), [("\u2714 " if NC["evaluated"][i] and hit(i) else "\u2718 " if NC["evaluated"][i] else "? ")
                                      + f"{names[i]} [{glabels[i]}]" for i in order], fontsize=8)
            for r_, i in enumerate(order):
                ax2.add_patch(Rectangle((refs.index(NC["pred"][i]) - 0.5, r_ - 0.5), 1, 1, fill=False,
                                                                        ec="#c53030" if NC["evaluated"][i] and not hit(i) else "white", lw=2.2))
                if len(refs) <= 6:
                    for q in range(len(refs)):
                        if np.isfinite(Dp[r_, q]):
                            ax2.text(q, r_, f"{Dp[r_, q]:.2g}", ha="center", va="center", fontsize=7, color="white" if Dp[r_, q] > 0.5 * fin.max() else "#222")
            fig.colorbar(im, ax=ax2, label=f"distance to the centroid in PC1\u2013PC{pl} (smaller = closer)", shrink=0.8)
            ax2.set_title("Distance of every file to each reference centroid", fontsize=11, fontweight="bold")

            ks = np.arange(1, nc + 1)
            accv = [np.nan if a is None else 100 * a for a in NC["acc"]]
            ax3.plot(ks, accv, color=PAL[0], marker="o", ms=8, lw=1.2)
            ax3.axvline(pl, color="#c53030", ls=":", lw=1.2)
            for k_, a_ in zip(ks, accv):
                ax3.annotate(f"{a_:.0f}%", (k_, a_), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
            ax3.set_xticks(ks[::1 if nc <= 12 else 2], [f"1\u2013{k}" if k > 1 else "1" for k in ks[::1 if nc <= 12 else 2]], fontsize=9)
            ax3.set_ylim(0, 112)
            ax3.set_xlabel("PCs used for the distance", fontsize=11)
            ax3.set_ylabel("Leave-one-out accuracy of the reference files (%)", fontsize=11)
            ax3.grid(alpha=0.3)
            ax3.set_title("Accuracy vs number of PCs", fontsize=11, fontweight="bold")
            nok = sum(hit(i) for i in ev_)
            fig.suptitle(f"{t}: {nok}/{len(ev_)} reference files assigned to their own label in PC1\u2013PC{pl}"
                         + (f";  mean margin {NC['sep']:+.2f}" if NC.get("sep") is not None else "")
                         + (f";  {len(unk)} unknown file(s) assigned" if unk else ""), fontweight="bold", fontsize=13)
            wrong = [names[i] for i in ev_ if not hit(i)]
            add("centroid", t, f"Reference labels: {', '.join(refs)}. The centroid (star) of a reference label is the average score of its files. "
                "Every file is assigned to the nearest centroid by Euclidean distance in the first PCs; a reference file is first removed "
                "from its own centroid (leave-one-out), so it cannot vote for itself. Left: the scores with the centroids; a dashed line joins a "
                "file to the centroid it was assigned to (red = a reference file that landed on the wrong label; grey = an unknown file, drawn hollow, "
                "that was not used to build any centroid). Middle: the distance of each file to each centroid; the white box marks the assigned "
                "centroid (red when wrong), \u2714 / \u2718 = right / wrong, ? = unknown file. Right: the same leave-one-out accuracy when 1, 1\u20132, "
                "1\u20133, \u2026 PCs are used; the red dotted line is the plane shown on the left. "
                "The margin is the mean of (d_other \u2212 d_own) / max(d_other, d_own): +1 = far from the other labels, 0 = on the border, "
                "negative = closer to another label. "
                + (f"Wrong: {', '.join(wrong)}. " if wrong else "No reference file was wrong. ")
                + "An unknown file is always assigned to some reference, however far it is: read the distance, not only the label.",
                [r"$c_g = \frac{1}{n_g}\sum_{i \in g} t_i$   (centroid of reference label $g$, scores $t_i$ in the first $k$ PCs)",
                 r"$d_{ig} = \| t_i - c_{g \setminus i} \|$   ($i$ is left out of its own label)",
                 r"assigned label $= \arg\min_g d_{ig}$",
                 r"margin $= (d_{\mathrm{other}} - d_{\mathrm{own}})\ /\ \max(d_{\mathrm{other}}, d_{\mathrm{own}})$"], fig)
    return out


def tune_figure(info):
    """Figure of the learning step (see :mod:`tof_sims_driver.tune`); returns a PNG data URI.

    Top left: distance of the A and B centroids on every PC (in SDs of that PC; greyed = too little variance to compare).
    Top right: loading spectrum of the chosen PC. Below: pair plot of the first PCs and the chosen one, centroids as stars,
    a dashed line between them; red frames = the chosen PC."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    A, B = info["refs"]
    S, evr, dz, pc = info["scores"], info["evr"], info["dz"], info["pc"]
    K = S.shape[1]
    lab = np.array(info["labels"])
    col = {A: PAL[0], B: PAL[1]}
    mk = {A: "o", B: "s"}
    shown = sorted(set(range(min(K, 4))) | {pc})
    nP = len(shown)
    best = info["best"]
    elig = evr >= info["min_var"]
    with _FIG_LOCK:
        cell = 2.7
        nG = nP - 1                                       # pair plot: PC pairs only (lower triangle)
        fig = Figure(figsize=(max(12.0, cell * nG), 4.4 + cell * nG), layout="constrained")
        gs = fig.add_gridspec(nG + 1, max(2, nG), height_ratios=[1.7] + [1] * nG)
        half = max(1, max(2, nG) // 2)
        ax = fig.add_subplot(gs[0, :half])
        xs = np.arange(1, K + 1)
        ax.bar(xs, dz, color=[("#c53030" if k == pc else PAL[0]) if elig[k] else "#cbd5e0" for k in range(K)])
        for k in range(K):
            ax.annotate(f"{dz[k]:.2f}\n{100 * evr[k]:.0f}%", (xs[k], dz[k]), textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=8)
        ax.set_xticks(xs, [f"PC{k}" for k in xs], fontsize=9)
        ax.set_ylabel("centroid distance (SD of the PC)", fontsize=10)
        ax.set_ylim(0, 1.25 * float(dz.max()) if dz.max() > 0 else 1)
        ax.set_title(f"Centroid distance per PC (label: distance, % variance;\ngrey = under {100 * info['min_var']:.0f}% variance, not compared)",
                     fontsize=10, fontweight="bold")
        ax.grid(alpha=0.3, axis="y")
        ax2 = fig.add_subplot(gs[0, half:])
        mz, ld = np.asarray(info["mz"]), np.asarray(info["loading"])
        ax2.vlines(mz, 0, ld, color=PAL[2], lw=1.0)
        ax2.axhline(0, color="#555", lw=0.6)
        for j in np.argsort(-np.abs(ld))[:8]:
            ax2.annotate(f"{mz[j]:g}", (mz[j], ld[j]), textcoords="offset points", xytext=(0, 4 if ld[j] >= 0 else -11),
                         ha="center", fontsize=8)
        ax2.set_xlabel("m/z", fontsize=10)
        ax2.set_ylabel("loading", fontsize=10)
        ax2.set_title(f"Loadings of PC{pc + 1} ({100 * evr[pc]:.1f}% of the variance):\nthe m/z to analyse", fontsize=10, fontweight="bold")
        ax2.grid(alpha=0.3)
        for r_ in range(nG):
            for c_ in range(max(2, nG)):
                a = fig.add_subplot(gs[r_ + 1, c_])
                if c_ > r_:
                    a.set_visible(False)
                    continue
                i, j = shown[r_ + 1], shown[c_]
                cen = {}
                for g in (A, B):
                    m = lab == g
                    a.scatter(S[m, j], S[m, i], s=40, c=col[g], marker=mk[g], edgecolor="white", lw=0.5, zorder=3,
                              label=f"{g} (n={int(m.sum())})")
                    cen[g] = (S[m, j].mean(), S[m, i].mean())
                a.plot([cen[A][0], cen[B][0]], [cen[A][1], cen[B][1]], ls="--", lw=1.0, color="#333", zorder=2)
                for g in (A, B):
                    a.scatter(*cen[g], s=230, marker="*", c=col[g], edgecolor="black", lw=1.0, zorder=5)
                a.grid(alpha=0.3)
                a.tick_params(labelsize=7)
                a.set_xlabel(f"PC{j + 1} ({100 * evr[j]:.1f}%)", fontsize=9)
                a.set_ylabel(f"PC{i + 1} ({100 * evr[i]:.1f}%)", fontsize=9)
                if r_ == 0 and c_ == 0:
                    a.legend(fontsize=7, loc="best")
                if pc in (i, j):
                    for sp in a.spines.values():
                        sp.set_edgecolor("#c53030")
                        sp.set_linewidth(2.0)
        what = ("closest" if info["goal"] == "closest" else "most separating")
        fig.suptitle(f"{info['title']}: PC{pc + 1} is the {what} PC (distance {dz[pc]:.2f} SD)\n"
                     f"bin {best['bin']:g}, NBR \u2265 {best['ratio_min']:g}, TIC \u2265 {100 * best['frac_min']:g}%, {best['n_features']} bins",
                     fontsize=11, fontweight="bold")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=_dpi_for(fig, hi=100), bbox_inches="tight")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

