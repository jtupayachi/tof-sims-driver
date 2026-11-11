"""PCA of TOF-SIMS subsample spectra (Yang et al., Biointerphases 14, 051004, 2019).

Two feature modes:

* ``all``  - every m/z bin of the whole spectrum is a feature (All-peak PCA);
* ``mine`` - only the peak windows picked by the user are features (Selected-peak PCA).

Preprocessing, in order: total-ion normalisation -> square root -> drop constant features ->
mean centring (no unit-variance scaling) -> PCA (SVD).
Every step is reported with its formula and the numbers of the run (``res["steps"]``).
"""
import numpy as np

from .store import MASS_MAX

RATIO_MIN = 3.0          # a peak must be >= 3x its strongest neighbouring feature ...
FRAC_MIN = 0.005         # ... and carry >= 0.5% of the total ion intensity to "pass" (defaults; per-run override)
N_LABELS = 12            # labelled features per PC in the figures (per-run override)
TOP_PRESELECT = 40       # most influential features pre-ticked in the All-peak PCA peak list


def _tohost(a):
    for attr in ("get", "to_numpy"):                 # cupy / cudf -> numpy
        f = getattr(a, attr, None)
        if callable(f):
            try:
                return np.asarray(f())
            except Exception:
                pass
    return np.asarray(a)


def _num(v):
    """float for JSON; None stands for infinity (neighbours have no signal)."""
    v = float(v)
    return v if np.isfinite(v) else None


def run_pca(X, ncomp):
    """PCA on the host matrix X. Prefer cuML (GPU), fall back to scikit-learn, then numpy SVD.
    Returns (scores, explained_variance_ratio, components, backend_name)."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    try:
        from cuml import PCA as cuPCA
        m = cuPCA(n_components=ncomp)
        scores = _tohost(m.fit_transform(X))
        return scores, _tohost(m.explained_variance_ratio_), _tohost(m.components_), "cuML (GPU)"
    except Exception as e:
        cuml_err = f"{type(e).__name__}: {e}"
    try:
        from sklearn.decomposition import PCA as skPCA
        m = skPCA(n_components=ncomp)
        scores = m.fit_transform(X)
        return scores, m.explained_variance_ratio_, m.components_, f"scikit-learn (CPU; cuML: {cuml_err})"
    except Exception:
        Xc = X - X.mean(0)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        var = S ** 2 / max(1, len(X) - 1)
        return U[:, :ncomp] * S[:ncomp], (var / var.sum())[:ncomp], Vt[:ncomp], "numpy SVD"


# --------------------------------------------------------------------------- steps & formulas

def pca_steps(mode, st=None):
    """Ordered preprocessing / PCA steps with an HTML formula each. `st` holds the numbers of a run
    (see compute_pca); without it only the generic description is returned (for the UI template)."""
    st = st or {}
    g = st.get
    mine = mode == "mine"
    rmin, fmin = g("ratio_min", RATIO_MIN), g("frac_min", FRAC_MIN)
    steps = []

    def add(title, formula, detail=""):
        steps.append(dict(title=title, formula=formula, detail=detail))

    if mine:
        add("Integrate each selected peak window",
            "x<sub>ip</sub> = #{ ions of subsample i : lo<sub>p</sub> &le; m &lt; hi<sub>p</sub> },&nbsp; "
            "m = ((t &minus; k<sub>0</sub>) / s<sub>f</sub>)<sup>2</sup>",
            (f"n = {g('n')} subsamples &times; p = {g('p')} peaks (counts read straight from the "
             "time-of-flight channels)") if st else "one integrated raw count per peak and subsample")
    else:
        add("Bin the whole spectrum (every bin is a feature)",
            "x<sub>ij</sub> = #{ ions of subsample i : j&middot;w &le; m &lt; (j+1)&middot;w },&nbsp; "
            "m = ((t &minus; k<sub>0</sub>) / s<sub>f</sub>)<sup>2</sup>",
            (f"n = {g('n')} subsamples &times; p = {g('p')} bins of w = {g('bin')} u up to "
             f"m/z {g('mass_max'):.0f}") if st else "no peak identification or selection")
    add("Peak-quality metrics (raw counts, before normalisation)",
        "I<sub>j</sub> = &Sigma;<sub>i</sub> x<sub>ij</sub>;&nbsp; "
        "f<sub>j</sub> = I<sub>j</sub> / &Sigma;<sub>i</sub> N<sub>i</sub> (share of all ions);&nbsp; "
        "r<sub>j</sub> = I<sub>j</sub> / max(I<sub>j&minus;1</sub>, I<sub>j+1</sub>) (neighbours in m/z); "
        f"pass &hArr; r<sub>j</sub> &ge; {rmin:g} (NBR) and f<sub>j</sub> &ge; {100 * fmin:g}% (TIC)",
        ((f"{g('n_pass')} of {g('p')} features pass; the others are removed from the PCA"
          if g("filter_peaks") else f"{g('n_pass')} of {g('p')} features pass (filter off: all kept)")
         if st else "optionally removes bins that are not real peaks (All-peak PCA)"))
    if g("normalize", "tic") == "tic":
        add("Total-ion normalisation per subsample",
            "x&#771;<sub>ij</sub> = x<sub>ij</sub> / &Sigma;<sub>k</sub> x<sub>ik</sub>",
            ("each row is divided by the file's total over every bin, before the filter" if g("filter_peaks")
             else "each row sums to 1 (sum over the " + ("selected peaks" if mine else "spectrum bins") + ")"))
    else:
        add("Total-ion normalisation", "x&#771;<sub>ij</sub> = x<sub>ij</sub>", "skipped (normalize = none)")
    if g("sqrt", True):
        add("Square-root transform (Poisson variance stabilisation)",
            "y<sub>ij</sub> = &radic;x&#771;<sub>ij</sub>", "")
    else:
        add("Square-root transform", "y<sub>ij</sub> = x&#771;<sub>ij</sub>", "skipped")
    add("Drop empty / constant features",
        "keep j &hArr; s<sub>j</sub> = sd<sub>i</sub>(y<sub>ij</sub>) &gt; 0",
        f"{g('n_dropped')} dropped, p&prime; = {g('p_kept')} kept" if st else "")
    add("Mean centring (no unit-variance scaling)",
        "z<sub>ij</sub> = y<sub>ij</sub> &minus; y&#772;<sub>j</sub>,&nbsp; "
        "y&#772;<sub>j</sub> = (1/n) &Sigma;<sub>i</sub> y<sub>ij</sub>",
        (f"applied: max |column mean| after centring = {g('center_check'):.1e}" if st else "applied")
        + " &mdash; every peak keeps its own variance (Yang et al. 2019)")
    add("PCA by singular value decomposition",
        "Z = U S V<sup>T</sup>;&nbsp; scores T = U S = Z V;&nbsp; loading p<sub>k</sub> = k-th row of "
        "V<sup>T</sup> (unit eigenvector of C = Z<sup>T</sup>Z / (n&minus;1));&nbsp; "
        "&lambda;<sub>k</sub> = s<sub>k</sub><sup>2</sup> / (n&minus;1);&nbsp; "
        "EV<sub>k</sub> = &lambda;<sub>k</sub> / &Sigma;<sub>l</sub> &lambda;<sub>l</sub>",
        (f"K = {g('K')} components (&le; min(n&minus;1, p&prime;)); backend: {g('backend')}") if st else "")
    return steps


def selection_steps(st=None):
    """How the peaks of the Selected-peak PCA are chosen (in the All-peak PCA tab)."""
    st = st or {}
    rmin, fmin = st.get("ratio_min", RATIO_MIN), st.get("frac_min", FRAC_MIN)
    out = [
        dict(title="Run the All-peak PCA",
             formula="features = every m/z bin of width w (steps 1&ndash;8 of the All-peak PCA)",
             detail="pick subsamples, labels, bin width; click Run PCA"),
        dict(title="Rank every bin by its influence on the PCs",
             formula="m<sub>j</sub> = max<sub>k</sub> |p<sub>kj</sub>|",
             detail=f"the {TOP_PRESELECT} most influential bins are pre-ticked"),
        dict(title="Quality filter (button \u201cselect passing\u201d)",
             formula=f"keep j &hArr; r<sub>j</sub> &ge; {rmin:g} and f<sub>j</sub> &ge; {100 * fmin:g}%",
             detail="r = intensity / strongest neighbouring bin, f = share of all ions"),
        dict(title="Manual curation",
             formula="tick / untick peaks in the peak list (or \u201cpeaks\u2026\u201d) and subsamples in "
                     "\u201cfiles\u2026\u201d",
             detail="the selection is passed live to the Selected-peak PCA tab"),
        dict(title="Turn each selected bin into a peak window",
             formula="[lo<sub>p</sub>, hi<sub>p</sub>) = [m<sub>p</sub> &minus; w/2, m<sub>p</sub> + w/2)",
             detail=""),
        dict(title="Selected-peak PCA",
             formula="integrate the windows, then the same preprocessing and PCA",
             detail=(f"{st['p']} peaks &times; {st['n']} subsamples" if st else "")),
    ]
    return out


# --------------------------------------------------------------------------- nearest centroid

def nearest_centroid(scores, labels, refs=None, plane=2):
    """Leave-one-out nearest-centroid classification in PC space.

    The centroid of a reference label is the mean score of its files. Each file is assigned to the nearest
    centroid; a reference file is left out of its own centroid first, so it cannot vote for itself. Files whose
    label is not a reference are 'unknown': they are only assigned, never used for a centroid.
    `refs` = the reference labels (default: every non-empty label). Returns None unless 2+ references have files.
    Distances are Euclidean in the first `plane` PCs (the plane that is plotted); `acc[k-1]` is the accuracy of
    the reference files when the first k PCs are used, `sep` the mean silhouette-like margin
    (d_other - d_own) / max(d_other, d_own) in the plane."""
    S = np.asarray(scores, float)
    n, K = S.shape
    lab = [str(x or "").strip() for x in labels] + [""] * (n - len(labels))
    refs = [g for g in dict.fromkeys(refs if refs else lab) if g and g in lab]
    if len(refs) < 2:
        return None
    members = [np.flatnonzero([x == g for x in lab[:n]]) for g in refs]
    plane = max(1, min(int(plane), K))

    def dist(k):
        D = np.full((n, len(refs)), np.inf)
        for q, m in enumerate(members):
            for i in range(n):
                mm = m[m != i]
                if len(mm):
                    D[i, q] = np.linalg.norm(S[i, :k] - S[mm, :k].mean(0))
        return D

    is_ref = np.array([x in refs for x in lab[:n]])
    own = np.array([refs.index(x) if x in refs else -1 for x in lab[:n]])
    ok = is_ref & np.isfinite(dist(1)[np.arange(n), np.maximum(own, 0)])         # a lone file has no centroid of its own
    acc = []
    for k in range(1, K + 1):
        D = dist(k)
        acc.append(float(np.mean(np.argmin(D, 1)[ok] == own[ok])) if ok.any() else None)
    D = dist(plane)
    pred = np.argmin(D, 1)
    marg = []
    for i in np.flatnonzero(ok):
        other = np.delete(D[i], own[i])
        d_o, d_n = D[i, own[i]], float(other.min())
        marg.append((d_n - d_o) / max(d_n, d_o, 1e-12))
    return dict(refs=refs, plane=plane, is_ref=[bool(v) for v in is_ref], evaluated=[bool(v) for v in ok],
                pred=[refs[j] for j in pred], dist=[[_num(v) for v in r] for r in D],
                centroids=[S[m].mean(0).tolist() for m in members], acc=acc,
                sep=float(np.mean(marg)) if marg else None)


# --------------------------------------------------------------------------- PCA

def compute_pca(store, names, labels, ncomp, normalize, sqrt=True, mode="all",
                bin_width=1.0, mass_max=MASS_MAX, peaks=None, figures=True,
                ratio_min=RATIO_MIN, frac_min=FRAC_MIN, n_labels=N_LABELS, filter_peaks=False, ref_labels=None):
    """PCA over subsample spectra. mode='all' uses the whole binned spectrum (every m/z bin) as
    features; mode='mine' uses only the supplied peaks (list of {name, lo, hi}).

    With ``filter_peaks`` (All-peak PCA only) the bins failing the quality filter - intensity < `ratio_min` x
    the strongest neighbouring bin (NBR) or < `frac_min` of the total ion count (TIC) - are removed before the
    PCA. Normalisation still divides by each file's full total, so filtered and unfiltered runs are comparable.

    `store` needs ``coarse_vector(name, width, mass_max)``, ``peak_counts(name, windows)`` and
    ``total_ions(name)`` (see :class:`tof_sims_driver.store.Store`)."""
    mode = "mine" if str(mode).lower().startswith("mine") else "all"
    if mode == "mine":
        feats = [p for p in (peaks or []) if "lo" in p and "hi" in p]
        if not feats:
            raise ValueError("no peaks selected - pick peaks in the All-peak PCA first")
        windows = [(float(f["lo"]), float(f["hi"])) for f in feats]
        X = np.vstack([store.peak_counts(n, windows) for n in names]).astype(np.float64)
        masses = np.array([(lo + hi) / 2 for lo, hi in windows], dtype=np.float64)
        lo_arr = np.array([w[0] for w in windows], dtype=np.float64)
        hi_arr = np.array([w[1] for w in windows], dtype=np.float64)
        feat_names = [str(f.get("name", "") or "") for f in feats]
    else:
        X = np.vstack([store.coarse_vector(n, bin_width, mass_max) for n in names]).astype(np.float64)
        masses = (np.arange(X.shape[1]) + 0.5) * bin_width
        lo_arr, hi_arr = masses - bin_width / 2, masses + bin_width / 2
        feat_names = ["" for _ in range(X.shape[1])]
    n, p = X.shape
    # peak-quality metrics on raw counts pooled over samples (before any normalisation)
    inten = X.sum(0)
    total_ions = float(sum(store.total_ions(nm) for nm in names)) or 1.0
    frac = inten / total_ions                         # share of the total ion intensity
    order = np.argsort(masses)
    so = inten[order]
    nb_s = np.zeros_like(so)                          # strongest adjacent feature in m/z order
    if len(so) > 1:
        nb_s[1:] = so[:-1]
        nb_s[:-1] = np.maximum(nb_s[:-1], so[1:])
    nb = np.zeros_like(inten)
    nb[order] = nb_s
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(nb > 0, inten / nb, np.where(inten > 0, np.inf, 0.0))
    passing = (ratio >= ratio_min) & (frac >= frac_min)
    n_pass = int(passing.sum())
    s = X.sum(1, keepdims=True)                      # per-file total over every bin, taken before any filtering
    s[s == 0] = 1.0
    filtered = bool(filter_peaks) and mode == "all"
    if filtered:
        if n_pass < 3:
            raise ValueError(
                f"only {n_pass} of {p} bins pass the filter (NBR >= {ratio_min:g}, TIC >= {100 * frac_min:g}%) - "
                "lower the limits, use a finer bin width, or switch the filter off")
        X, masses, lo_arr, hi_arr = X[:, passing], masses[passing], lo_arr[passing], hi_arr[passing]
        feat_names = [feat_names[i] for i in np.flatnonzero(passing)]
        ratio, frac, inten = ratio[passing], frac[passing], inten[passing]
    n_filtered = p - X.shape[1]
    raw = X.copy()                                   # raw ion counts, kept for the export
    if normalize == "tic":                           # total-ion normalisation per sample
        X = X / s
    if sqrt:                                          # Yang et al.: square-root transform
        X = np.sqrt(np.clip(X, 0.0, None))
    keep = X.std(0) > 0                              # drop empty / constant features
    Xk = X[:, keep]
    raw = raw[:, keep]
    if Xk.shape[1] == 0:
        raise ValueError("no signal in the selected samples")
    sel = np.flatnonzero(keep)
    masses, lo_arr, hi_arr = masses[keep], lo_arr[keep], hi_arr[keep]
    feat_names = [feat_names[i] for i in sel]
    ratio, frac, inten = ratio[keep], frac[keep], inten[keep]
    Xk = Xk - Xk.mean(0)                             # mean centering (Yang et al.)
    center_check = float(np.abs(Xk.mean(0)).max())
    ncomp = max(1, min(int(ncomp), Xk.shape[0] - 1, Xk.shape[1]))
    scores, evr, comps, backend = run_pca(Xk, ncomp)
    scores = np.asarray(scores, float)[:, :ncomp]
    evr = np.asarray(evr, float)[:ncomp]
    comps = np.asarray(comps, float)[:ncomp]
    loadings = []
    for c in range(comps.shape[0]):
        idx = np.argsort(-np.abs(comps[c]))[:10]
        loadings.append([{"mz": float(masses[i]), "w": float(comps[c][i]),
                          "name": feat_names[i], "ratio": _num(ratio[i]), "frac": float(frac[i])}
                         for i in idx])
    stats = dict(n=n, p=p,  bin=bin_width, mass_max=mass_max, n_pass=n_pass, normalize=normalize,
                 sqrt=bool(sqrt), n_dropped=int(X.shape[1] - Xk.shape[1]), n_filtered=int(n_filtered),
                 filter_peaks=filtered, p_kept=int(Xk.shape[1]),
                 center_check=center_check, K=int(ncomp), backend=backend.split(" (")[0],
                 ratio_min=ratio_min, frac_min=frac_min)
    centroid = nearest_centroid(scores, labels, ref_labels) if scores.shape[1] >= 2 else None
    res = dict(backend=backend, names=list(names), labels=list(labels), mode=mode, bin=bin_width,
               normalize=str(normalize), sqrt=bool(sqrt),
               scores=scores.tolist(), explained=evr.tolist(),
               n_features=int(Xk.shape[1]), n_components=int(ncomp), loadings=loadings,
               peak_mz=[float(m) for m in masses], peak_names=list(feat_names),
               peak_lo=[float(v) for v in lo_arr], peak_hi=[float(v) for v in hi_arr],
               peak_counts=[float(v) for v in inten],
               counts=np.rint(raw).astype(np.int64).tolist(),        # files x features, raw ion counts before any normalisation
               total_ions=[int(store.total_ions(nm)) for nm in names],
               norm_totals=[float(v) for v in s[:, 0]] if filtered else None,
               peak_ratio=[_num(r) for r in ratio], peak_frac=[float(f) for f in frac],
               comps=comps.tolist(), mean_centered=True, center_check=center_check,
               ratio_min=float(ratio_min), frac_min=float(frac_min), filter_peaks=filtered,
               n_filtered=int(n_filtered), n_pass=n_pass, n_bins=int(p),
               n_labels=int(n_labels), centroid=centroid,
               steps=pca_steps(mode, stats),
               selection_steps=selection_steps(stats) if mode == "mine" else [])
    if figures:
        from .figures import make_figures
        res["figures"] = make_figures(res)
    return res
