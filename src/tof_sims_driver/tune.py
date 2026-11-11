"""Learn the All-peak PCA settings (bin width, NBR limit, TIC limit) from two labels, and find the PC that links them.

Compound A (a standard, e.g. 6PPD) and compound B (a sample, e.g. FI-sample1) are two labels of the file list.
Only their files are used. For every bin width / NBR / TIC combination the bins are filtered and a PCA is run on
those files (the PCA never sees the labels). The result is the setting and the PC on which the centroids of A and B
are nearest, in units of the spread of that PC (``d = |mean_A - mean_B| / SD of all scores on the PC``). Only PCs that
carry at least ``min_var`` of the variance count, because a noise PC makes anything look close. The best value is the
minimum over settings and PCs, so it is compared with the same minimum for randomly shuffled A/B labels (``p_corr``):
a small ``d`` with a large ``p_corr`` is what chance gives.

The PCs belong to the A+B PCA, not to the later run on all files: use the loadings returned here to read them.
"""
import numpy as np

from .store import MASS_MAX

BINS = (0.25, 0.5, 1.0, 2.0)
RATIOS = (0.0, 1.5, 2.0, 3.0, 5.0)
FRACS = (0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01)      # fractions of all ions (0.01 = 1 %)
MIN_FEATURES = 10                                       # a PCA of fewer bins says nothing about the spectrum
MIN_VAR = 0.03                                          # a PC must explain this share to be compared
N_PC = 10
N_PERM = 500


def _fit(X, norm_tot, normalize, sqrt, k=N_PC):
    """Scores, explained-variance ratios, loadings and kept column indices of the PCA of X (files x bins)."""
    if normalize == "tic":
        X = X / norm_tot
    if sqrt:
        X = np.sqrt(np.clip(X, 0.0, None))
    cols = np.flatnonzero(X.std(0) > 0)
    X = X[:, cols]
    if len(cols) < 2:
        return None
    X = X - X.mean(0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    k = max(2, min(int(k), len(S)))
    var = S ** 2
    return U[:, :k] * S[:k], var[:k] / var.sum(), Vt[:k], cols


def _dz(S, isA):
    """Per PC: centroid distance of A and B, raw and divided by the SD of all scores on that PC."""
    raw = np.abs(S[isA].mean(0) - S[~isA].mean(0))
    return raw, raw / np.maximum(S.std(0, ddof=1), 1e-300)


def _prep(store, rn, w, mass_max, tot):
    """Binned spectra of the A+B files and the quality metrics of every bin (same definitions as compute_pca)."""
    X = np.vstack([store.coarse_vector(n, w, mass_max) for n in rn]).astype(np.float64)
    s = X.sum(1, keepdims=True)
    s[s == 0] = 1.0
    inten = X.sum(0)
    nb = np.zeros_like(inten)
    nb[1:] = inten[:-1]
    nb[:-1] = np.maximum(nb[:-1], inten[1:])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(nb > 0, inten / nb, np.where(inten > 0, np.inf, 0.0))
    return X, s, inten, ratio, inten / tot


def tune_filter(store, names, labels, refs, normalize="tic", sqrt=True, mass_max=MASS_MAX,
                bins=BINS, ratios=RATIOS, fracs=FRACS, min_features=MIN_FEATURES, min_var=MIN_VAR, top=10,
                n_perm=N_PERM):
    """Grid search over (bin, NBR, TIC) on the files of labels A (``refs[0]``) and B (``refs[1]``).
    Returns ``{results, best, baseline, pcs, loadings, figure, ...}``; settings that keep exactly the same bins are
    listed once, with the loosest limits."""
    refs = [str(r).strip() for r in refs if str(r).strip()]
    if len(set(refs)) != 2:
        raise ValueError("pick two different labels")
    lab = [str(x or "").strip() for x in labels]
    idx = [i for i, x in enumerate(lab) if x in refs]
    for r in refs:
        if sum(lab[i] == r for i in idx) < 2:
            raise ValueError(f"label '{r}' needs at least 2 subsamples")
    rn, rl = [names[i] for i in idx], [lab[i] for i in idx]
    isA = np.array([x == refs[0] for x in rl])
    tot = float(sum(store.total_ions(n) for n in rn)) or 1.0
    results, fits = [], {}
    for w in bins:
        X, s, inten, ratio, frac = _prep(store, rn, w, mass_max, tot)
        seen = set()
        for rmin in sorted(ratios):
            for fmin in sorted(fracs):
                keep = (ratio >= rmin) & (frac >= fmin) & (inten > 0)
                key = keep.tobytes()
                if key in seen or keep.sum() < min_features:
                    continue
                seen.add(key)
                fit = _fit(X[:, keep], s, normalize, sqrt)
                if fit is None:
                    continue
                S, evr, _, _ = fit
                row = dict(bin=float(w), ratio_min=float(rmin), frac_min=float(fmin), filter=bool(rmin > 0 or fmin > 0),
                           n_features=int(keep.sum()))
                raw, dz = _dz(S, isA)
                ok = evr >= min_var
                if not ok.any():
                    continue
                k = int(np.flatnonzero(ok)[np.argmin(dz[ok])])
                row.update(pc=k + 1, dz=float(dz[k]), raw=float(raw[k]), evr=float(evr[k]))
                results.append(row)
                fits[(row["bin"], row["ratio_min"], row["frac_min"])] = (S, evr, ok)
    if not results:
        raise ValueError("no setting keeps enough bins or PCs - lower the minimum or use more or different files")
    results.sort(key=lambda r: (r["dz"], -r["n_features"]))
    filt = [r for r in results if r["filter"]]
    pool = filt or results
    best = pool[0]
    out = dict(refs=refs, n_files=len(idx), n_tested=len(results), min_features=int(min_features),
               min_var=float(min_var), best=best, results=results[:int(top)],
               baseline=[r for r in results if not r["filter"]][:len(bins)])
    out["p_corr"] = _perm_p(pool, fits, isA, best["dz"], n_perm)
    out["n_compared"] = int(sum(int(fits[(r["bin"], r["ratio_min"], r["frac_min"])][2].sum()) for r in pool))
    detail = _detail(store, rn, rl, isA, refs, best, normalize, sqrt, mass_max, tot, min_var)
    out.update(pcs=detail["pcs"], loadings=detail["loadings"], figure=detail["figure"])
    return out


def _perm_p(pool, fits, isA, observed, n_perm):
    """Chance of a minimum distance at least this small when the A/B labels are shuffled (same group sizes), taken
    over the same settings and eligible PCs as the real minimum, so that picking the best one is accounted for."""
    n, nA = len(isA), int(isA.sum())
    rng = np.random.default_rng(0)
    M = np.zeros((n_perm, n))
    for r in range(n_perm):
        M[r, rng.permutation(n)[:nA]] = 1.0
    null = np.full(n_perm, np.inf)
    for r in pool:
        S, _, ok = fits[(r["bin"], r["ratio_min"], r["frac_min"])]
        sd = np.maximum(S.std(0, ddof=1), 1e-300)
        d = np.abs(M @ S / nA - (1.0 - M) @ S / (n - nA)) / sd
        null = np.minimum(null, d[:, ok].min(1))
    return float((1 + np.sum(null <= observed + 1e-12)) / (n_perm + 1))


def _detail(store, rn, rl, isA, refs, best, normalize, sqrt, mass_max, tot, min_var):
    """Everything about the best setting: per-PC distances, loadings of the chosen PC and the figure."""
    from .figures import tune_figure
    X, s, inten, ratio, frac = _prep(store, rn, best["bin"], mass_max, tot)
    keep = (ratio >= best["ratio_min"]) & (frac >= best["frac_min"]) & (inten > 0)
    cols_all = np.flatnonzero(keep)
    S, evr, comps, cols = _fit(X[:, keep], s, normalize, sqrt)
    raw, dz = _dz(S, isA)
    pc = best["pc"] - 1
    feat = cols_all[cols]
    mz = (feat + 0.5) * best["bin"]
    load = comps[pc]
    top = np.argsort(-np.abs(load))[:12]
    loadings = [dict(mz=float(mz[j]), w=float(load[j])) for j in top]
    pcs = [dict(pc=k + 1, evr=float(evr[k]), raw=float(raw[k]), dz=float(dz[k]), eligible=bool(evr[k] >= min_var))
           for k in range(len(evr))]
    title = f"Closest PC of {refs[0]} and {refs[1]}"
    fig = tune_figure(dict(refs=refs, labels=rl, names=rn, scores=S, evr=evr, dz=dz, raw=raw, min_var=min_var, pc=pc,
                           mz=mz, loading=load, best=best, title=title))
    return dict(pcs=pcs, loadings=loadings, figure=fig)
