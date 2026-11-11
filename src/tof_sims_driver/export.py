"""Export PCA results as a zip of plain CSV tables, settings and the figures.

``results_zip([res_all, res_selected], name="my analysis")`` takes results returned by
:func:`tof_sims_driver.pca.compute_pca` (or loaded from a session file) and returns the bytes of a zip with, per PCA:

    counts_raw.csv          raw ion counts, one row per file, one column per feature (before any normalisation)
    scores.csv              where each file sits on every PC
    loadings.csv            one row per feature: m/z window, counts, share of ions, neighbour ratio, loading on every PC
    explained_variance.csv  share of the variance of every PC
    ellipses_90.csv         the 90% ellipse of every label group for every pair of PCs (what the scores figure draws)
    label_loadings.csv      per label group (only when files are labelled): every feature's difference from the average file and
                            the part of it carried by each PC (the per-label dot-plot figure)
    label_pc_scores.csv     per label group: mean score and share of its separation on every PC (the label x PC heat map)
    nearest_centroid.csv    per file: reference / unknown, nearest reference centroid (leave-one-out) and the distances to them
    files.csv, settings.json
    figures/*.png           the figures

so every picture can be redrawn in Excel, R or Python. Cells are written so a spreadsheet never runs them as formulas.
"""
import base64
import csv
import io
import json
import math
import re
import time
import zipfile

import numpy as np

from .figures import ellipse_params, label_loadings

FOLDERS = {"all": "all_peak_pca", "mine": "selected_peak_pca"}
TITLES = {"all": "All-peak PCA", "mine": "Selected-peak PCA"}
MAX_PNG_BYTES = 60 << 20
_PNG = "data:image/png;base64,"


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _arr(a, n=None):
    return isinstance(a, list) and (n is None or len(a) == n) and all(_num(v) for v in a)


def validate_result(res):
    """Raise ValueError (with a plain message) unless `res` is a usable PCA result."""
    if not isinstance(res, dict):
        raise ValueError("a result is not an object")
    if res.get("mode") not in FOLDERS:
        raise ValueError("a result has no valid mode")
    names = res.get("names")
    if not isinstance(names, list) or not names or not all(isinstance(v, str) for v in names):
        raise ValueError("file names are missing")
    n, k = len(names), res.get("n_components")
    peak_mz = res.get("peak_mz")
    p = len(peak_mz) if isinstance(peak_mz, list) else -1
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 50 or p < 1:
        raise ValueError("sizes are wrong")
    labels = res.get("labels")
    if not isinstance(labels, list) or len(labels) != n or not all(isinstance(v, str) for v in labels):
        raise ValueError("labels do not match the files")
    scores = res.get("scores")
    if not isinstance(scores, list) or len(scores) != n or not all(_arr(r, k) for r in scores):
        raise ValueError("scores do not match")
    if not _arr(res.get("explained"), k):
        raise ValueError("explained variance does not match")
    comps = res.get("comps")
    if not isinstance(comps, list) or len(comps) != k or not all(_arr(r, p) for r in comps):
        raise ValueError("loadings do not match")
    for key in ("peak_mz", "peak_lo", "peak_hi", "peak_counts", "peak_frac"):
        if not _arr(res.get(key), p):
            raise ValueError(f"{key} does not match")
    ratio = res.get("peak_ratio")
    if not isinstance(ratio, list) or len(ratio) != p or not all(v is None or _num(v) for v in ratio):
        raise ValueError("peak_ratio does not match")
    counts = res.get("counts")
    if counts is not None and (not isinstance(counts, list) or len(counts) != n or not all(_arr(r, p) for r in counts)):
        raise ValueError("the raw counts do not match")
    for f in res.get("figures") or []:
        if not (isinstance(f, dict) and isinstance(f.get("png"), str) and f["png"].startswith(_PNG) and len(f["png"]) < MAX_PNG_BYTES * 4 // 3 + 64):
            raise ValueError("a figure is not a valid image")


def _cell(v):
    """One CSV cell: floats keep full precision; text that a spreadsheet could read as a formula gets a leading apostrophe."""
    if v is None:
        return ""
    if isinstance(v, float):
        return "inf" if v == math.inf else repr(v)
    if isinstance(v, str) and v[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + v
    return v


def _csv(rows):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    for r in rows:
        w.writerow([_cell(v) for v in r])
    return buf.getvalue()


def _feature_names(res):
    """Column names of the features, ``lo-hi`` of each window; duplicates get a numeric suffix."""
    seen, out = {}, []
    for lo, hi in zip(res["peak_lo"], res["peak_hi"]):
        base = f"mz_{lo:g}-{hi:g}"
        seen[base] = seen.get(base, 0) + 1
        out.append(base if seen[base] == 1 else f"{base}_{seen[base]}")
    return out


def tables(res):
    """The CSV tables of one result as ``{file name: text}`` (``counts_raw.csv`` only when the result carries raw counts)."""
    validate_result(res)
    names, labels, k = res["names"], res["labels"], res["n_components"]
    pcs = [f"PC{i + 1}" for i in range(k)]
    feats = _feature_names(res)
    out = {}
    tot = res.get("total_ions")
    nt = res.get("norm_totals")
    nt = nt if isinstance(nt, list) and len(nt) == len(names) and all(isinstance(v, (int, float)) for v in nt) else None
    out["files.csv"] = _csv([["file", "label"] + (["total_ions"] if tot else []) + (["norm_total"] if nt else [])]
                            + [[n, lb] + ([tot[i]] if tot else []) + ([nt[i]] if nt else [])
                               for i, (n, lb) in enumerate(zip(names, labels))])
    if res.get("counts") is not None:
        out["counts_raw.csv"] = _csv([["file"] + feats] + [[n] + [int(v) for v in row] for n, row in zip(names, res["counts"])])
    nc = res.get("centroid")
    refs = nc.get("refs") if isinstance(nc, dict) else None
    if (isinstance(refs, list) and refs and all(isinstance(g, str) for g in refs)
            and all(isinstance(nc.get(k), list) and len(nc[k]) == len(names) for k in ("pred", "dist", "is_ref", "evaluated"))
            and all(isinstance(r, list) and len(r) == len(refs) for r in nc["dist"])):
        out["nearest_centroid.csv"] = _csv(
            [["file", "label", "role", "assigned_label", "correct"] + [f"distance_to_{g}" for g in refs]]
            + [[n, lb, "reference" if nc["is_ref"][i] else "unknown", nc["pred"][i],
                (nc["pred"][i] == lb) if nc["evaluated"][i] else ""] + ["" if d is None else d for d in nc["dist"][i]]
               for i, (n, lb) in enumerate(zip(names, labels))])
    out["scores.csv"] = _csv([["file", "label"] + pcs] + [[n, lb] + [float(v) for v in row]
                                                          for n, lb, row in zip(names, labels, res["scores"])])
    ev = [100.0 * float(v) for v in res["explained"]]
    out["explained_variance.csv"] = _csv([["pc", "explained_percent", "cumulative_percent"]]
                                         + [[pcs[i], ev[i], float(np.sum(ev[:i + 1]))] for i in range(k)])
    rmin, fmin = res.get("ratio_min", 3.0), res.get("frac_min", 0.005)
    comps = np.asarray(res["comps"], float)
    rows = [["mz", "lo", "hi", "counts", "percent_of_all_ions", "ratio_to_strongest_neighbour", "passes_filter", "max_abs_loading"]
            + [f"{c}_loading" for c in pcs]]
    for j in range(len(res["peak_mz"])):
        ratio = res["peak_ratio"][j]
        ok = (ratio is None or ratio >= rmin) and res["peak_frac"][j] >= fmin
        rows.append([res["peak_mz"][j], res["peak_lo"][j], res["peak_hi"][j], res["peak_counts"][j], 100.0 * res["peak_frac"][j],
                     math.inf if ratio is None else ratio, int(bool(ok)), float(np.abs(comps[:, j]).max())]
                    + [float(c) for c in comps[:, j]])
    out["loadings.csv"] = _csv(rows)
    S = np.asarray(res["scores"], float)
    rows = [["group", "pc_x", "pc_y", "n_files", "center_x", "center_y", "semi_major", "semi_minor", "angle_deg"]]
    for g in dict.fromkeys(lb.strip() for lb in labels if lb.strip()):
        ii = [i for i, lb in enumerate(labels) if lb.strip() == g]
        for a in range(k):
            for b in range(a + 1, k):
                e = ellipse_params(S[ii, a], S[ii, b])
                if e:
                    rows.append([g, pcs[a], pcs[b], e["n"], e["cx"], e["cy"], e["semi_major"], e["semi_minor"], e["angle_deg"]])
    out["ellipses_90.csv"] = _csv(rows)
    LL = label_loadings(res)
    if LL is not None:
        rows = [["label", "n_files", "feature", "mz", "lo", "hi", "d_vs_average_file"] + [f"{c}_part" for c in pcs]]
        for q, g in enumerate(LL["groups"]):
            for j in np.argsort(-np.abs(LL["d"][q])):
                rows.append([g, LL["n"][q], feats[j], res["peak_mz"][j], res["peak_lo"][j], res["peak_hi"][j], float(LL["d"][q][j])]
                            + [float(v) for v in LL["parts"][q][:, j]])
        out["label_loadings.csv"] = _csv(rows)
        out["label_pc_scores.csv"] = _csv([["label", "n_files"] + [f"{c}_mean_score" for c in pcs] + [f"{c}_share" for c in pcs]]
                                          + [[g, LL["n"][q]] + [float(v) for v in LL["tbar"][q]] + [float(v) for v in LL["share"][q]]
                                             for q, g in enumerate(LL["groups"])])
    return out


def _settings(res):
    keys = ("mode", "backend", "bin", "normalize", "sqrt", "n_components", "n_features", "ratio_min", "frac_min", "filter_peaks", "n_bins", "n_filtered", "mean_centered")
    d = {k: res[k] for k in keys if k in res}
    d["n_files"] = len(res["names"])
    d["tab"] = TITLES[res["mode"]]
    d["has_raw_counts"] = res.get("counts") is not None
    return json.dumps(d, indent=1)


def _slug(text, default="figure"):
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")[:40] or default


README = """TOF-SIMS PCA export
===================
{title}
Made {when} with tof-sims-driver {version}.

Each folder is one PCA. Every file is plain CSV (comma separated, one header row).

  counts_raw.csv          raw ion counts. One row per file, one column per feature (a column name like mz_22-23 is the m/z window
                          22 to 23). These are the numbers the PCA starts from, before normalising. Features that were identical
                          in every file (empty ones) were dropped before the PCA and are not listed.{counts_note}
  scores.csv              where each file sits on each PC (the scores figures and the interactive biplot).
  explained_variance.csv  the share of the differences between files that each PC captures (the scree figure).
  loadings.csv            one row per feature: its m/z window, counts added up over all files, share of all ions, how many times
                          higher it is than its strongest neighbour, whether it passes the thresholds in settings.json, and its
                          loading on every PC (the loading-spectrum and driving-peak figures).
  nearest_centroid.csv    only when 2 or more labels are references. For each file: its role (reference / unknown), the label of the
                          nearest reference centroid, whether that is right (reference files only, the file itself left out of its
                          centroid) and its distance to every reference centroid in the first two PCs.
  ellipses_90.csv         for each label group and each pair of PCs: the 90% ellipse (centre, semi-axes, angle of the long axis in
                          degrees from the x axis). Only groups with a label and 3 or more files get one (the scores figure).
  label_loadings.csv      only when files have labels. For each label and feature: how much higher (+) or lower (-) the label's average
                          spectrum is than the average of all files, and the part of that carried by each PC (the per-label dot-plot figure).
  label_pc_scores.csv     only when files have labels. For each label: its mean score on every PC and the share of its separation
                          each PC carries (the label x PC heat map).
  files.csv               the files in the order of the rows above, with your label for each and its total ion count. When the
                          peak filter was on, norm_total is the file's count over every bin (before filtering).
  settings.json           the settings of the run.
  figures/                the figures as images.

To redo the PCA from counts_raw.csv:
    X = counts_raw (files x features)
    X = X / X.sum(axis=1, keepdims=True)   # only if normalize is "tic"; if filter_peaks is true divide by norm_total (files.csv) instead
    X = sqrt(X)                            # only if sqrt is true
    Z = X - X.mean(axis=0)                 # mean centring, no scaling
    scores = Z @ loadings.T                # loadings = the PC1..PCk loading columns of loadings.csv, features x k
PC signs can differ between programs; the shape of the result does not.
"""


def results_zip(results, name=""):
    """Zip (as bytes) of the tables, settings and figures of one or two PCA results; see the module docstring."""
    results = [r for r in results if r]
    if not results:
        raise ValueError("nothing to export: there is no result")
    modes = [r.get("mode") for r in results]
    if len(set(modes)) != len(modes):
        raise ValueError("two results of the same PCA tab")
    from . import __version__
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        notes = []
        for res in results:
            validate_result(res)
            folder = FOLDERS[res["mode"]]
            for fname, text in tables(res).items():
                z.writestr(f"{folder}/{fname}", text)
            if res.get("counts") is None:
                notes.append(f"{TITLES[res['mode']]}: no raw counts (it was saved before they were included)")
            z.writestr(f"{folder}/settings.json", _settings(res))
            for i, f in enumerate(res.get("figures") or [], 1):
                z.writestr(f"{folder}/figures/{i:02d}_{_slug(f.get('id') or f.get('title'))}.png", base64.b64decode(f["png"][len(_PNG):]))
        z.writestr("README.txt", README.format(
            title=(str(name).strip()[:80] + "\n") if str(name).strip() else "", when=time.strftime("%Y-%m-%d %H:%M"), version=__version__,
            counts_note=("\n                          Note: " + "; ".join(notes) + ".") if notes else ""))
    return buf.getvalue()
