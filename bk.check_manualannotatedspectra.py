#!/usr/bin/env python3
"""check_manualannotatedspectra.py

Batch-process every sample folder written by data_load.py and plot the raw spectrum with the
expert-identified peaks (raw_peaks.parquet) overlaid.

For each sample it writes   <sample>/<ext>/plots/raw_spectrum_3_ranges.png   (0-150, 150-300, 300-800)
For each group of replicates (6PPD_neg1..6, 6PPD_pos1..6) it writes stacked, publication-style
figures (Spot 1/2/3 stacked, normalized intensity, x4 / x6 scaling beyond 400 / 600 u) into
<root>/plots/.

Usage:
  python check_manualannotatedspectra.py /home/cloud/tof_sims_driver/processed
  python check_manualannotatedspectra.py .../processed --xmin 300 --xmax 700 --mult 400:4,600:6 --spots 3
  python check_manualannotatedspectra.py .../processed --red 305:18      # optional red tick series
  python check_manualannotatedspectra.py .../processed --inspect         # also dump parquet/npz info
  Also writes one interactive HTML per replicate group (zoom, pan, hover, range slider):
  <root>/plots/interactive_<group>_<ext>.html   (--no-html to skip, --bin 0.01, --cdn)
"""
import argparse
import re
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANGES = [(0, 150), (150, 300), (300, 800)]

# --------------------------------------------------------------------------- helpers

def pretty(name):
    """IONTOF stores subscripts as C_4H_3F_2- ; render them with mathtext."""
    return re.sub(r"_(\d+)", r"$_{\1}$", name)


def load_peaks(directory):
    """Return the expert peak table (numeric masses only), or None."""
    p = Path(directory) / "raw_peaks.parquet"
    if not p.exists():
        print("  no raw_peaks.parquet: plotting without annotations")
        return None
    pk = pd.read_parquet(p)
    if "cmass" not in pk.columns:
        print("  WARNING: raw_peaks has no masses; re-run data_load.py (patched version)")
        return None
    for c in ("lmass", "cmass", "umass"):
        pk[c] = pd.to_numeric(pk[c], errors="coerce")
    pk = pk[pk["cmass"].notna()].copy()
    if pk.empty:
        print("  raw_peaks has no peaks with a valid mass")
        return None
    pk["lmass"] = pk["lmass"].fillna(pk["cmass"] - 0.5)
    pk["umass"] = pk["umass"].fillna(pk["cmass"] + 0.5)
    pk["name"] = pk["name"].fillna("").astype(str)
    print(f"  {len(pk)} annotated peaks, m/z {pk['cmass'].min():.2f}-{pk['cmass'].max():.2f}")
    return pk


def peak_heights(pk, mass, y):
    """Max of y inside each peak's [lmass, umass] window."""
    hs = []
    for l, u in zip(pk["lmass"], pk["umass"]):
        i0, i1 = np.searchsorted(mass, [l, u])
        hs.append(y[i0:i1].max() if i1 > i0 else 0.0)
    return np.asarray(hs)


_spark = None


def get_spark():
    global _spark
    if _spark is None:
        from pyspark.sql import SparkSession
        _spark = (SparkSession.builder.appName("SpectrumPlot").master("local[*]")
                  .config("spark.sql.shuffle.partitions", "8").getOrCreate())
        _spark.sparkContext.setLogLevel("ERROR")
    return _spark


def load_spectrum(path, lo=0, hi=800):
    """Read a raw_spectrum.parquet with Spark; return (mass, counts) sorted by mass."""
    from pyspark.sql import functions as F
    df = get_spark().read.parquet(str(path)).select("mass", "counts")
    sp = (df.filter((F.col("mass") >= lo) & (F.col("mass") < hi))
          .groupBy("mass").agg(F.sum("counts").alias("counts")).orderBy("mass"))
    pdf = sp.toPandas()
    return pdf["mass"].to_numpy(float), pdf["counts"].to_numpy(float)


# --------------------------------------------------------------------------- plots

def three_range(name, mass, counts, peaks, out_dir, top=40, log=False):
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    for ax, (low, high) in zip(axes, RANGES):
        sel = (mass >= low) & (mass < high)
        m, c = mass[sel], counts[sel]
        ax.plot(m, c, linewidth=0.7, color="black")
        ax.set_xlim(low, high)
        ax.set_xlabel("Mass (u)")
        ax.set_ylabel("Counts")
        ax.grid(True, alpha=0.2)
        if log:
            ax.set_yscale("log")
        title = f"Mass range: {low}-{high}"
        if peaks is not None and len(m):
            pk = peaks[(peaks["cmass"] >= low) & (peaks["cmass"] < high)].copy()
            pk["height"] = peak_heights(pk, m, c)
            title += f"  |  {len(pk)} annotated peaks"
            for _, r in pk.iterrows():
                ax.axvspan(r["lmass"], r["umass"], color="tab:orange", alpha=0.25, lw=0)
            for _, r in pk.nlargest(top, "height").iterrows():
                if r["height"] > 0:
                    ax.annotate(pretty(r["name"]) if r["name"] else f"{r['cmass']:.2f}",
                                (r["cmass"], r["height"]), xytext=(0, 6),
                                textcoords="offset points", rotation=90, ha="center",
                                va="bottom", fontsize=7, color="tab:red")
            ax.margins(y=0.25)
        ax.set_title(title)
    fig.suptitle(f"{name}: raw spectrum" + (" with expert annotations" if peaks is not None else ""))
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "raw_spectrum_3_ranges.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# def stacked(items, out_path, xmin, xmax, mult, red=None):
#     """Publication-style stacked spectra.

#     items: list of (name, mass, counts, peaks). Intensity is normalized to total counts;
#     the signal is multiplied by the factor of the last threshold in `mult` that is <= m/z.
#     """
#     n = len(items)
#     fig, axes = plt.subplots(n, 1, figsize=(11, 3.4 * n), sharex=True, squeeze=False)
#     axes = axes.ravel()

#     panels = []
#     for name, mass, counts, peaks in items:
#         sel = (mass >= xmin) & (mass <= xmax)
#         x = mass[sel]
#         y = counts[sel] / counts.sum()
#         f = np.ones_like(x)
#         for thr, k in sorted(mult):
#             f[x >= thr] = k
#         panels.append((x, y * f))
#     has_peaks = any(p is not None for _, _, _, p in items)
#     top = max((p[1].max() for p in panels if len(p[1])), default=1.0)
#     ymax = top * (1.6 if has_peaks else 1.25)

#     for ax, (name, _, _, peaks), (x, y) in zip(axes, items, panels):
#         ax.plot(x, y, color="black", linewidth=0.8)
#         ax.set_xlim(xmin, xmax)
#         ax.set_ylim(0, ymax)
#         ax.text(0.98, 0.88, name, transform=ax.transAxes, ha="right", fontsize=12)
#         for thr, _k in mult:
#             if xmin < thr < xmax:
#                 ax.axvline(thr, color="black", linestyle="--", linewidth=2)
#         if red:
#             start, step = red
#             for xr in np.arange(start, xmax, step):
#                 if xr >= xmin:
#                     ax.plot([xr, xr], [0.28 * ymax, 0.34 * ymax], color="red", linewidth=2)
#         if peaks is not None and len(x):
#             pk = peaks[(peaks["cmass"] >= xmin) & (peaks["cmass"] <= xmax)].copy()
#             pk["height"] = peak_heights(pk, x, y)
#             for _, r in pk.iterrows():
#                 base = r["height"] + 0.03 * ymax
#                 ax.plot([r["cmass"]] * 2, [base, base + 0.05 * ymax], color="blue", linewidth=2)
#                 ax.text(r["cmass"], base + 0.06 * ymax,
#                         pretty(r["name"]) if r["name"] else f"{r['cmass']:.0f}",
#                         rotation=90, ha="center", va="bottom", fontsize=9)

#     span = xmax - xmin
#     for thr, k in mult:
#         if xmin < thr < xmax:
#             axes[0].annotate("", xy=(thr + 0.07 * span, ymax * 1.06), xytext=(thr, ymax * 1.06),
#                              arrowprops=dict(arrowstyle="->"), annotation_clip=False)
#             axes[0].text(thr + 0.08 * span, ymax * 1.06, f"x {k}", va="center", fontsize=12,
#                          clip_on=False)

#     axes[-1].set_xlabel("m/z (amu)", fontsize=14)
#     fig.supylabel("Normalized intensity", fontsize=14)
#     fig.subplots_adjust(hspace=0.1)
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(out_path, dpi=200, bbox_inches="tight")
#     plt.close(fig)
#     print(f"saved {out_path}")


# --------------------------------------------------------------------------- interactive HTML

def sub_html(name):
    """C_4H_3F_2- -> C<sub>4</sub>H<sub>3</sub>F<sub>2</sub>-"""
    return re.sub(r"_(\d+)", r"<sub>\1</sub>", name)


def bin_spectrum(mass, counts, width, hi):
    """Sum counts onto a regular grid; bin i covers [i*width, (i+1)*width)."""
    n = int(np.ceil(hi / width))
    idx = np.floor(mass / width).astype(np.int64)
    ok = (idx >= 0) & (idx < n)
    return np.bincount(idx[ok], weights=counts[ok], minlength=n)


# Re-fits every Y axis to the tallest visible peak whenever the X range changes
# (range slider, preset buttons, scroll zoom); box-zoom keeps the Y range you drew.
INTERACTIVE_JS = r"""
var gd = document.getElementById('{plot_id}');
var N = __N__, CDX = __CDX__, COARSE = __COARSE__;
function ax(p, k){ return k === 1 ? p : p + k; }
function xrange(ev){
  for (var k = 1; k <= N; k++){
    var s = ax('xaxis', k);
    if (ev[s + '.range[0]'] !== undefined) return [ev[s + '.range[0]'], ev[s + '.range[1]']];
    if (ev[s + '.range']) return ev[s + '.range'];
  }
  return null;
}
function touchesY(ev){
  return Object.keys(ev).some(function(k){ return /^yaxis\d*\.(range|autorange|type)/.test(k); });
}
gd.on('plotly_relayout', function(ev){
  if (touchesY(ev)) return;
  var r = xrange(ev); if (!r) return;
  var lo = Math.min(r[0], r[1]), hi = Math.max(r[0], r[1]);
  var j0 = Math.max(0, Math.floor(lo / CDX)), j1 = Math.floor(hi / CDX), ymax = 0;
  for (var k = 0; k < N; k++){
    var c = COARSE[k];
    for (var j = j0; j <= j1 && j < c.length; j++) if (c[j] > ymax) ymax = c[j];
  }
  if (ymax <= 0) return;
  var upd = {};
  for (var k = 1; k <= N; k++){
    var ya = ax('yaxis', k);
    if (gd._fullLayout[ya] && gd._fullLayout[ya].type === 'log') return;
    upd[ya + '.range'] = [0, ymax * 1.3];
  }
  Plotly.relayout(gd, upd);
});
"""


def interactive(items, out_path, title, hi=800.0, width=0.01, xmin=300.0, xmax=700.0, cdn=False):
    """One interactive HTML per replicate group: stacked panels, linked zoom, hover values,
    range slider, preset ranges, log/linear Y, peak-label toggle."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("plotly is not installed -> skipping HTML (pip install plotly)")
        return
    n = len(items)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.03)
    peak_traces, coarse = [], []
    ratio = max(1, int(round(0.1 / width)))
    cdx = ratio * width
    label_default = "markers"

    for i, (name, mass, counts, peaks) in enumerate(items, start=1):
        sfx = "" if i == 1 else str(i)
        y = (bin_spectrum(mass, counts, width, hi) / counts.sum()).astype(np.float32)
        coarse.append([float(f"{v:.3g}") for v in np.maximum.reduceat(y, np.arange(0, len(y), ratio))])
        fig.add_trace(go.Scattergl(
            x0=width / 2, dx=width, y=y, mode="lines", name=name,
            line=dict(color="black", width=1),
            hovertemplate="m/z %{x:.3f}<br>%{y:.3e}<extra>" + name + "</extra>"), row=i, col=1)
        fig.add_annotation(text=f"<b>{name}</b>", xref=f"x{sfx} domain", yref=f"y{sfx} domain",
                           x=0.99, y=0.93, xanchor="right", showarrow=False, font=dict(size=13))
        if peaks is not None and len(peaks):
            pk = peaks.copy()
            centers = (np.arange(len(y)) + 0.5) * width
            pk["h"] = peak_heights(pk, centers, y)
            for _, r in pk.iterrows():
                fig.add_vrect(x0=r["lmass"], x1=r["umass"], fillcolor="orange", opacity=0.25,
                              line_width=0, layer="below", row=i, col=1)
            label_default = "markers+text" if len(pk) <= 25 else "markers"
            fig.add_trace(go.Scatter(
                x=pk["cmass"], y=pk["h"], mode=label_default, name=f"{name} peaks",
                text=[sub_html(t) or f"{m:.2f}" for t, m in zip(pk["name"], pk["cmass"])],
                textposition="top center", marker=dict(symbol="triangle-down", size=9, color="blue"),
                hovertemplate="<b>%{text}</b><br>m/z %{x:.3f}<br>%{y:.3e}<extra></extra>"),
                row=i, col=1)
            peak_traces.append(len(fig.data) - 1)

    def ymax_in(lo, hi_):
        j0, j1 = int(lo / cdx), int(hi_ / cdx) + 1
        return max((max(c[j0:j1], default=0) for c in coarse), default=0) or 1.0

    H = 260 * n + 240
    plot_h = H - 240
    ylim = ymax_in(xmin, xmax) * 1.3
    fig.update_layout(template="plotly_white", height=H, hovermode="x", showlegend=False,
                      margin=dict(t=150, b=90, l=80, r=30), dragmode="zoom",
                      title=dict(text=title, x=0.01, y=0.985, font=dict(size=18)))
    fig.update_xaxes(range=[xmin, xmax], showspikes=True, spikemode="across", spikethickness=1,
                     spikedash="dot", spikecolor="gray")
    fig.update_xaxes(title_text="m/z (amu)", row=n, col=1,
                     rangeslider=dict(visible=True, thickness=0.05, range=[0, hi]))
    fig.update_yaxes(range=[0, ylim], rangemode="tozero", title_text="Normalized intensity")

    xr = lambda a, b: {f"xaxis{'' if k == 1 else k}.range": [a, b] for k in range(1, n + 1)}
    yset = lambda **kw: {f"yaxis{'' if k == 1 else k}.{key}": v for k in range(1, n + 1)
                         for key, v in kw.items()}
    presets = [("Full", 0, hi), ("0-150", 0, 150), ("150-300", 150, 300), ("300-700", 300, 700),
               ("300-800", 300, hi)]
    menus = [dict(type="buttons", direction="right", showactive=False, x=0.0, xanchor="left",
                  y=1 + 42 / plot_h, yanchor="bottom", pad=dict(r=4),
                  buttons=[dict(label=l, method="relayout", args=[xr(a, b)]) for l, a, b in presets])]
    b2 = [dict(label="Linear Y", method="relayout", args=[{**yset(type="linear"), **yset(autorange=True)}]),
          dict(label="Log Y", method="relayout", args=[{**yset(type="log"), **yset(autorange=True)}])]
    if peak_traces:
        b2 += [dict(label="Peak labels on", method="restyle",
                    args=[{"mode": "markers+text"}, peak_traces]),
               dict(label="Peak labels off", method="restyle", args=[{"mode": "markers"}, peak_traces])]
    menus.append(dict(type="buttons", direction="right", showactive=False, x=0.0, xanchor="left",
                      y=1 + 8 / plot_h, yanchor="bottom", pad=dict(r=4), buttons=b2))
    fig.update_layout(updatemenus=menus)

    js = (INTERACTIVE_JS.replace("__N__", str(n)).replace("__CDX__", repr(cdx))
          .replace("__COARSE__", str(coarse)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn" if cdn else True, post_script=js,
                   config=dict(scrollZoom=True, displaylogo=False,
                               toImageButtonOptions=dict(format="png", scale=2)))
    print(f"saved {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


# --------------------------------------------------------------------------- optional inspection

def inspect_parquet(path):
    print("\n" + "=" * 80 + f"\nPARQUET: {path}\n" + "=" * 80)
    try:
        df = pd.read_parquet(path)
        print(f"Shape:   {df.shape}\nColumns: {list(df.columns)}\n\nDtypes:\n{df.dtypes}")
        print("\nFirst 10 rows:\n" + df.head(10).to_string())
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")


def plot_array(arr, key, path, output_dir, max_images=16):
    shape = arr.shape
    if arr.ndim == 2 or (arr.ndim == 3 and shape[-1] in (1, 3, 4)):
        images = arr[None, ...]
    elif arr.ndim == 3 or (arr.ndim == 4 and shape[-1] in (1, 3, 4)):
        images = arr
    else:
        print(f"  Not recognized as an image array: {shape}")
        return
    n = min(len(images), max_images)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    axes = axes.ravel()
    for i in range(n):
        img = images[i]
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img[..., 0]
        img = img.astype(np.float32)
        vmin, vmax = np.nanmin(img), np.nanmax(img)
        if vmin != vmax:
            img = (img - vmin) / (vmax - vmin)
        axes[i].imshow(img, cmap="gray" if img.ndim == 2 else None)
        axes[i].set_title(f"Image {i}")
        axes[i].axis("off")
    for i in range(n, len(axes)):
        axes[i].axis("off")
    fig.suptitle(f"{path.name}\n{key} | shape={shape} | dtype={arr.dtype}", fontsize=12)
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{path.stem}_{key}_raw.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {out}")


def inspect_npz(path, output_dir):
    print("\n" + "=" * 80 + f"\nNPZ: {path}\n" + "=" * 80)
    try:
        data = np.load(path, allow_pickle=False)
        print(f"Arrays: {len(data.files)}\nKeys:   {data.files}")
        for key in data.files:
            arr = data[key]
            print(f"\n{'-' * 60}\nKey: {key}  Shape: {arr.shape}  Dtype: {arr.dtype}")
            if arr.size:
                print(f"Min: {arr.min()}  Max: {arr.max()}  Mean: {arr.mean()}")
            if arr.ndim >= 2:
                plot_array(arr, key, path, output_dir)
        data.close()
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")


# --------------------------------------------------------------------------- driver

def find_datasets(root):
    """Every folder holding a raw_spectrum.parquet -> (sample, ext, folder)."""
    out = []
    for p in sorted(Path(root).rglob("raw_spectrum.parquet")):
        if "plots" in p.parts or "_probe" in p.parts:
            continue
        d = p.parent
        out.append((d.parent.name, d.name, d))
    return out


def natural_key(sample):
    m = re.match(r"^(.*?)(\d+)$", sample)
    return (m.group(1), int(m.group(2))) if m else (sample, 0)


def parse_mult(s):
    return [(float(a), float(b)) for a, b in (t.split(":") for t in s.split(",") if t)] if s else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="processed directory (or a single sample/ext directory)")
    ap.add_argument("--top", type=int, default=40, help="max labels per range in the 3-range plot")
    ap.add_argument("--log", action="store_true", help="log y axis in the 3-range plot")
    ap.add_argument("--xmin", type=float, default=300)
    ap.add_argument("--xmax", type=float, default=700)
    ap.add_argument("--mult", default="400:4,600:6",
                    help="scaling steps for the stacked plot, 'mz:factor,...'; empty string = none")
    ap.add_argument("--spots", type=int, default=3, help="spectra per stacked figure")
    ap.add_argument("--red", default=None, help="optional red tick series 'start:step', e.g. 305:18")
    ap.add_argument("--inspect", action="store_true", help="also dump other parquet/npz files")
    ap.add_argument("--no-html", action="store_true", help="skip the interactive HTML files")
    ap.add_argument("--bin", type=float, default=0.01, help="HTML bin width in u (smaller = bigger files)")
    ap.add_argument("--cdn", action="store_true", help="HTML loads plotly.js from a CDN (small files, needs internet)")
    a = ap.parse_args()

    root = Path(a.root)
    datasets = sorted(find_datasets(root), key=lambda t: (t[1], natural_key(t[0])))
    if not datasets:
        print(f"No raw_spectrum.parquet found under {root}")
        return
    print(f"Found {len(datasets)} samples: {[d[0] for d in datasets]}")
    mult = parse_mult(a.mult)
    red = tuple(float(v) for v in a.red.split(":")) if a.red else None
    hi = max(800, a.xmax + 1)

    loaded = {}   # (ext, sample) -> (mass, counts, peaks)
    for sample, ext, d in datasets:
        print(f"\n[{sample}/{ext}]")
        try:
            mass, counts = load_spectrum(d / "raw_spectrum.parquet", 0, hi)
            peaks = load_peaks(d)
            three_range(sample, mass, counts, peaks, d / "plots", a.top, a.log)
            loaded[(ext, sample)] = (mass, counts, peaks)
            if a.inspect:
                for p in sorted(d.glob("*")):
                    if p.suffix == ".parquet" and p.name != "raw_spectrum.parquet":
                        inspect_parquet(p)
                    elif p.suffix == ".npz":
                        inspect_npz(p, d / "plots")
        except Exception:
            print(f"  FAILED {sample}/{ext}:")
            traceback.print_exc()

    # stacked figures per replicate group (e.g. 6PPD_neg, 6PPD_pos)
    print("\n--- stacked figures ---")
    groups = {}
    for (ext, sample), (mass, counts, peaks) in loaded.items():
        groups.setdefault((ext, natural_key(sample)[0].rstrip("_")), []).append(
            (natural_key(sample)[1], sample, mass, counts, peaks))
    for (ext, grp), lst in sorted(groups.items()):
        lst.sort(key=lambda t: t[0])
        for i in range(0, len(lst), a.spots):
            chunk = lst[i:i + a.spots]
            items = [(s, m, c, p) for _, s, m, c, p in chunk]
            tag = f"{chunk[0][0]}-{chunk[-1][0]}"
            out = root / "plots" / f"stacked_{grp}_{tag}_{ext}_{int(a.xmin)}-{int(a.xmax)}.png"
            # stacked(items, out, a.xmin, a.xmax, mult, red)

    if not a.no_html:
        print("\n--- interactive HTML ---")
        for (ext, grp), lst in sorted(groups.items()):
            items = [(s, m, c, p) for _, s, m, c, p in sorted(lst, key=lambda t: t[0])]
            interactive(items, root / "plots" / f"interactive_{grp}_{ext}.html", f"{grp} ({ext})",
                        hi=hi, width=a.bin, xmin=a.xmin, xmax=a.xmax, cdn=a.cdn)

    if _spark is not None:
        _spark.stop()


if __name__ == "__main__":
    main()
