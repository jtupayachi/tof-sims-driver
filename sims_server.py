#!/usr/bin/env python3
"""sims_server.py - interactive TOF-SIMS browser (one page for positive, one for negative).

  python sims_server.py /home/cloud/tof_sims_driver/processed      # http://127.0.0.1:8765
  python sims_server.py .../processed --port 9000 --host 0.0.0.0

Open http://HOST:PORT/ and pick Positive or Negative. Needs raw_events_ch.npy /
raw_events_pix.npy / raw_summary.json / raw_peaks.parquet written by data_load.py (.itm files).

What the page does (all backed by the APIs below):
  * pick the polarity (pos / neg) and a single subsample (pos3, neg1, ...)
  * raw-count spectrum (not normalized), zoom with the bottom range slider
  * the file's own expert ("true") peaks AND peaks you create here, both toggleable
  * add / remove peaks (stored in <root>/user_peaks.json)
  * 2D image for the visible m/z range or any selected peak (true or user)
  * draw a box on the total image -> spectrum of just that region (ROI, down to one pixel)

APIs:
  GET  /                             the single-page app (?pol=pos|neg just preselects)
  GET  /plotly.js                    bundled Plotly (offline) or a CDN shim
  GET  /api/samples                  [{sample, polarity, w, h, events}]
  GET  /api/truepeaks?sample=S       expert peaks  [{name, cmass, lo, hi, group}]
  GET  /api/spectrum?sample=S&...    raw-count spectrum {x0, dx, y[], events, roi}
                                     optional box x0,y0,x1,y1 (pixels) -> ROI spectrum
                                     optional bin (u, default 0.02), mass (max u, default 900)
  GET  /api/image?sample=S&lo=&hi=   2D image, lo <= m/z < hi   (uint32 LE, headers X-W, X-H)
  GET  /api/image?sample=S&all=1     2D image of every ion
  GET  /api/peaks                    user peaks  [{id, name, lo, hi}]
  POST /api/peaks                    {name, lo, hi} -> create
  DELETE /api/peaks?id=N             remove
"""
import argparse
import json
import math
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

STEP = 8_000_000            # events per pass (bounds memory for wide ranges / ROI scans)
MASS_MAX = 900.0           # default upper m/z of the rebuilt spectrum
BIN = 0.02                 # default spectrum bin width (u)
STATIC_OK = {".html", ".png", ".json", ".css", ".js", ".svg", ".txt"}


def polarity_of(name):
    n = name.lower()
    return "pos" if "pos" in n else "neg" if "neg" in n else "other"


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.lock = threading.Lock()
        self.samples = {}
        self.truepeaks = {}                      # sample -> list (lazy, cached)
        self.veccache = {}                       # (name, bin, mass_max) -> coarse spectrum vector (PCA)
        self.pk_file = self.root / "user_peaks.json"
        for p in sorted(self.root.rglob("raw_events_ch.npy")):
            d = p.parent
            try:
                info = json.loads((d / "raw_summary.json").read_text())
                ch = np.load(p, mmap_mode="r")
                name = d.parent.name
                self.samples[name] = dict(
                    dir=d, sf=float(info["sf"]), k0=float(info["k0"]),
                    w=int(info["image_width"]), h=int(info["image_height"]),
                    polarity=polarity_of(name), ch=ch, pix=None)
            except Exception as e:
                print(f"  skip {d}: {type(e).__name__}: {e}")
        print(f"{len(self.samples)} samples with ion events: {list(self.samples)}")

    def _pix(self, s):
        if s["pix"] is None:
            s["pix"] = np.load(s["dir"] / "raw_events_pix.npy", mmap_mode="r")
        return s["pix"]

    def image(self, name, lo=None, hi=None):
        s = self.samples[name]
        ch, pix, W, H = s["ch"], self._pix(s), s["w"], s["h"]
        if lo is None:
            i0, i1 = 0, len(ch)
        else:                                   # channel t = sf*sqrt(m) + k0 (same as data_load.py)
            c0 = s["sf"] * math.sqrt(max(lo, 0.0)) + s["k0"]
            c1 = s["sf"] * math.sqrt(max(hi, 0.0)) + s["k0"]
            i0, i1 = (int(v) for v in np.searchsorted(ch, [math.ceil(c0), math.ceil(c1)]))
        acc = np.zeros(H * W, np.int64)
        for a in range(i0, i1, STEP):
            p = np.asarray(pix[a:min(a + STEP, i1)]).astype(np.int64)
            flat = (p >> 16) * W + (p & 0xFFFF)
            flat = flat[flat < H * W]
            acc += np.bincount(flat, minlength=H * W)
        return acc.astype(np.uint32).reshape(H, W)

    def spectrum(self, name, box=None, width=BIN, mass_max=MASS_MAX):
        """Raw-count spectrum rebuilt from the ion events. With box=(x0,y0,x1,y1) only ions whose
        pixel lies in that inclusive rectangle count (the ROI; a single pixel is x0==x1, y0==y1)."""
        s = self.samples[name]
        ch, sf, k0 = s["ch"], s["sf"], s["k0"]
        cmax = int(sf * math.sqrt(max(mass_max, 0.0)) + k0) + 2
        hist = np.zeros(cmax, np.int64)
        N = len(ch)
        events = 0
        pix = self._pix(s) if box is not None else None
        for a in range(0, N, STEP):
            b = min(a + STEP, N)
            c = np.asarray(ch[a:b]).astype(np.int64)
            if c[0] >= cmax:                     # events are sorted by channel -> nothing left
                break
            in_rng = c < cmax
            if box is None:
                cc = c[in_rng]
            else:
                x0, y0, x1, y1 = box
                p = np.asarray(pix[a:b]).astype(np.int64)
                xs, ys = p & 0xFFFF, p >> 16
                cc = c[in_rng & (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)]
            events += int(cc.size)
            if cc.size:
                hist += np.bincount(cc, minlength=cmax)[:cmax]
        cidx = np.arange(cmax)
        mass = ((cidx - k0) / sf) ** 2
        nb = int(mass_max / width) + 1
        bi = np.floor(mass / width).astype(np.int64)
        ok = (bi >= 0) & (bi < nb) & (hist > 0)
        y = np.bincount(bi[ok], weights=hist[ok], minlength=nb)[:nb].astype(np.int64)
        return dict(x0=width / 2, dx=width, y=y.tolist(), events=events, roi=box is not None)

    def coarse_vector(self, name, width, mass_max):
        """Whole-sample spectrum on a coarse mass grid, used as a PCA feature vector (cached)."""
        key = (name, round(width, 4), round(mass_max, 2))
        with self.lock:
            if key in self.veccache:
                return self.veccache[key]
        y = np.asarray(self.spectrum(name, None, width, mass_max)["y"], np.float64)
        with self.lock:
            self.veccache[key] = y
        return y

    def true_peaks(self, name):
        if name in self.truepeaks:
            return self.truepeaks[name]
        out = []
        f = self.samples[name]["dir"] / "raw_peaks.parquet"
        try:
            import pandas as pd
            df = pd.read_parquet(f)
            for c in ("lmass", "cmass", "umass"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df[df["cmass"].notna() & (df["name"].astype(str).str.lower() != "total")]
            for _, r in df.iterrows():
                cm = float(r["cmass"])
                lo = float(r["lmass"]) if np.isfinite(r["lmass"]) else cm - 0.25
                hi = float(r["umass"]) if np.isfinite(r["umass"]) else cm + 0.25
                out.append(dict(name=str(r["name"] or ""), cmass=cm, lo=lo, hi=hi,
                                group=str(r["group"]) if isinstance(r["group"], str) else ""))
            out.sort(key=lambda p: p["cmass"])
        except Exception as e:
            print(f"  true_peaks {name}: {type(e).__name__}: {e}")
        self.truepeaks[name] = out
        return out

    def peaks(self):
        if self.pk_file.exists():
            try:
                return json.loads(self.pk_file.read_text())
            except Exception:
                return []
        return []

    def save_peaks(self, lst):
        tmp = self.pk_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(lst, indent=1))
        tmp.replace(self.pk_file)


# --------------------------------------------------------------------------- Plotly (offline)

_PLOTLY = None


def plotly_js():
    global _PLOTLY
    if _PLOTLY is None:
        try:
            from plotly.offline import get_plotlyjs
            _PLOTLY = get_plotlyjs()
        except Exception:
            _PLOTLY = ("document.write('<script src=\"https://cdn.plot.ly/"
                       "plotly-2.35.2.min.js\"><\\/script>');")
    return _PLOTLY


# --------------------------------------------------------------------------- PCA

def _tohost(a):
    for attr in ("get", "to_numpy"):                 # cupy / cudf -> numpy
        f = getattr(a, attr, None)
        if callable(f):
            try:
                return np.asarray(f())
            except Exception:
                pass
    return np.asarray(a)


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


def compute_pca(store, names, labels, width, mass_max, ncomp, normalize, standardize):
    X = np.vstack([store.coarse_vector(n, width, mass_max) for n in names]).astype(np.float64)
    nfeat = X.shape[1]
    if normalize == "tic":                           # total-ion normalisation per sample
        s = X.sum(1, keepdims=True)
        s[s == 0] = 1.0
        X = X / s
    keep = X.std(0) > 0                              # drop empty / constant m/z bins
    Xk = X[:, keep]
    masses = (np.flatnonzero(keep) + 0.5) * width
    if Xk.shape[1] == 0:
        raise ValueError("no signal in the selected samples")
    if standardize:
        Xk = (Xk - Xk.mean(0)) / Xk.std(0)
    ncomp = max(1, min(int(ncomp), Xk.shape[0] - 1, Xk.shape[1]))
    scores, evr, comps, backend = run_pca(Xk, ncomp)
    scores, evr, comps = np.asarray(scores), np.asarray(evr), np.asarray(comps)
    loadings = []
    for c in range(comps.shape[0]):
        idx = np.argsort(-np.abs(comps[c]))[:10]
        loadings.append([{"mz": float(masses[i]), "w": float(comps[c][i])} for i in idx])
    return dict(backend=backend, names=names, labels=labels,
               scores=scores[:, :ncomp].tolist(), explained=evr[:ncomp].tolist(),
               n_features=int(Xk.shape[1]), n_components=int(ncomp), loadings=loadings)


def make_handler(store):
    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            pass

        def _send(self, code, body=b"", ctype="application/json", extra=None):
            if isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Expose-Headers", "X-W, X-H")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj), "application/json")

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path in ("", "/", "/index.html"):
                    return self._send(200, APP_HTML, "text/html")
                if u.path == "/plotly.js":
                    return self._send(200, plotly_js(), "application/javascript")
                if u.path == "/api/samples":
                    return self._json([dict(sample=n, polarity=s["polarity"], w=s["w"], h=s["h"],
                                            events=int(len(s["ch"]))) for n, s in store.samples.items()])
                if u.path == "/api/truepeaks":
                    name = q.get("sample", "")
                    if name not in store.samples:
                        return self._json({"error": "unknown sample"}, 404)
                    return self._json(store.true_peaks(name))
                if u.path == "/api/peaks":
                    return self._json(store.peaks())
                if u.path == "/api/spectrum":
                    name = q.get("sample", "")
                    if name not in store.samples:
                        return self._json({"error": "unknown sample"}, 404)
                    box = None
                    if all(k in q for k in ("x0", "y0", "x1", "y1")):
                        s = store.samples[name]
                        xa, xb = sorted((int(float(q["x0"])), int(float(q["x1"]))))
                        ya, yb = sorted((int(float(q["y0"])), int(float(q["y1"]))))
                        box = (max(0, xa), max(0, ya), min(s["w"] - 1, xb), min(s["h"] - 1, yb))
                    width = float(q.get("bin", BIN))
                    mass_max = float(q.get("mass", MASS_MAX))
                    return self._json(store.spectrum(name, box, width, mass_max))
                if u.path == "/api/image":
                    name = q.get("sample", "")
                    if name not in store.samples:
                        return self._json({"error": "unknown sample"}, 404)
                    if q.get("all"):
                        img = store.image(name)
                    else:
                        lo, hi = float(q["lo"]), float(q["hi"])
                        if not (math.isfinite(lo) and math.isfinite(hi) and hi > lo):
                            return self._json({"error": "need lo < hi"}, 400)
                        img = store.image(name, lo, hi)
                    return self._send(200, img.astype("<u4").tobytes(), "application/octet-stream",
                                      {"X-W": str(img.shape[1]), "X-H": str(img.shape[0])})
                return self._static(u.path)
            except (KeyError, ValueError) as e:
                self._json({"error": f"bad request: {e}"}, 400)
            except Exception as e:
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        def _static(self, path):
            f = (store.root / unquote(path).lstrip("/")).resolve()
            if store.root not in f.parents or not f.is_file() or f.suffix.lower() not in STATIC_OK:
                return self._json({"error": "not found"}, 404)
            return self._send(200, f.read_bytes(), mimetypes.guess_type(f.name)[0] or "application/octet-stream")

        def do_POST(self):
            path = urlparse(self.path).path
            if path == "/api/pca":
                return self._do_pca()
            if path != "/api/peaks":
                return self._json({"error": "not found"}, 404)
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                lo, hi = float(body["lo"]), float(body["hi"])
                name = str(body.get("name", "")).strip()[:80] or f"m/z {lo:.3f}-{hi:.3f}"
                if not (math.isfinite(lo) and math.isfinite(hi) and 0 <= lo < hi):
                    return self._json({"error": "need 0 <= lo < hi"}, 400)
            except Exception as e:
                return self._json({"error": f"bad request: {e}"}, 400)
            with store.lock:
                lst = store.peaks()
                p = dict(id=max([x["id"] for x in lst], default=0) + 1, name=name, lo=lo, hi=hi)
                lst.append(p)
                store.save_peaks(lst)
            self._json(p, 201)

        def _do_pca(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                items = body.get("samples", [])
                names = [str(it["sample"]) for it in items]
                labels = [str(it.get("label", "")) for it in items]
                if len(names) < 2:
                    return self._json({"error": "add at least 2 subsamples"}, 400)
                for n in names:
                    if n not in store.samples:
                        return self._json({"error": f"unknown sample {n}"}, 400)
                width = float(body.get("bin", 1.0))
                mass_max = float(body.get("mass", MASS_MAX))
                ncomp = int(body.get("n_components", 2))
                normalize = str(body.get("normalize", "tic"))
                standardize = bool(body.get("standardize", True))
            except Exception as e:
                return self._json({"error": f"bad request: {e}"}, 400)
            try:
                self._json(compute_pca(store, names, labels, width, mass_max, ncomp, normalize, standardize))
            except Exception as e:
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        def do_DELETE(self):
            u = urlparse(self.path)
            if u.path != "/api/peaks":
                return self._json({"error": "not found"}, 404)
            try:
                pid = int(parse_qs(u.query)["id"][0])
            except Exception:
                return self._json({"error": "need ?id=N"}, 400)
            with store.lock:
                store.save_peaks([x for x in store.peaks() if x["id"] != pid])
            self._json({"deleted": pid})

    return H


# --------------------------------------------------------------------------- single-page app

APP_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TOF-SIMS browser</title>
<script src="/plotly.js"></script>
<style>
  body{margin:0;font:13px/1.4 system-ui,Arial,sans-serif;color:#1a1a1a}
  .tabs{display:flex;gap:4px;padding:6px 10px 0;background:#e6e6ea;border-bottom:1px solid #bbb;
        position:sticky;top:0;z-index:40}
  .tabs button{padding:7px 18px;border:1px solid #bbb;border-bottom:none;background:#f3f3f6;
               cursor:pointer;border-radius:7px 7px 0 0;font-size:13px}
  .tabs button.on{background:#fff;font-weight:bold;border-color:#8a8a8a}
  .pane{display:none}
  .pane.on{display:block}
  .placeholder{max-width:760px;margin:30px auto;padding:0 20px;color:#333}
  .placeholder h2{margin:0 0 6px}
  .placeholder .soon{display:inline-block;background:#edf2fb;color:#2b6cb0;border:1px solid #c4d4ee;
                     border-radius:4px;padding:1px 8px;font-size:12px;margin-bottom:12px}
  .placeholder ul{line-height:1.7}
  header{display:flex;flex-wrap:wrap;gap:14px;align-items:center;padding:10px 14px;
         background:#f5f5f7;border-bottom:1px solid #ccc;position:sticky;top:39px;z-index:30}
  header b{font-size:15px}
  select,input,button{font:inherit;padding:3px 6px}
  .pol button{padding:4px 12px;border:1px solid #888;background:#fff;cursor:pointer}
  .pol button.on{background:#2b6cb0;color:#fff;border-color:#2b6cb0}
  main{display:flex;gap:14px;align-items:flex-start;padding:12px}
  #spec{flex:1 1 640px;min-width:420px}
  aside{flex:0 0 420px;display:flex;flex-direction:column;gap:12px}
  .card{border:1px solid #d0d0d0;border-radius:6px;padding:10px}
  .card h3{margin:0 0 8px;font-size:13px}
  .imgs{display:flex;gap:14px;flex-wrap:wrap}
  .imgwrap{text-align:center}
  .imgwrap canvas{image-rendering:pixelated;border:1px solid #999;background:#000;cursor:crosshair}
  .cap{font-size:11px;color:#444;max-width:180px;margin-top:2px}
  table{width:100%;border-collapse:collapse;font-size:12px}
  td,th{padding:2px 4px;border-bottom:1px solid #eee;text-align:left}
  .peaklist{max-height:200px;overflow:auto}
  .muted{color:#777}
  .tag{display:inline-block;padding:0 5px;border-radius:3px;font-size:11px;color:#fff}
  .tag.true{background:#2b6cb0}.tag.user{background:#c53030}
  button.sm{padding:1px 6px;font-size:11px}
  .roi{color:#c53030;font-weight:bold}
  label.inl{display:inline-flex;gap:4px;align-items:center}
</style>
</head>
<body>
<div class="tabs">
  <button id="tabbtn_spectra" class="on" data-pane="spectra">Spectra &amp; 2D</button>
  <button id="tabbtn_stats" data-pane="stats">Statistical methods</button>
  <button id="tabbtn_ml" data-pane="ml">ML / DL</button>
</div>

<section id="pane_spectra" class="pane on">
<header>
  <b>TOF-SIMS</b>
  <span class="pol"><button id="pol_pos">Positive</button><button id="pol_neg">Negative</button></span>
  <label>Subsample <select id="sample"></select></label>
  <label class="inl"><input type="checkbox" id="show_true" checked> true peaks</label>
  <label class="inl"><input type="checkbox" id="show_user" checked> my peaks</label>
  <label>bin <select id="bin"><option value="0.01">0.01</option>
     <option value="0.02" selected>0.02</option><option value="0.05">0.05</option>
     <option value="0.1">0.10</option></select> u</label>
  <span id="status" class="muted"></span>
</header>

<main>
  <div id="spec"></div>
  <aside>
    <div class="card">
      <h3>2D images <span class="muted" id="roi_state"></span></h3>
      <div style="margin-bottom:6px">
        <label>scale <select id="scale"><option>linear</option><option>sqrt</option><option>log</option></select></label>
        <button id="clear_roi" class="sm" disabled>clear ROI</button>          <button id="reset_2d" class="sm">reset 2D</button>        <span class="muted">drag a box on Total to make a ROI (down to 1 px)</span>
      </div>
      <div class="imgs">
        <div class="imgwrap"><div><b>Total</b> (all ions)</div>
          <canvas id="cv_total" width="128" height="128" style="width:180px"></canvas>
          <div class="cap" id="cap_total"></div></div>
        <div class="imgwrap"><div><b>Selection</b>
            <button id="dl_sel" class="sm" title="download a high-resolution PNG of this image">⬇ PNG</button></div>
          <canvas id="cv_sel" width="128" height="128" style="width:180px"></canvas>
          <div class="cap" id="cap_sel"></div></div>
      </div>
    </div>

    <div class="card">
      <h3>Add a peak</h3>
      <div style="display:flex;gap:6px;flex-wrap:wrap;align-items:center">
        <input id="pk_name" placeholder="name (optional)" style="width:120px">
        <input id="pk_lo" type="number" step="0.001" placeholder="lo m/z" style="width:80px">
        <input id="pk_hi" type="number" step="0.001" placeholder="hi m/z" style="width:80px">
        <button id="pk_fill" class="sm">from zoom</button>
        <button id="pk_add">add</button>
      </div>
      <div id="pk_msg" class="muted" style="margin-top:4px"></div>
    </div>

    <div class="card">
      <h3>Peaks</h3>
      <div class="peaklist">
        <table id="pk_table"><tbody></tbody></table>
      </div>
    </div>
  </aside>
</main>
</section>

<section id="pane_stats" class="pane">
  <div style="padding:14px;max-width:1150px;margin:auto">
    <h2 style="margin:4px 0">PCA of subsample spectra</h2>
    <p class="muted">Add subsamples, give each a label (group), then run PCA on their raw-count
       spectra. GPU-accelerated with cuML when available.</p>
    <div style="display:flex;gap:20px;flex-wrap:wrap;align-items:flex-start">
      <div class="card" style="flex:0 0 440px">
        <h3>Subsamples &amp; labels</h3>
        <div style="margin-bottom:6px;display:flex;gap:6px;flex-wrap:wrap">
          <button id="pca_add" class="sm">+ add</button>
          <button id="pca_add_pos" class="sm">+ all positive</button>
          <button id="pca_add_neg" class="sm">+ all negative</button>
          <button id="pca_clear" class="sm">clear</button>
        </div>
        <table id="pca_rows"><tbody></tbody></table>
        <h3 style="margin-top:12px">Parameters</h3>
        <div style="display:grid;grid-template-columns:auto auto;gap:7px 10px;align-items:center;max-width:320px">
          <label>components</label><input id="pca_nc" type="number" min="2" max="10" value="2" style="width:70px">
          <label>feature bin (u)</label><select id="pca_bin"><option>0.5</option><option selected>1</option><option>2</option><option>5</option></select>
          <label>normalize</label><select id="pca_norm"><option value="tic">total-ion</option><option value="none">none</option></select>
          <label>standardize (z-score)</label><input id="pca_std" type="checkbox" checked>
        </div>
        <button id="pca_run" style="margin-top:12px">Run PCA</button>
        <span id="pca_msg" class="muted" style="margin-left:8px"></span>
      </div>
      <div style="flex:1 1 520px;min-width:440px">
        <div id="pca_scores" style="height:430px"></div>
        <div id="pca_scree" style="height:230px"></div>
        <div id="pca_load" style="font-size:12px;color:#333"></div>
      </div>
    </div>
  </div>
</section>

<section id="pane_ml" class="pane">
  <div class="placeholder">
    <h2>ML / DL</h2>
    <div class="soon">coming soon</div>
    <p>Machine-learning on the spectra and images. Planned:</p>
    <ul>
      <li>Dimensionality reduction (PCA / UMAP) of per-pixel or per-sample spectra.</li>
      <li>Unsupervised clustering / phase maps over the 2D field of view.</li>
      <li>Supervised classification of subsamples or regions from their spectra.</li>
      <li>Non-negative matrix factorisation of the image stack into components.</li>
      <li>Model training, inspection and overlay of predictions back onto the images.</li>
    </ul>
  </div>
</section>

<script>
const $ = id => document.getElementById(id);
const API = p => fetch(p).then(r => r.json());
// one in-flight request per key; starting a new one aborts the previous (no pile-ups / stuck UI)
const CTRL = {};
function sig(key){ if (CTRL[key]) CTRL[key].abort(); const c = new AbortController(); CTRL[key] = c; return c.signal; }
function abortAll(){ Object.values(CTRL).forEach(c => { try { c.abort(); } catch(e){} }); }
const isAbort = e => e && e.name === 'AbortError';
async function apiKey(p, key){ const r = await fetch(p, {signal: sig(key)}); return r.json(); }
const MASSMAX = 900;
let POL = (new URLSearchParams(location.search).get('pol') || 'pos').startsWith('neg') ? 'neg' : 'pos';
let SAMPLES = [], SAMPLE = null, SPEC = null, ROI = null;
let TRUE = [], USER = [];
const sub = s => (s || '').replace(/_(\d+)/g, '$1');   // C_4H_3 -> C4H3 (compact label)

// ---- spectrum helpers -----------------------------------------------------
function xAt(i){ return SPEC.x0 + i * SPEC.dx; }
function idxAt(mz){ return Math.round((mz - SPEC.x0) / SPEC.dx); }
function peakHeight(lo, hi){
  if (!SPEC) return 0;
  let i0 = Math.max(0, idxAt(lo)), i1 = Math.min(SPEC.y.length - 1, idxAt(hi)), m = 0;
  for (let i = i0; i <= i1; i++) if (SPEC.y[i] > m) m = SPEC.y[i];
  return m;
}
function ymaxIn(lo, hi){
  let i0 = Math.max(0, idxAt(lo)), i1 = Math.min(SPEC.y.length - 1, idxAt(hi)), m = 0;
  for (let i = i0; i <= i1; i++) if (SPEC.y[i] > m) m = SPEC.y[i];
  return m || 1;
}

// ---- Plotly spectrum ------------------------------------------------------
let relTimer = null, lastX = [0, MASSMAX];

function markerTrace(list, color, name){
  const xs = [], ys = [], tx = [];
  list.forEach(p => { const c = p.cmass != null ? p.cmass : (p.lo + p.hi) / 2;
    xs.push(c); ys.push(peakHeight(p.lo, p.hi)); tx.push(sub(p.name) || c.toFixed(2)); });
  return {x: xs, y: ys, text: tx, mode: 'markers+text', name, textposition: 'top center',
          textfont: {size: 9, color}, cliponaxis: false,
          marker: {symbol: 'triangle-down', size: 9, color},
          hovertemplate: '<b>%{text}</b><br>m/z %{x:.3f}<br>%{y:,}<extra>' + name + '</extra>'};
}

function renderSpectrum(keepX){
  if (!SPEC) return;
  const traces = [{type: 'scattergl', x0: SPEC.x0, dx: SPEC.dx, y: SPEC.y, mode: 'lines',
                   line: {color: '#111', width: 1}, name: SAMPLE,
                   hovertemplate: 'm/z %{x:.3f}<br>%{y:,} counts<extra></extra>'}];
  const shapes = [];
  function addPeaks(list, color){
    list.forEach(p => shapes.push({type: 'rect', xref: 'x', yref: 'paper', y0: 0, y1: 1,
      x0: p.lo, x1: p.hi, fillcolor: color, opacity: 0.15, line: {width: 0}, layer: 'below'}));
  }
  if ($('show_true').checked){ addPeaks(TRUE, '#2b6cb0'); traces.push(markerTrace(TRUE, '#2b6cb0', 'true peaks')); }
  if ($('show_user').checked){ addPeaks(USER, '#c53030'); traces.push(markerTrace(USER, '#c53030', 'my peaks')); }

  const x = keepX ? lastX : [SPEC.roi ? SPEC.x0 : 0, MASSMAX];
  const layout = {
    margin: {t: 36, r: 12, b: 60, l: 70}, hovermode: 'x', dragmode: 'zoom', showlegend: false,
    title: {text: SAMPLE + (SPEC.roi ? '  —  ROI (' + SPEC.events.toLocaleString() + ' ions)'
                                     : '  —  whole spectrum (' + SPEC.events.toLocaleString() + ' ions)'),
            x: 0.01, font: {size: 15}},
    xaxis: {title: 'm/z (u)', range: x.slice(), rangeslider: {visible: true, thickness: 0.07, range: [0, MASSMAX]},
            showspikes: true, spikemode: 'across', spikethickness: 1, spikedash: 'dot', spikecolor: '#999'},
    yaxis: {title: 'Counts (raw)', rangemode: 'tozero', range: [0, ymaxIn(x[0], x[1]) * 1.25]},
    shapes
  };
  Plotly.react('spec', traces, layout, {scrollZoom: true, displaylogo: false, responsive: true});
  wireSpec();
}

let wired = false;
function wireSpec(){
  if (wired) return; wired = true;
  const gd = $('spec');
  gd.on('plotly_relayout', ev => {
    let r = null;
    if (ev['xaxis.range']) r = ev['xaxis.range'];
    else if (ev['xaxis.range[0]'] !== undefined) r = [ev['xaxis.range[0]'], ev['xaxis.range[1]']];
    if (!r) return;
    lastX = [Math.min(r[0], r[1]), Math.max(r[0], r[1])];
    const touchesY = Object.keys(ev).some(k => /^yaxis\.(range|autorange)/.test(k));
    if (!touchesY) Plotly.relayout(gd, {'yaxis.range': [0, ymaxIn(lastX[0], lastX[1]) * 1.25]});
    clearTimeout(relTimer);
    relTimer = setTimeout(() => updateSelection(lastX[0], lastX[1]), 220);
  });
  gd.on('plotly_click', ev => {
    if (!ev.points || !ev.points.length) return;
    const p = ev.points[0];
    const all = TRUE.concat(USER);
    let best = null, bd = 1e9;
    all.forEach(q => { const d = Math.abs((q.cmass != null ? q.cmass : (q.lo + q.hi) / 2) - p.x);
      if (d < bd){ bd = d; best = q; } });
    if (best && bd < 1) updateSelection(best.lo, best.hi, sub(best.name) || best.cmass.toFixed(3));
    else updateSelection(p.x - SPEC.dx, p.x + SPEC.dx);
  });
}

// ---- 2D images ------------------------------------------------------------
const LUT = []; for (let i = 0; i < 256; i++){ const x = i / 255, c = v => Math.round(255 * Math.min(1, Math.max(0, v)));
  LUT.push([c(2 * x), c(2 * x - 0.5), c(2 * x - 1)]); }     // afmhot
let TOTAL = null;    // {arr, w, h}
let selMeta = null;  // last Selection image {lo, hi, label} for the download button

async function fetchImage(params, key){
  const r = await fetch('/api/image?' + params, key ? {signal: sig(key)} : undefined);
  const w = +r.headers.get('X-W'), h = +r.headers.get('X-H');
  const buf = await r.arrayBuffer();
  return {arr: new Uint32Array(buf), w, h};
}
function drawImage(cv, img, scale){
  cv.width = img.w; cv.height = img.h;
  let vmax = 0; for (let i = 0; i < img.arr.length; i++) if (img.arr[i] > vmax) vmax = img.arr[i];
  const ctx = cv.getContext('2d'), id = ctx.createImageData(img.w, img.h), lv = Math.log1p(vmax);
  for (let i = 0; i < img.arr.length; i++){
    const v = img.arr[i];
    let t = vmax > 0 ? (scale === 'sqrt' ? Math.sqrt(v / vmax) : scale === 'log' ? Math.log1p(v) / lv : v / vmax) : 0;
    const c = LUT[Math.max(0, Math.min(255, Math.round(t * 255)))];
    id.data[4 * i] = c[0]; id.data[4 * i + 1] = c[1]; id.data[4 * i + 2] = c[2]; id.data[4 * i + 3] = 255;
  }
  ctx.putImageData(id, 0, 0);
  cv._img = img;
  return vmax;
}
function fmt(v){ return v >= 1e5 ? v.toExponential(2) : Math.round(v).toLocaleString(); }
function stats(arr){ let mx = 0, tc = 0; for (let i = 0; i < arr.length; i++){ tc += arr[i]; if (arr[i] > mx) mx = arr[i]; } return [mx, tc]; }

async function loadTotal(){
  try { TOTAL = await fetchImage('sample=' + encodeURIComponent(SAMPLE) + '&all=1', 'total'); }
  catch (e) { if (isAbort(e)) return; $('cap_total').textContent = 'image error'; return; }
  drawImage($('cv_total'), TOTAL, $('scale').value);
  const [mx, tc] = stats(TOTAL.arr);
  $('cap_total').textContent = TOTAL.w + 'x' + TOTAL.h + ' px | MC ' + fmt(mx) + ' | TC ' + fmt(tc);
  drawRoiRect();
}
let selToken = 0;
async function updateSelection(lo, hi, label){
  if (!SAMPLE || lo >= hi) return;
  const tok = ++selToken;
  let img;
  try { img = await fetchImage('sample=' + encodeURIComponent(SAMPLE) + '&lo=' + lo + '&hi=' + hi, 'sel'); }
  catch (e) { if (isAbort(e)) return; $('cap_sel').textContent = 'image error'; return; }
  if (tok !== selToken) return;
  selMeta = {lo, hi, label: label || ''};
  drawImage($('cv_sel'), img, $('scale').value);
  const [mx, tc] = stats(img.arr);
  $('cap_sel').textContent = (label ? label + ' | ' : '') + 'm/z ' + lo.toFixed(2) + '–' + hi.toFixed(2)
    + ' | MC ' + fmt(mx) + ' | TC ' + fmt(tc);
}
function downloadSelection(){
  const cv = $('cv_sel');
  if (!cv._img) return;
  const scale = 8;                                   // 128 px -> 1024 px, crisp (no smoothing)
  const big = document.createElement('canvas');
  big.width = cv._img.w * scale; big.height = cv._img.h * scale;
  const ctx = big.getContext('2d'); ctx.imageSmoothingEnabled = false;
  ctx.drawImage(cv, 0, 0, big.width, big.height);
  const tag = selMeta ? 'mz' + selMeta.lo.toFixed(2) + '-' + selMeta.hi.toFixed(2) : 'sel';
  big.toBlob(b => { const a = document.createElement('a'); a.href = URL.createObjectURL(b);
    a.download = ((SAMPLE || 'sample') + '_' + tag + '.png').replace(/[^\w.-]+/g, '_');
    a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000); }, 'image/png');
}

// ---- ROI drawing on the Total image --------------------------------------
let dragging = null;
function cvPixel(cv, e){
  const r = cv.getBoundingClientRect();
  return [Math.max(0, Math.min(cv.width - 1, Math.floor((e.clientX - r.left) / r.width * cv.width))),
          Math.max(0, Math.min(cv.height - 1, Math.floor((e.clientY - r.top) / r.height * cv.height)))];
}
function drawRoiRect(){
  if (!TOTAL) return;
  const cv = $('cv_total');
  drawImage(cv, TOTAL, $('scale').value);
  if (!ROI) return;
  const ctx = cv.getContext('2d');
  ctx.strokeStyle = '#39ff14'; ctx.lineWidth = Math.max(1, cv.width / 128);
  ctx.strokeRect(ROI[0], ROI[1], ROI[2] - ROI[0] + 1, ROI[3] - ROI[1] + 1);
}
function initRoi(){
  const cv = $('cv_total');
  cv.onmousedown = e => { dragging = cvPixel(cv, e); };
  cv.onmousemove = e => {
    if (!TOTAL) return;
    const [x, y] = cvPixel(cv, e);
    cv.title = 'x ' + x + ', y ' + y + ': ' + (TOTAL.arr[y * cv.width + x] || 0);
    if (dragging){ ROI = [Math.min(dragging[0], x), Math.min(dragging[1], y),
                          Math.max(dragging[0], x), Math.max(dragging[1], y)]; drawRoiRect(); }
  };
  cv.onmouseup = e => {
    if (!dragging) return;
    const [x, y] = cvPixel(cv, e);
    ROI = [Math.min(dragging[0], x), Math.min(dragging[1], y),
           Math.max(dragging[0], x), Math.max(dragging[1], y)];
    dragging = null; drawRoiRect(); applyRoi();
  };
}
async function applyRoi(){
  $('clear_roi').disabled = false;
  const [x0, y0, x1, y1] = ROI;
  $('roi_state').textContent = 'ROI x' + x0 + '–' + x1 + ', y' + y0 + '–' + y1;
  $('status').innerHTML = '<span class="roi">loading ROI spectrum…</span>';
  let spec;
  try {
    spec = await apiKey('/api/spectrum?sample=' + encodeURIComponent(SAMPLE)
      + '&bin=' + $('bin').value + '&x0=' + x0 + '&y0=' + y0 + '&x1=' + x1 + '&y1=' + y1, 'spec');
  } catch (e) {
    if (isAbort(e)) return;
    $('status').innerHTML = '<span class="roi">ROI failed: ' + e.message + ' (try reset 2D)</span>';
    return;
  }
  SPEC = spec; recomputeMarkers(); renderSpectrum(true);
  $('status').innerHTML = '<span class="roi">ROI: ' + SPEC.events.toLocaleString() + ' ions</span>';
}
function clearRoi(){
  abortAll();
  ROI = null; $('clear_roi').disabled = true; $('roi_state').textContent = ''; drawRoiRect();
  loadSpectrum(false);
}
// reset the 2D panel (and any ROI) to a clean state: whole spectrum, total image, full-range selection
async function reset2D(){
  abortAll();
  ROI = null; $('clear_roi').disabled = true; $('roi_state').textContent = '';
  lastX = [0, MASSMAX];
  await loadSpectrum(false);
  await loadTotal();
  await updateSelection(0, MASSMAX, 'full');
}

// ---- peaks ----------------------------------------------------------------
function recomputeMarkers(){
  TRUE.forEach(p => p.h = peakHeight(p.lo, p.hi));
  USER.forEach(p => p.h = peakHeight(p.lo, p.hi));
}
function renderPeakTable(){
  const tb = $('pk_table').querySelector('tbody');
  const rows = [];
  rows.push('<tr><th>type</th><th>name</th><th>m/z</th><th>window</th><th></th></tr>');
  TRUE.forEach(p => rows.push('<tr><td><span class="tag true">true</span></td><td>' + (sub(p.name) || '—')
    + '</td><td>' + p.cmass.toFixed(3) + '</td><td>' + p.lo.toFixed(2) + '–' + p.hi.toFixed(2)
    + '</td><td><button class="sm" data-v="' + p.lo + ',' + p.hi + ',' + (sub(p.name) || p.cmass.toFixed(2))
    + '">2D</button></td></tr>'));
  USER.forEach(p => rows.push('<tr><td><span class="tag user">mine</span></td><td>' + (p.name || '—')
    + '</td><td>' + ((p.lo + p.hi) / 2).toFixed(3) + '</td><td>' + p.lo.toFixed(2) + '–' + p.hi.toFixed(2)
    + '</td><td><button class="sm" data-v="' + p.lo + ',' + p.hi + ',' + (p.name || '')
    + '">2D</button> <button class="sm" data-del="' + p.id + '">✕</button></td></tr>'));
  tb.innerHTML = rows.join('');
  tb.querySelectorAll('button[data-v]').forEach(b => b.onclick = () => {
    const [lo, hi, lab] = b.dataset.v.split(','); updateSelection(+lo, +hi, lab); });
  tb.querySelectorAll('button[data-del]').forEach(b => b.onclick = async () => {
    await fetch('/api/peaks?id=' + b.dataset.del, {method: 'DELETE'}); await loadUserPeaks(); });
}
async function loadUserPeaks(){
  USER = await API('/api/peaks');
  recomputeMarkers(); renderPeakTable(); renderSpectrum(true);
}
async function addPeak(){
  const lo = parseFloat($('pk_lo').value), hi = parseFloat($('pk_hi').value);
  if (!(lo < hi)){ $('pk_msg').textContent = 'need lo < hi'; return; }
  const r = await fetch('/api/peaks', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({name: $('pk_name').value, lo, hi})});
  const j = await r.json();
  $('pk_msg').textContent = r.ok ? 'added ' + (j.name || '') : (j.error || 'error');
  if (r.ok){ $('pk_name').value = $('pk_lo').value = $('pk_hi').value = ''; await loadUserPeaks(); }
}

// ---- loaders --------------------------------------------------------------
async function loadSpectrum(keepX){
  $('status').textContent = 'loading spectrum…';
  try { SPEC = await apiKey('/api/spectrum?sample=' + encodeURIComponent(SAMPLE) + '&bin=' + $('bin').value, 'spec'); }
  catch (e) { if (isAbort(e)) return; $('status').textContent = 'spectrum error: ' + e.message; return; }
  recomputeMarkers(); renderSpectrum(keepX);
  $('status').textContent = SPEC.events.toLocaleString() + ' ions';
}
async function loadSample(name){
  SAMPLE = name; ROI = null; $('clear_roi').disabled = true; $('roi_state').textContent = '';
  lastX = [0, MASSMAX];
  TRUE = await API('/api/truepeaks?sample=' + encodeURIComponent(name));
  await loadUserPeaks();                       // sets USER and renders table
  await loadSpectrum(false);
  await loadTotal();
  updateSelection(0, MASSMAX);
}
function fillSamples(){
  const sel = $('sample'); sel.innerHTML = '';
  const list = SAMPLES.filter(s => s.polarity === POL).sort((a, b) => a.sample.localeCompare(b.sample));
  list.forEach(s => { const o = document.createElement('option'); o.value = s.sample;
    o.textContent = s.sample + '  (' + s.events.toLocaleString() + ' ions)'; sel.appendChild(o); });
  $('pol_pos').classList.toggle('on', POL === 'pos');
  $('pol_neg').classList.toggle('on', POL === 'neg');
  if (list.length) loadSample(list[0].sample);
  else { $('spec').innerHTML = '<p class="muted" style="padding:20px">No ' + POL + ' samples found.</p>'; }
}

// ---- wire up --------------------------------------------------------------
$('pol_pos').onclick = () => { POL = 'pos'; fillSamples(); };
$('pol_neg').onclick = () => { POL = 'neg'; fillSamples(); };
$('sample').onchange = e => loadSample(e.target.value);
$('show_true').onchange = $('show_user').onchange = () => renderSpectrum(true);
$('bin').onchange = () => ROI ? applyRoi() : loadSpectrum(true);
$('scale').onchange = () => { drawRoiRect(); if ($('cv_sel')._img) drawImage($('cv_sel'), $('cv_sel')._img, $('scale').value); };
$('clear_roi').onclick = clearRoi;
$('reset_2d').onclick = reset2D;
$('pk_add').onclick = addPeak;
$('pk_fill').onclick = () => { $('pk_lo').value = lastX[0].toFixed(3); $('pk_hi').value = lastX[1].toFixed(3); };

// ---- PCA (Statistical methods tab) ----------------------------------------
let pcaRows = [];
function sampleOptions(sel){
  return SAMPLES.map(s => '<option value="' + s.sample + '"' + (s.sample === sel ? ' selected' : '') + '>'
    + s.sample + '</option>').join('');
}
function renderPcaRows(){
  const tb = $('pca_rows').querySelector('tbody');
  if (!pcaRows.length){ tb.innerHTML = '<tr><td class="muted">no subsamples yet — use the buttons above</td></tr>'; return; }
  tb.innerHTML = pcaRows.map((r, i) =>
    '<tr><td><select data-i="' + i + '" class="pca_s">' + sampleOptions(r.sample) + '</select></td>'
    + '<td><input data-i="' + i + '" class="pca_l" value="' + (r.label || '').replace(/"/g, '&quot;')
    + '" placeholder="label" style="width:110px"></td>'
    + '<td><button class="sm" data-del="' + i + '">\u2715</button></td></tr>').join('');
  tb.querySelectorAll('.pca_s').forEach(s => s.onchange = e => pcaRows[+e.target.dataset.i].sample = e.target.value);
  tb.querySelectorAll('.pca_l').forEach(s => s.oninput = e => pcaRows[+e.target.dataset.i].label = e.target.value);
  tb.querySelectorAll('button[data-del]').forEach(b => b.onclick = () => { pcaRows.splice(+b.dataset.del, 1); renderPcaRows(); });
}
function pcaAdd(){ pcaRows.push({sample: SAMPLES[0] && SAMPLES[0].sample, label: ''}); renderPcaRows(); }
function pcaAddPol(pol){ SAMPLES.filter(s => s.polarity === pol).forEach(s => pcaRows.push({sample: s.sample, label: pol})); renderPcaRows(); }

const PAL = ['#2b6cb0','#c53030','#2f855a','#b7791f','#6b46c1','#319795','#d53f8c','#718096'];
async function runPCA(){
  if (pcaRows.length < 2){ $('pca_msg').textContent = 'add at least 2 subsamples'; return; }
  $('pca_msg').textContent = 'running… (first run loads cuML, ~10 s)';
  let res;
  try {
    const r = await fetch('/api/pca', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({samples: pcaRows, n_components: +$('pca_nc').value, bin: +$('pca_bin').value,
        normalize: $('pca_norm').value, standardize: $('pca_std').checked})});
    res = await r.json();
    if (!r.ok){ $('pca_msg').textContent = res.error || 'error'; return; }
  } catch (e) { $('pca_msg').textContent = 'error: ' + e.message; return; }
  $('pca_msg').textContent = res.backend + '  |  ' + res.n_features + ' features';
  renderPCA(res);
}
function renderPCA(res){
  const labels = res.labels.map((l, i) => l || res.names[i]);
  const groups = {};
  res.scores.forEach((sc, i) => { const g = labels[i] || '?'; (groups[g] = groups[g] || []).push({sc, name: res.names[i]}); });
  const ev = res.explained.map(v => (100 * v).toFixed(1));
  const traces = Object.keys(groups).map((g, k) => ({
    x: groups[g].map(p => p.sc[0]), y: groups[g].map(p => p.sc[1] != null ? p.sc[1] : 0),
    text: groups[g].map(p => p.name), mode: 'markers+text', textposition: 'top center', name: g,
    marker: {size: 12, color: PAL[k % PAL.length]}, type: 'scatter'}));
  Plotly.newPlot('pca_scores', traces, {
    title: 'PCA scores (' + res.backend + ')', margin: {t: 40},
    xaxis: {title: 'PC1 (' + ev[0] + '%)', zeroline: true},
    yaxis: {title: 'PC2 (' + (ev[1] || '0') + '%)', zeroline: true},
    legend: {title: {text: 'label'}}}, {displaylogo: false, responsive: true});
  Plotly.newPlot('pca_scree', [{x: res.explained.map((_, i) => 'PC' + (i + 1)),
    y: res.explained.map(v => 100 * v), type: 'bar', marker: {color: '#2b6cb0'}}],
    {title: 'Explained variance (%)', margin: {t: 40}, yaxis: {title: '%'}}, {displaylogo: false, responsive: true});
  $('pca_load').innerHTML = res.loadings.slice(0, 2).map((ld, c) => '<b>PC' + (c + 1) + '</b> top m/z: '
    + ld.slice(0, 6).map(e => e.mz.toFixed(1) + ' (' + e.w.toFixed(2) + ')').join(', ')).join('<br>');
}

// ---- tabs -----------------------------------------------------------------
document.querySelectorAll('.tabs button').forEach(b => b.onclick = () => {
  document.querySelectorAll('.tabs button').forEach(x => x.classList.toggle('on', x === b));
  document.querySelectorAll('.pane').forEach(p => p.classList.toggle('on', p.id === 'pane_' + b.dataset.pane));
  if (!window.Plotly) return;
  if (b.dataset.pane === 'spectra') Plotly.Plots.resize('spec');
  if (b.dataset.pane === 'stats') ['pca_scores', 'pca_scree'].forEach(id => { if ($(id) && $(id).data) Plotly.Plots.resize(id); });
});

$('dl_sel').onclick = downloadSelection;
$('pca_add').onclick = pcaAdd;
$('pca_add_pos').onclick = () => pcaAddPol('pos');
$('pca_add_neg').onclick = () => pcaAddPol('neg');
$('pca_clear').onclick = () => { pcaRows = []; renderPcaRows(); };
$('pca_run').onclick = runPCA;

initRoi();
API('/api/samples').then(s => { SAMPLES = s; fillSamples(); renderPcaRows(); });
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="processed directory written by data_load.py")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    store = Store(a.root)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(store))
    print(f"serving {store.root} on http://{a.host}:{a.port}/   (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
