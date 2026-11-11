"""Interactive TOF-SIMS browser (HTTP server + single-page app).

  tofsims-serve /path/to/processed                      # http://127.0.0.1:8765
  tofsims-serve /path/to/processed --port 9000 --host 0.0.0.0
  python -m tof_sims_driver.server /path/to/processed    # same, without the console script

Reference data (ROOT) and uploaded replicates (--uploads, default ROOT_uploads) are kept in separate
directories on disk (uploads can be deleted, ROOT never is) but listed together. Open http://HOST:PORT/ and pick Positive or Negative. Needs raw_events_ch.npy /
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
  GET  /api/samples                  [{sample, polarity, w, h, events, source: reference|user, label, bytes, files, stored, partial?}]
                                     files = the formats the sample has (itm, itax, ...); stored = it has an .itax/.ita spectrum;
                                     partial = uploaded .itax still waiting for its .itm (events 0, not usable in PCA)
  GET  /api/config                   {uploads, max_upload_mb, exts, version}
  POST /api/upload?filename=F&label=L&polarity=auto|pos|neg|other   raw file body -> queued job
  GET  /api/upload/status            conversion jobs of uploaded files
  DELETE /api/upload?sample=user:S   remove an uploaded sample (reference data cannot be deleted)
  GET  /api/truepeaks?sample=S       expert peaks  [{name, cmass, lo, hi, group}]
  GET  /api/spectrum?sample=S&...    raw-count spectrum {x0, dx, y[], events, roi}
                                     optional box x0,y0,x1,y1 (pixels) -> ROI spectrum
                                     optional bin (u, default 0.02), mass (max u, default 900)
                                     optional source=stored -> SurfaceLab's own spectrum from the sample's .itax / .ita
  GET  /api/image?sample=S&lo=&hi=   2D image, lo <= m/z < hi   (uint32 LE, headers X-W, X-H)
  GET  /api/image?sample=S&all=1     2D image of every ion
  GET  /api/peaks                    user peaks  [{id, name, lo, hi}]
  POST /api/peaks                    {name, lo, hi} -> create
  POST /api/pca                      {samples:[{sample,label}], n_components, normalize, sqrt, mode, bin|peaks,
                                       filter_peaks, ratio_min, frac_min, ref_labels}
                                     a sample may be a region of interest: "NAME@roi:x0,y0,x1,y1" (pixels, inclusive)
                                     -> scores, loadings, steps (with formulas), figures
  POST /api/pca/tune                 {samples:[{sample,label}], refs:[labelA,labelB], normalize, sqrt, goal: closest|separate}
                                     grid search of bin / NBR / TIC on the files of the two labels; closest = the setting and PC
                                     where their centroids are nearest (with a shuffled-label p), separate = best leave-one-out
                                     nearest-centroid accuracy in PC1-PC2
                                     -> {best, results, baseline, pcs, loadings, figure, p_corr}
  GET  /api/pca/steps?mode=all|mine  preprocessing / peak-selection steps with formulas (no data)
  POST /api/export                   {name, results:[<PCA result>, ...]} -> zip of CSV tables (raw counts, scores, loadings, ...) + figures
  DELETE /api/peaks?id=N             remove
"""
import argparse
import json
import logging
import math
import mimetypes
import traceback
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from urllib.parse import parse_qs, unquote, urlparse

from .pca import FRAC_MIN, N_LABELS, RATIO_MIN, compute_pca, pca_steps, selection_steps
from .store import BIN, MASS_MAX, Store
from .tune import tune_filter
from . import __version__
from .data_load import EXTS
from .export import results_zip
from .uploads import UploadError, UploadManager

LOG = logging.getLogger("sims")

MAX_EXPORT_BYTES = 400 << 20            # largest accepted export request
STATIC_OK = {".html", ".png", ".json", ".css", ".js", ".svg", ".txt"}


def app_html():
    """The single-page app (package data ``static/app.html``)."""
    return resources.files(__package__).joinpath("static/app.html").read_text(encoding="utf-8")



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

def _dir_bytes(d):
    return sum(f.stat().st_size for f in Path(d).rglob("*") if f.is_file())


def make_handler(store, uploader=None):
    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            LOG.info("%s - %s", self.address_string(), fmt % a)

        def handle_one_request(self):
            try:
                super().handle_one_request()
            except Exception:
                LOG.error("connection error:\n%s", traceback.format_exc())
                raise

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
                    return self._send(200, app_html(), "text/html", {"Cache-Control": "no-store"})
                if u.path == "/plotly.js":
                    return self._send(200, plotly_js(), "application/javascript")
                if u.path == "/api/samples":
                    rows = [dict(sample=n, polarity=s["polarity"], w=s["w"], h=s["h"],
                                 events=int(len(s["ch"])), source=s["source"], label=s["label"],
                                 bytes=_dir_bytes(s["dir"]), files=s["files"], stored=bool(s["stored"]))
                            for n, s in list(store.samples.items())]
                    rows += [dict(sample=n, polarity=p["polarity"], w=0, h=0, events=0, source="user", label=p["label"],
                                  bytes=_dir_bytes(p["dir"]), files=p["files"], stored=p["stored"], partial=True)
                             for n, p in store.partial_user().items()]       # uploaded .itax waiting for its .itm
                    return self._json(rows)
                if u.path == "/api/config":
                    return self._json(dict(uploads=uploader is not None,
                                           max_upload_mb=(uploader.max_bytes >> 20) if uploader else 0,
                                           exts=sorted(EXTS), version=__version__))
                if u.path == "/api/upload/status":
                    return self._json(uploader.jobs() if uploader else [])
                if u.path == "/api/truepeaks":
                    name = q.get("sample", "")
                    if name not in store.samples:
                        return self._json({"error": "unknown sample"}, 404)
                    return self._json(store.true_peaks(name))
                if u.path == "/api/peaks":
                    return self._json(store.peaks())
                if u.path == "/api/pca/steps":
                    mode = "mine" if q.get("mode", "all").startswith("mine") else "all"
                    return self._json(dict(steps=pca_steps(mode),
                                           selection_steps=selection_steps() if mode == "mine" else []))
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
                    if q.get("source") == "stored" and box is None:       # SurfaceLab's own spectrum from the .itax / .ita
                        if not store.has_stored(name):
                            return self._json({"error": "this sample has no stored spectrum (.itax)"}, 404)
                        return self._json(store.stored_spectrum(name, width, mass_max))
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
                LOG.warning("400 %s %s: %s", self.command, self.path, e)
                self._json({"error": f"bad request: {e}"}, 400)
            except Exception as e:
                LOG.error("500 %s %s:\n%s", self.command, self.path, traceback.format_exc())
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
            if path == "/api/pca/tune":
                return self._do_tune()
            if path == "/api/upload":
                return self._do_upload()
            if path == "/api/export":
                return self._do_export()
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

        def _do_upload(self):
            if uploader is None:
                return self._json({"error": "uploads are disabled on this server"}, 403)
            q = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
            try:
                length = int(self.headers.get("Content-Length", 0))
                job = uploader.receive(q.get("filename", ""), self.rfile, length, q.get("label", ""),
                                       q.get("polarity", "auto"))
            except UploadError as e:
                return self._json({"error": str(e)}, 400)
            except Exception as e:
                LOG.error("500 /api/upload:\n%s", traceback.format_exc())
                return self._json({"error": f"{type(e).__name__}: {e}"}, 500)
            LOG.info("uploaded %s (%d bytes) -> %s", job["filename"], job["size"], job["sample"])
            self._json(job, 202)

        def _do_export(self):
            try:
                length = int(self.headers.get("Content-Length", 0))
                if length > MAX_EXPORT_BYTES:
                    return self._json({"error": "the results are too large to export"}, 413)
                body = json.loads(self.rfile.read(length) or b"{}")
                results = body.get("results")
                if not isinstance(results, list) or not 1 <= len(results) <= 2:
                    return self._json({"error": "send one or two PCA results"}, 400)
                data = results_zip(results, str(body.get("name", "")))
            except (ValueError, KeyError, TypeError) as e:
                return self._json({"error": f"cannot export: {e}"}, 400)
            except Exception as e:
                LOG.error("500 /api/export:\n%s", traceback.format_exc())
                return self._json({"error": f"{type(e).__name__}: {e}"}, 500)
            self._send(200, data, "application/zip", {"Content-Disposition": 'attachment; filename="tofsims-data.zip"'})

        def _do_tune(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                items = body.get("samples", [])
                names = [str(it["sample"]) for it in items]
                labels = [str(it.get("label", "")) for it in items]
                refs = [str(r) for r in body.get("refs", [])]
                normalize = str(body.get("normalize", "tic"))
                sqrt = bool(body.get("sqrt", True))
                for n in names:
                    if not store.has(n):
                        return self._json({"error": f"unknown sample or region {n}"}, 400)
                    if store.total_ions(n) < 1:
                        return self._json({"error": f"{n} has no ions"}, 400)
            except Exception as e:
                return self._json({"error": f"bad request: {e}"}, 400)
            try:
                self._json(tune_filter(store, names, labels, refs, normalize, sqrt, goal=str(body.get("goal", "closest"))))
            except ValueError as e:
                self._json({"error": str(e)}, 400)
            except Exception as e:
                LOG.error("500 /api/pca/tune:\n%s", traceback.format_exc())
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        def _do_pca(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                items = body.get("samples", [])
                names = [str(it["sample"]) for it in items]
                labels = [str(it.get("label", "")) for it in items]
                if len(names) < 2:
                    return self._json({"error": "add at least 2 subsamples"}, 400)
                for n in names:
                    if not store.has(n):
                        return self._json({"error": f"unknown sample or region {n}"}, 400)
                    if store.total_ions(n) < 1:
                        return self._json({"error": f"{n} has no ions"}, 400)
                ncomp = int(body.get("n_components", 2))
                normalize = str(body.get("normalize", "tic"))
                sqrt = bool(body.get("sqrt", True))
                mode = str(body.get("mode", "all"))
                bin_width = float(body.get("bin", 1.0))
                mass_max = float(body.get("mass", MASS_MAX))
                peaks = body.get("peaks") or None
                ratio_min = float(body.get("ratio_min", RATIO_MIN))
                frac_min = float(body.get("frac_min", FRAC_MIN))
                n_labels = int(body.get("n_labels", N_LABELS))
                filter_peaks = bool(body.get("filter_peaks", False))
                ref_labels = [str(r) for r in (body.get("ref_labels") or [])] or None
                if not (math.isfinite(ratio_min) and ratio_min >= 0 and 0 <= frac_min <= 1 and 0 <= n_labels <= 60):
                    return self._json({"error": "need ratio_min >= 0, 0 <= frac_min <= 1, 0 <= n_labels <= 60"}, 400)
            except Exception as e:
                return self._json({"error": f"bad request: {e}"}, 400)
            try:
                self._json(compute_pca(store, names, labels, ncomp, normalize, sqrt, mode,
                                       bin_width, mass_max, peaks, ratio_min=ratio_min, frac_min=frac_min,
                                       n_labels=n_labels, filter_peaks=filter_peaks, ref_labels=ref_labels))
            except Exception as e:
                LOG.error("500 /api/pca:\n%s", traceback.format_exc())
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        def do_DELETE(self):
            u = urlparse(self.path)
            if u.path == "/api/upload":
                key = parse_qs(u.query).get("sample", [""])[0]
                if uploader is None or not key.startswith("user:"):
                    return self._json({"error": "only uploaded samples can be deleted"}, 403)
                try:
                    store.delete_user_sample(key)
                except KeyError:
                    return self._json({"error": "unknown sample"}, 404)
                return self._json({"deleted": key})
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


def main(argv=None):
    ap = argparse.ArgumentParser(description="Interactive TOF-SIMS browser")
    ap.add_argument("root", help="processed directory written by data_load.py")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--log", default="sims_server.log", help="log file (default: <root>/sims_server.log)")
    ap.add_argument("--uploads", default=None,
                    help="directory for replicates uploaded through the browser; must be outside ROOT "
                         "(default: <ROOT>_uploads next to ROOT)")
    ap.add_argument("--no-upload", action="store_true", help="disable uploads (read-only server)")
    ap.add_argument("--max-upload-gb", type=float, default=8.0, help="largest accepted upload (default 8)")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    user_root = None if a.no_upload else Path(a.uploads or root.parent / (root.name + "_uploads")).resolve()
    store = Store(root, user_root)
    uploader = UploadManager(store, int(a.max_upload_gb * (1 << 30))) if user_root else None
    logfile = a.log if "/" in a.log else str(store.root / a.log)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(logfile), logging.StreamHandler()])
    LOG.info("serving %s on http://%s:%s/  (log: %s)", store.root, a.host, a.port, logfile)
    if uploader:
        LOG.info("uploads enabled -> %s (anyone who can reach this port can upload; use --no-upload to disable)",
                 store.user_root)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(store, uploader))
    print(f"serving {store.root} on http://{a.host}:{a.port}/   (Ctrl-C to stop)   log: {logfile}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

