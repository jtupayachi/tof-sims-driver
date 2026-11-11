"""Access to processed IONTOF data (ion events, spectra, images, peaks) written by ``data_load``."""
import json
import math
import threading
from pathlib import Path

import numpy as np

STEP = 8_000_000            # events per pass (bounds memory for wide ranges / ROI scans)
MASS_MAX = 900.0           # default upper m/z of the rebuilt spectrum
BIN = 0.02                 # default spectrum bin width (u)
FINE = 0.01                # u; fine grid the PCA bins are summed from


def polarity_of(name):
    n = name.lower()
    return "pos" if "pos" in n else "neg" if "neg" in n else "other"


USER_PREFIX = "user:"        # key prefix of uploaded samples (they never share a namespace with the reference data)
POLARITIES = ("pos", "neg", "other")
ROI_TAG = "@roi:"            # "6PPD_pos3@roi:10,12,40,50" = the pixels x 10-40, y 12-50 of sample 6PPD_pos3, used as a sample of its own
ALL_EXTS = ("itm", "itax", "ita", "itmx")      # the folders a measurement can have: <sample>/<ext>/raw_*
STORED_EXTS = ("itax", "ita")                  # the formats that keep SurfaceLab's own spectrum (shown in the Spectra tab)


def roi_key(base, box):
    return f"{base}{ROI_TAG}{int(box[0])},{int(box[1])},{int(box[2])},{int(box[3])}"


def split_roi(name):
    """'base@roi:x0,y0,x1,y1' -> (base, (x0, y0, x1, y1)); a plain sample name -> (name, None). Raises ValueError if malformed."""
    i = name.find(ROI_TAG)
    if i < 0:
        return name, None
    parts = name[i + len(ROI_TAG):].split(",")
    if len(parts) != 4:
        raise ValueError(f"bad region of interest in {name!r}")
    x0, y0, x1, y1 = (int(p) for p in parts)
    return name[:i], (x0, y0, x1, y1)


class Store:
    """Two datasets kept strictly apart on disk and in memory:

    * ``root``      - the reference ("ground truth") processed data, never written to by the browser;
    * ``user_root`` - replicates uploaded through the browser (``incoming/`` = original files,
      ``processed/`` = ``data_load`` output). Their sample keys are prefixed with ``user:``.
    """

    def __init__(self, root, user_root=None):
        self.root = Path(root).resolve()
        self.user_root = Path(user_root).resolve() if user_root else None
        if self.user_root and (self.user_root == self.root or self.root in self.user_root.parents
                               or self.user_root in self.root.parents):
            raise ValueError("the uploads directory must be separate from the reference data directory")
        self.lock = threading.Lock()
        self.samples = {}
        self.truepeaks = {}                      # sample -> list (lazy, cached)
        self.veccache = {}                       # (name, bin, mass_max) -> coarse spectrum vector (PCA)
        self.pk_file = self.root / "user_peaks.json"
        self.samples.update(self._scan(self.root, "reference"))
        self.rescan_user()
        print(f"{len(self.samples)} samples with ion events ({sum(1 for s in self.samples.values() if s['source'] == 'user')} uploaded)")

    @staticmethod
    def source_of(name):
        return "user" if name.startswith(USER_PREFIX) else "reference"

    def _scan(self, base, source):
        """Load every sample under `base`; returns ``{key: sample}`` (nothing is registered)."""
        found = {}
        for p in sorted(base.rglob("raw_events_ch.npy")):
            d = p.parent
            try:
                info = json.loads((d / "raw_summary.json").read_text())
                ch = np.load(p, mmap_mode="r")
                name = d.parent.name
                meta = {}
                if source == "user":
                    try:
                        meta = json.loads((d.parent / "meta.json").read_text())
                    except Exception:
                        pass
                    name = USER_PREFIX + name
                pol = meta.get("polarity") if meta.get("polarity") in POLARITIES else polarity_of(name)
                stored = next((d.parent / e for e in STORED_EXTS if (d.parent / e / "raw_spectrum.parquet").is_file()), None)
                found[name] = dict(
                    dir=d, sf=float(info["sf"]), k0=float(info["k0"]),
                    w=int(info["image_width"]), h=int(info["image_height"]),
                    polarity=pol, ch=ch, pix=None, source=source,
                    label=str(meta.get("label", "")), original=meta.get("original", ""),
                    stored=stored, files=[e for e in ALL_EXTS if (d.parent / e).is_dir()])
            except Exception as e:
                print(f"  skip {d}: {type(e).__name__}: {e}")
        return found

    def rescan_user(self):
        """Reload the uploaded samples from disk (after an upload finished or a sample was deleted). The swap is
        atomic, so a concurrent request never sees the uploaded dataset half-empty."""
        found = {}
        if self.user_root and (self.user_root / "processed").is_dir():
            found = self._scan(self.user_root / "processed", "user")
        with self.lock:
            samples = {k: v for k, v in self.samples.items() if v["source"] != "user"}
            samples.update(found)
            self.samples = samples
            for k in [k for k in self.truepeaks if isinstance(k, str) and k.startswith(USER_PREFIX)]:
                del self.truepeaks[k]
            for k in [k for k in self.veccache if k[0].startswith(USER_PREFIX)]:
                del self.veccache[k]

    def user_sample_dir(self, key):
        """``<user_root>/processed/<stem>`` of an uploaded sample key, or None if `key` is not one."""
        if not self.user_root or not key.startswith(USER_PREFIX):
            return None
        d = (self.user_root / "processed" / key[len(USER_PREFIX):]).resolve()
        return d if (self.user_root / "processed").resolve() in d.parents else None

    def partial_user(self):
        """Uploads that have no ion events yet (only an .itax / .ita so far): ``{key: {files, label, polarity, dir}}``.
        They wait for the matching .itm and cannot be used in PCA until it arrives."""
        out = {}
        proc = self.user_root / "processed" if self.user_root else None
        if proc is None or not proc.is_dir():
            return out
        for d in sorted(p for p in proc.iterdir() if p.is_dir()):
            key = USER_PREFIX + d.name
            files = [e for e in ALL_EXTS if (d / e).is_dir()]
            if key in self.samples or not files:
                continue
            try:
                meta = json.loads((d / "meta.json").read_text())
            except Exception:
                meta = {}
            pol = meta.get("polarity") if meta.get("polarity") in POLARITIES else polarity_of(d.name)
            out[key] = dict(files=files, label=str(meta.get("label", "")), polarity=pol, dir=d,
                            stored=any((d / e / "raw_spectrum.parquet").is_file() for e in STORED_EXTS))
        return out

    def delete_user_sample(self, key):
        import shutil
        d = self.user_sample_dir(key)
        if d is None or (key not in self.samples and key not in self.partial_user()):
            raise KeyError(key)
        shutil.rmtree(d, ignore_errors=True)
        inc = (self.user_root / "incoming").resolve()
        for e in ALL_EXTS:                                   # every file uploaded for this sample (.itm, .itax, ...)
            f = (inc / f"{d.name}.{e}").resolve()
            if inc in f.parents and f.is_file():
                f.unlink()
        self.rescan_user()

    def has_stored(self, name):
        s = self.samples.get(name)
        return bool(s and s.get("stored"))

    def stored_spectrum(self, name, width=BIN, mass_max=MASS_MAX):
        """SurfaceLab's own spectrum of the sample (from its .itax / .ita), re-binned like :meth:`spectrum`."""
        s = self.samples[name]
        if not s.get("stored"):
            raise KeyError(name)
        key = (name, "stored")
        with self.lock:
            arrs = self.veccache.get(key)
        if arrs is None:
            import pandas as pd
            df = pd.read_parquet(s["stored"] / "raw_spectrum.parquet", columns=["mass", "counts"])
            m, c = df["mass"].to_numpy(np.float64), df["counts"].to_numpy(np.float64)
            ok = np.isfinite(m) & np.isfinite(c) & (m >= 0)
            arrs = (m[ok], c[ok])
            with self.lock:
                self.veccache[key] = arrs
        m, c = arrs
        nb = int(mass_max / width) + 1
        bi = np.floor(m / width).astype(np.int64)
        ok = (bi >= 0) & (bi < nb) & (c != 0)
        y = np.bincount(bi[ok], weights=c[ok], minlength=nb)[:nb]
        return dict(x0=width / 2, dx=width, y=np.round(y, 3).tolist(), events=int(round(float(c.sum()))), roi=False, source="stored")

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

    def has(self, name):
        """True for a loaded sample, or for a region of interest that lies inside a loaded sample's image."""
        try:
            base, box = split_roi(str(name))
        except ValueError:
            return False
        s = self.samples.get(base)
        if s is None:
            return False
        if box is None:
            return True
        x0, y0, x1, y1 = box
        return 0 <= x0 <= x1 < s["w"] and 0 <= y0 <= y1 < s["h"]

    def _spectrum_of(self, name, width, mass_max):
        base, box = split_roi(name)
        return self.spectrum(base, box, width, mass_max)

    def roi_events(self, base, box):
        """Number of ion events (any mass) whose pixel lies in the inclusive rectangle `box` of sample `base` (cached)."""
        key = (base, "roi_events", tuple(box))
        with self.lock:
            if key in self.veccache:
                return self.veccache[key]
        s = self.samples[base]
        pix = self._pix(s)
        x0, y0, x1, y1 = box
        n = 0
        for a in range(0, len(pix), STEP):
            p = np.asarray(pix[a:a + STEP]).astype(np.int64)
            xs, ys = p & 0xFFFF, p >> 16
            n += int(np.count_nonzero((xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)))
        with self.lock:
            self.veccache[key] = n
        return n

    def fine_vector(self, name):
        """Whole-sample (or region-of-interest) raw-count spectrum on the fine grid (``FINE`` u up to ``MASS_MAX``), cached.
        One pass over the ion events serves every coarser PCA bin width."""
        key = (name, "fine")
        with self.lock:
            if key in self.veccache:
                return self.veccache[key]
        y = np.asarray(self._spectrum_of(name, FINE, MASS_MAX)["y"], np.float64)
        with self.lock:
            self.veccache[key] = y
        return y

    def coarse_vector(self, name, width, mass_max):
        """Whole-sample spectrum on a coarse mass grid, used as a PCA feature vector (cached). Widths that are a
        multiple of the fine grid are summed from it; others are binned directly."""
        k = width / FINE
        if abs(k - round(k)) < 1e-9 and mass_max == MASS_MAX:
            k = int(round(k))
            fine = self.fine_vector(name)
            nb = int(mass_max / width) + 1
            pad = np.zeros(nb * k)
            pad[:min(len(fine), nb * k)] = fine[:nb * k]
            return pad.reshape(nb, k).sum(1)
        key = (name, round(width, 4), round(mass_max, 2))
        with self.lock:
            if key in self.veccache:
                return self.veccache[key]
        y = np.asarray(self._spectrum_of(name, width, mass_max)["y"], np.float64)
        with self.lock:
            self.veccache[key] = y
        return y

    def total_ions(self, name):
        base, box = split_roi(name)
        return int(len(self.samples[base]["ch"])) if box is None else self.roi_events(base, box)

    def peak_counts(self, name, windows):
        """Integrated raw counts over each (lo, hi) m/z window, read straight from the sorted
        ion-event channels (one feature value per window, used as a PCA feature vector). For a region of
        interest only the ions inside its rectangle count."""
        base, box = split_roi(name)
        s = self.samples[base]
        ch, sf, k0 = s["ch"], s["sf"], s["k0"]
        pix = self._pix(s) if box is not None else None
        out = np.zeros(len(windows), np.float64)
        for i, (lo, hi) in enumerate(windows):
            c0 = math.ceil(sf * math.sqrt(max(lo, 0.0)) + k0)
            c1 = math.ceil(sf * math.sqrt(max(hi, 0.0)) + k0)
            a, b = (int(v) for v in np.searchsorted(ch, [c0, c1]))
            if box is None:
                out[i] = max(0, b - a)
                continue
            x0, y0, x1, y1 = box
            for u in range(a, b, STEP):
                p = np.asarray(pix[u:min(u + STEP, b)]).astype(np.int64)
                xs, ys = p & 0xFFFF, p >> 16
                out[i] += np.count_nonzero((xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1))
        return out

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
