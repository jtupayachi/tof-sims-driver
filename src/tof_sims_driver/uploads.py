"""Replicates uploaded through the browser: stored apart from the reference data and processed in the background.

Layout under the uploads directory (``Store.user_root``)::

    incoming/<stem>.<ext>                the file as uploaded
    processed/<stem>/meta.json           label, polarity, original file name
    processed/<stem>/<ext>/raw_*         ``data_load`` output (same layout as the reference data)

Files of one measurement that share a name but differ in extension belong to one sample: the ``.itm`` gives the ion
events (PCA, 2D images, spectrum rebuilt from the events) and the ``.itax`` gives SurfaceLab's own stored spectrum
(Spectra tab). They can be uploaded in either order, so ``X.itm`` and ``X.itax`` end up in ``processed/X/itm`` and
``processed/X/itax``. A second file with the *same* extension and name is a different upload and gets ``X_2``.
"""
import json
import re
import threading
import time
from pathlib import Path

from .data_load import EXTS, process_file
from .store import POLARITIES, USER_PREFIX, polarity_of

CHUNK = 1 << 20


class UploadError(ValueError):
    pass


def safe_stem(filename):
    """File name without directories / extension, reduced to ``[A-Za-z0-9._-]``."""
    base = Path(str(filename).replace("\\", "/")).name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(base).stem).strip("._")[:80]
    if not stem:
        raise UploadError("file name has no usable characters")
    return stem


class UploadManager:
    def __init__(self, store, max_bytes=8 << 30):
        if store.user_root is None:
            raise ValueError("uploads are disabled (no uploads directory)")
        self.store = store
        self.max_bytes = int(max_bytes)
        self.root = store.user_root
        self._jobs = []
        self._lock = threading.Lock()
        self._work = threading.Lock()            # one conversion at a time (pySPM needs a lot of memory)
        self._next = 1

    # -- bookkeeping ---------------------------------------------------------------------------
    def jobs(self):
        with self._lock:
            return [dict(j) for j in self._jobs][-50:]

    def _new_job(self, filename, size):
        with self._lock:
            job = dict(id=self._next, filename=filename, size=size, status="uploading", message="",
                       sample="", started=time.time(), finished=None)
            self._next += 1
            self._jobs.append(job)
            return job

    def _set(self, job, **kw):
        with self._lock:
            job.update(kw)

    # -- receiving -----------------------------------------------------------------------------
    def receive(self, filename, rfile, length, label="", polarity="auto"):
        """Stream `length` bytes from `rfile` to ``incoming/`` and queue the conversion. Returns the job."""
        ext = Path(filename).suffix.lower()
        if ext not in EXTS:
            raise UploadError(f"unsupported file type {ext or '(none)'}; use {', '.join(sorted(EXTS))}")
        if not 0 < length <= self.max_bytes:
            raise UploadError(f"size must be between 1 byte and {self.max_bytes >> 20} MB")
        stem = safe_stem(filename)
        if polarity not in POLARITIES + ("auto",):
            raise UploadError("polarity must be auto, pos, neg or other")
        (self.root / "incoming").mkdir(parents=True, exist_ok=True)
        (self.root / "processed").mkdir(parents=True, exist_ok=True)
        with self._lock:                         # reserve a unique stem, or join the other file of the same measurement
            attach = False
            sub = ext.lstrip(".")
            d0 = self.root / "processed" / stem
            if d0.is_dir() and not (d0 / sub).exists() and not (self.root / "incoming" / (stem + ext)).exists():
                attach = True
                (d0 / sub).mkdir()
            else:
                base, k = stem, 1
                while (self.root / "processed" / stem).exists() or (self.root / "incoming" / (stem + ext)).exists():
                    k += 1
                    stem = f"{base}_{k}"
                (self.root / "processed" / stem).mkdir()
        dest = self.root / "incoming" / (stem + ext)
        part = dest.with_suffix(ext + ".part")
        job = self._new_job(stem + ext, length)
        job["sample"] = USER_PREFIX + stem
        try:
            left = length
            with open(part, "wb") as f:
                while left:
                    buf = rfile.read(min(CHUNK, left))
                    if not buf:
                        raise UploadError("connection closed before the whole file arrived")
                    f.write(buf)
                    left -= len(buf)
            part.replace(dest)
        except Exception as e:
            part.unlink(missing_ok=True)
            self._cleanup(stem, dest)
            self._set(job, status="error", message=str(e), finished=time.time())
            raise
        pol = polarity_of(stem) if polarity == "auto" else polarity
        meta_file = self.root / "processed" / stem / "meta.json"
        meta = dict(label=str(label).strip()[:80], polarity=pol, original=dest.name, uploaded=time.time())
        if attach and meta_file.is_file():       # keep what the first file said unless this one says something new
            try:
                old = json.loads(meta_file.read_text())
                meta.update(old)
                if str(label).strip():
                    meta["label"] = str(label).strip()[:80]
                if polarity != "auto":
                    meta["polarity"] = polarity
            except Exception:
                pass
        meta_file.write_text(json.dumps(meta, indent=1))
        self._set(job, status="queued")
        threading.Thread(target=self._run, args=(job, stem, dest), daemon=True).start()
        return dict(job)

    def _cleanup(self, stem, dest):
        """Remove this file and its converted output; the sample folder goes too once no other file of it is left."""
        import shutil
        d = self.root / "processed" / stem
        shutil.rmtree(d / Path(dest).suffix.lstrip(".").lower(), ignore_errors=True)
        Path(dest).unlink(missing_ok=True)
        if d.is_dir() and not any(p.is_dir() for p in d.iterdir()):
            shutil.rmtree(d, ignore_errors=True)

    # -- converting ----------------------------------------------------------------------------
    def _run(self, job, stem, src):
        with self._work:
            self._set(job, status="processing")
            out = self.root / "processed" / stem / src.suffix.lstrip(".").lower()
            try:
                ok, msg = process_file(src, out, keep_events=True)
            except Exception as e:               # a reader must never take the server down
                ok, msg = False, f"{type(e).__name__}: {e}"
            if not ok:
                self._cleanup(stem, src)
                self._set(job, status="error", message=msg[:400], finished=time.time())
                return
            self.store.rescan_user()
            if job["sample"] in self.store.samples:
                self._set(job, status="done", message="ready", finished=time.time())
            elif (out / "raw_spectrum.parquet").is_file():       # an .itax / .ita: kept, waits for the .itm of the same name
                self._set(job, status="done", finished=time.time(),
                          message="spectrum stored; upload the .itm with the same file name to use this sample in PCA and 2D")
            else:
                self._cleanup(stem, src)
                self._set(job, status="error", finished=time.time(),
                          message="the file holds no ion events and no spectrum, so it cannot be used; nothing was kept")
