#!/usr/bin/env python3
"""data_load.py - READ step. Only extracts what is stored in the IONTOF files.
Every output file starts with raw_ :

  raw_spectrum.parquet   channel, mass, counts        (ITM/ITA/ITAX)
  raw_summary.json       metadata                     (ITM/ITA/ITAX)
  raw_peaks.parquet      id, name, SN, lmass/cmass/umass   (ITA/ITAX)
  raw_images.npz         summed image per peak        (ITA)
  raw_profiles.parquet   intensity vs scan per peak   (ITAX; ITA if per-scan data exists)
  raw_snapshots.npz      camera images                (ITAX)

Usage: python data_load.py <input_dir> <output_dir>
"""
import json, struct, sys, traceback, warnings, zlib, contextlib
from pathlib import Path
import numpy as np
import pandas as pd
import pySPM
from pySPM.Block import Block, MissingBlockError
from data_probe import probe

warnings.filterwarnings("ignore", message=".*DEPRECATED.*")
EXTS = {".itm", ".ita", ".itax", ".itmx"}


def decode_spectrum(raw, slen):
    """Spectrum blobs differ between software versions: float64/float32, zlib-compressed or not.
    Try every combination that gives exactly slen values."""
    tried = []
    cands = [("raw", raw)]
    try:
        cands.append(("zlib", zlib.decompress(raw)))
    except Exception:
        pass
    for tag, buf in cands:
        for dt in ("<f8", "<f4"):
            need = slen * np.dtype(dt).itemsize
            tried.append(f"{tag}:{len(buf)}B")
            if len(buf) == need:
                return np.frombuffer(buf, dtype=dt).astype(np.float64)
            if need < len(buf) < need + 64:          # small header in front
                return np.frombuffer(buf[-need:], dtype=dt).astype(np.float64)
    raise ValueError(f"cannot decode spectrum: expected {slen} values, blob sizes {tried}")


class _Dummy:
    def get_key_value(self, *a, **k):
        return {"int": None, "float": None, "string": None}


@contextlib.contextmanager
def tolerant_blocks(*fragments):
    """pySPM's ITM constructor hard-fails on fields newer SurfaceLab files no longer store
    (e.g. Registration.Raster.ShotsPerPixel). Return a dummy for those instead."""
    orig = Block.goto

    def goto(self, path, lazy=False):
        try:
            return orig(self, path, lazy)
        except MissingBlockError:
            if any(f in path for f in fragments):
                print(f"  (note: '{path}' missing in file, using None)")
                return _Dummy()
            raise
    Block.goto = goto
    try:
        yield
    finally:
        Block.goto = orig


def ustr(d, key):
    """pySPM returns {'raw':..,'utf16':..,'float':..} dicts for block values."""
    v = d.get(key, {})
    return v.get("utf16", "") if isinstance(v, dict) else str(v)


def ufloat(d, key):
    v = d.get(key, {})
    return v.get("float", np.nan) if isinstance(v, dict) else np.nan


def peak_table(peaks):
    rows = []
    for k, p in peaks.items():
        rows.append(dict(id=k, name=ustr(p, "assign") or ustr(p, "desc"), SN=ustr(p, "SN"),
                         lmass=ufloat(p, "lmass"), cmass=ufloat(p, "cmass"), umass=ufloat(p, "umass")))
    return pd.DataFrame(rows)


def save_spectrum(m, s, ch, out):
    pd.DataFrame({"channel": ch, "mass": m, "counts": s}).to_parquet(out / "raw_spectrum.parquet")
    print(f"  raw_spectrum: {len(m)} pts, m/z {m.min():.2f}-{m.max():.2f}")


def do_itm_ita(path, out):
    is_ita = path.suffix.lower() == ".ita"
    with tolerant_blocks("ShotsPerPixel"):
        obj = (pySPM.ITA if is_ita else pySPM.ITM)(str(path))
    try:
        (out / "raw_summary.json").write_text(json.dumps(obj.get_summary(numeric=True), default=str, indent=2))
    except Exception as e:
        print(f"  summary failed: {e}")
    ch, s = obj.get_spectrum(time=True)
    save_spectrum(obj.channel2mass(ch), s, ch, out)
    if not (is_ita and getattr(obj, "Nimg", 0)):
        return

    pk = peak_table(obj.peaks)
    pk.to_parquet(out / "raw_peaks.parquet")
    imgs = {f"id{i:04d}": obj.get_added_image(i).astype(np.uint32) for i in range(obj.Nimg)}
    np.savez_compressed(out / "raw_images.npz", **imgs)
    print(f"  raw_images: {len(imgs)} peaks, {obj.sx}x{obj.sy}")

    # per-scan totals -> depth-profile candidate (only if the file kept per-scan data)
    try:
        prof = {}
        for _, r in pk.iterrows():
            prof[r["name"] or f"id{r['id']}"] = [
                obj.get_sum_image_by_sn(r["SN"], scans=sc, raw=True).sum() for sc in range(obj.Nscan)]
        pd.DataFrame(prof).rename_axis("scan").to_parquet(out / "raw_profiles.parquet")
        print(f"  raw_profiles: {len(prof)} peaks x {obj.Nscan} scans")
    except Exception as e:
        print(f"  no per-scan data in this file, skipping raw_profiles ({type(e).__name__})")


def do_itax(path, out):
    obj = pySPM.ITAX(str(path))
    r = obj.root
    slen = r.goto("CommonDataObjects/DataViewCollection/*/sizeSpectrum").get_long()
    raw = r.goto("CommonDataObjects/DataViewCollection/*/dataSource/simsDataCache/spectrum/correctedData").value
    s = decode_spectrum(raw, slen)
    ch = 2 * np.arange(slen)
    sf, k0 = obj.get_mass_cal()
    m = pySPM.utils.time2mass(ch, sf, k0)
    save_spectrum(m, s, ch, out)
    (out / "raw_summary.json").write_text(json.dumps(
        {"size": obj.size, "meas_options": obj.meas_options}, default=str, indent=2))

    rows, prof = [], {}
    for x in obj.root.goto("CommonDataObjects/MeasurementOptions/*/massintervals"):
        if x.name != "mi":
            continue
        v = x.dictList()
        name = v["assign"]["utf16"] or v["desc"]["utf16"]
        rows.append(dict(name=name, SN=v["SN"]["utf16"]))
        try:
            prof[name] = np.array(obj.getProfile(name))
        except Exception as e:
            print(f"  profile '{name}' failed: {e}")
    if rows:
        pd.DataFrame(rows).to_parquet(out / "raw_peaks.parquet")
    if prof:
        pd.DataFrame(prof).rename_axis("scan").to_parquet(out / "raw_profiles.parquet")
        print(f"  raw_profiles: {list(prof)}")
    snaps = obj.getSnapshots()
    if snaps:
        np.savez_compressed(out / "raw_snapshots.npz", *snaps)


def main(indir, outdir):
    files = sorted(p for p in Path(indir).iterdir() if p.suffix.lower() in EXTS)
    if not files:
        sys.exit(f"No {sorted(EXTS)} files in {indir}")
    probed = set()
    for p in files:
        ext = p.suffix.lower()
        out = Path(outdir) / p.stem.replace(" ", "_") / ext.lstrip(".")
        out.mkdir(parents=True, exist_ok=True)
        print(f"[{p.name}]")
        # .itmx has no official pySPM reader: try the ITAX-style reader, then the ITM one
        readers = {".itax": [do_itax], ".itmx": [do_itax, do_itm_ita]}.get(ext, [do_itm_ita])
        for n, fn in enumerate(readers):
            try:
                fn(p, out)
                break
            except Exception as e:
                print(f"  {fn.__name__} failed: {type(e).__name__}: {e}")
        else:
            print("  -> no reader worked")
            if ext not in probed:          # one structure dump per extension is enough
                probed.add(ext)
                try:
                    probe(str(p), 4, str(out / "probe.txt"))
                    print(f"  -> block tree written to {out / 'probe.txt'} (send me this)")
                except Exception as e:
                    print(f"  probe failed too: {e}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
