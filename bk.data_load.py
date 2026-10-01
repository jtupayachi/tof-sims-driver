#!/usr/bin/env python3
"""data_load.py - READ step. Only extracts what is stored in the IONTOF files.
Every output file starts with raw_ :

  raw_spectrum.parquet   channel, mass, counts        (ITM/ITA/ITAX)
  raw_summary.json       metadata                     (ITM/ITA/ITAX)
  raw_peaks.parquet      name, SN, (group), lmass/cmass/umass   (ITA/ITAX/ITM/ITMX)
  raw_images.npz         summed image per peak        (ITA)
  raw_profiles.parquet   intensity vs scan per peak   (ITAX, ITM; ITA if per-scan data exists)
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
    if not is_ita:                     # .itm: no images, but the expert peak list is stored
        try:
            rows = read_mass_intervals(obj.root)
            if rows:
                pd.DataFrame(rows).to_parquet(out / "raw_peaks.parquet")
                print(f"  raw_peaks: {len(rows)} mass intervals")
        except Exception as e:
            print(f"  peak list failed: {type(e).__name__}: {e}")
        return
    if not getattr(obj, "Nimg", 0):
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


def _s(v, k):
    return v.get(k, {}).get("utf16", "").strip("\x00")


def _f(v, k):
    x = v.get(k, {}).get("float")
    return float(x) if x is not None else np.nan


def collect_mi(blk, group=""):
    """Yield every mass interval, recursing into 'mig' groups (where the expert-identified
    peak lists live; the top-level block only holds the 'total' entry)."""
    for x in blk:
        if x.name == "mi":
            v = x.dictList()
            yield dict(name=_s(v, "assign") or _s(v, "desc"), SN=_s(v, "SN"), group=group,
                       lmass=_f(v, "lmass"), cmass=_f(v, "cmass"), umass=_f(v, "umass"))
        elif x.name == "mig":
            yield from collect_mi(x, _s(x.dictList(), "Name"))


MI_PATHS = [
    "CommonDataObjects/MeasurementOptions/*/massintervals",   # .itax
    "CommonDataObjects/MeasurementOptions/massintervals",     # .itmx (no GUID level)
    "Options/massintervals",                                  # .itm
]


def read_mass_intervals(root):
    """Expert-identified peaks, wherever this file flavour stores them."""
    for p in MI_PATHS:
        try:
            blk = root.goto(p)
        except Exception:
            continue
        rows = list(collect_mi(blk))
        if rows:
            print(f"  mass intervals read from '{p}'")
            return rows
    return []


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

    rows, prof = read_mass_intervals(obj.root), {}
    for r in rows:
        name = r["name"]
        try:
            prof[name] = np.array(obj.getProfile(name))
        except Exception as e:
            print(f"  profile '{name}' failed: {e}")
    print(f"  raw_peaks: {len(rows)} mass intervals "
          f"({sum(1 for r in rows if r['group'])} in expert groups)")
    if rows:
        pd.DataFrame(rows).to_parquet(out / "raw_peaks.parquet")
    if prof:
        pd.DataFrame(prof).rename_axis("scan").to_parquet(out / "raw_profiles.parquet")
        print(f"  raw_profiles: {list(prof)}")
    snaps = obj.getSnapshots()
    if snaps:
        np.savez_compressed(out / "raw_snapshots.npz", *snaps)


# ----------------------------------------------------------------------------- .itm (raw events)

@contextlib.contextmanager
def open_root(path):
    """Open an IONTOF block file: skip the 8-byte 'ITStrF01' header, yield the root Block."""
    with open(path, "rb") as f:
        f.read(8)
        yield Block(f)


def _f64(b):
    return struct.unpack("<d", b[:8])[0]


def peak_windows(rows, sf, k0):
    """Expert peaks -> (name, first_channel, last_channel) in raw 50 ps channels.
    Channel <-> mass:  m = ((t - k0) / sf)**2   =>   t = sf*sqrt(m) + k0"""
    out, seen = [], set()
    for r in rows:
        l, u = r["lmass"], r["umass"]
        if not (np.isfinite(l) and np.isfinite(u) and l > 0):
            continue
        name = r["name"] or f"m{r['cmass']:.2f}"
        while name in seen:
            name += "'"
        seen.add(name)
        out.append((name, sf * np.sqrt(l) + k0, sf * np.sqrt(u) + k0))
    return out


def decode_itm_rawdata(root, windows):
    """Read every '  14' block of /rawdata (zlib-compressed uint32 stream, see pySPM.ITM docs):
    a word with any of the top two bits set is a marker (x, y, pixel-id triplet); every other
    word is one detected ion, given as a raw time-of-flight channel (50 ps).
    Returns histogram (2 raw channels per bin, like the .ita/.itax spectra), events per scan,
    events per scan inside each expert peak window, and some statistics."""
    f = root.f
    lst = root.goto("rawdata").get_list()
    hist = np.zeros(1 << 20, np.int64)
    scan_counts = [0]
    prof = {n: [0] for n, _, _ in windows}
    st = dict(chunks=0, events=0, markers=0, bad=0)
    scan, seen6, pend, npend = 0, False, [], 0

    def flush():
        nonlocal hist, pend, npend
        if not pend:
            return
        v = np.concatenate(pend) >> 1
        pend, npend = [], 0
        mx = int(v.max())
        if mx >= hist.size:
            hist = np.concatenate([hist, np.zeros(mx + 1 - hist.size, np.int64)])
        hist += np.bincount(v, minlength=hist.size)

    for n, x in enumerate(lst):
        nm = x["name"].strip()
        if nm == "6":                                # start of a new scan
            if seen6:
                scan += 1
                scan_counts.append(0)
                for k in prof:
                    prof[k].append(0)
            seen6 = True
        elif nm == "14":
            f.seek(x["bidx"])
            child = Block(f)
            try:
                buf = zlib.decompress(child.value)
            except Exception:
                st["bad"] += 1
                continue
            w = np.frombuffer(buf[: len(buf) // 4 * 4], dtype="<u4")
            is_ev = (w & 0xC0000000) == 0
            ev = w[is_ev]
            st["chunks"] += 1
            st["events"] += ev.size
            st["markers"] += int(w.size - ev.size)
            scan_counts[scan] += ev.size
            for name, c0, c1 in windows:
                prof[name][scan] += int(np.count_nonzero((ev >= c0) & (ev < c1)))
            pend.append(ev)
            npend += ev.size
            if npend > 20_000_000:
                flush()
        if n and n % 5000 == 0:
            print(f"    ... {n}/{len(lst)} rawdata blocks")
    flush()
    return hist, scan_counts, prof, st


def do_itm_raw(path, out):
    """.itm has no stored spectrum: rebuild it from the raw ion events (uncorrected counts,
    no dead-time / field-of-view correction)."""
    with open_root(path) as root:
        sf = _f64(root.goto("MassScale/sf").value)
        k0 = _f64(root.goto("MassScale/k0").value)
        rows = read_mass_intervals(root)
        hist, scan_counts, prof, st = decode_itm_rawdata(root, peak_windows(rows, sf, k0))
    n = int(np.flatnonzero(hist)[-1]) + 1
    s = hist[:n].astype(np.float64)
    ch = 2 * np.arange(n)
    m = pySPM.utils.time2mass(ch, sf, k0)
    save_spectrum(m, s, ch, out)
    print(f"  events={st['events']:,} in {st['chunks']} chunks, {st['markers']:,} marker words, "
          f"{len(scan_counts)} scans, {st['bad']} undecodable chunks")
    (out / "raw_summary.json").write_text(json.dumps(
        dict(sf=sf, k0=k0, source="rebuilt from rawdata events (uncorrected)", **st,
             events_per_scan=scan_counts), default=str, indent=2))
    if rows:
        pd.DataFrame(rows).to_parquet(out / "raw_peaks.parquet")
        print(f"  raw_peaks: {len(rows)} mass intervals")
        print("  sanity check (expert centre mass vs. apex of the rebuilt spectrum):")
        for r in rows:
            if np.isfinite(r["lmass"]) and r["umass"] > r["lmass"]:
                sel = (m >= r["lmass"]) & (m <= r["umass"])
                if sel.any():
                    print(f"    {r['name'] or '-':<16} cmass {r['cmass']:8.3f}  "
                          f"apex {m[np.argmax(s * sel)]:8.3f}  counts {s[sel].sum():,.0f}")
    if prof:
        pd.DataFrame(prof).rename_axis("scan").to_parquet(out / "raw_profiles.parquet")
        print(f"  raw_profiles: {list(prof)}")


# ----------------------------------------------------------------------------- .itmx

def do_itmx(path, out):
    """Peaks + metadata only for now: the SIMSData blob is not decoded yet."""
    info = {}
    with open_root(path) as root:
        base = "CommonDataObjects/SIMSDataSet"
        for k in ("sizeX", "sizeY", "sizeZ", "sizeChannel"):
            try:
                info[k] = struct.unpack("<i", root.goto(f"{base}/{k}").value[:4])[0]
            except Exception:
                pass
        for k in ("sf", "k0", "channelwidth"):
            try:
                info[k] = _f64(root.goto(f"{base}/MassScale/{k}").value)
            except Exception:
                pass
        rows = read_mass_intervals(root)
    (out / "raw_summary.json").write_text(json.dumps(info, indent=2))
    if rows:
        pd.DataFrame(rows).to_parquet(out / "raw_peaks.parquet")
        print(f"  raw_peaks: {len(rows)} mass intervals; dims {info}")
    raise NotImplementedError("itmx SIMSData blob not decoded yet - run probe_itmx_simsdata.py")


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
        # .itm: rebuilt from raw events; .itmx: no pySPM reader (peaks only so far)
        readers = {".itax": [do_itax], ".itm": [do_itm_raw], ".itmx": [do_itmx]}.get(ext, [do_itm_ita])
        for n, fn in enumerate(readers):
            try:
                fn(p, out)
                break
            except Exception as e:
                print(f"  {fn.__name__} failed: {type(e).__name__}: {e}")
                if not isinstance(e, NotImplementedError):
                    traceback.print_exc(limit=-4)
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
