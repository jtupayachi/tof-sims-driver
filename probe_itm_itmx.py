#!/usr/bin/env python3
"""probe_itm_itmx.py - print what we need to write readers for .itm / .itmx spectra.

Usage: python probe_itm_itmx.py "<file.itm>" ["<file.itmx>" ...]   > probe_report.txt
Send me probe_report.txt.  Nothing is written besides stdout.
"""
import struct, sys, traceback, warnings, zlib
from itertools import islice
from pathlib import Path
from pySPM.Block import Block

warnings.filterwarnings("ignore")


def i32(b): return struct.unpack("<i", b[:4])[0] if b is not None and len(b) >= 4 else None
def f64(b): return struct.unpack("<d", b[:8])[0] if b is not None and len(b) >= 8 else None
def hexs(b, n=48): return b[:n].hex(" ") if b is not None else None


def get(root, path):
    try:
        return root.goto(path).value
    except Exception as e:
        return f"<missing: {type(e).__name__}>"


def try_zlib(b):
    for wbits, tag in ((15, "zlib"), (-15, "raw-deflate"), (31, "gzip")):
        for off in (0, 4, 8, 12, 16):
            try:
                d = zlib.decompress(b[off:], wbits)
                return f"{tag} OK at offset {off}: {len(d)} bytes, head={hexs(d, 32)}"
            except Exception:
                pass
    return "not zlib/deflate/gzip at offsets 0-16"


def itmx(root):
    base = "CommonDataObjects/SIMSDataSet"
    print("\n### ITMX SIMSDataSet")
    dims = {}
    for k in ("sizeX", "sizeY", "sizeZ", "sizeChannel", "shotsPerPixel", "subphaseIndex",
              "firstScanIndex", "fileVersion0", "fileVersion1"):
        dims[k] = i32(get(root, f"{base}/{k}") if not isinstance(get(root, f"{base}/{k}"), str) else None)
        print(f"{k:15s} = {dims[k]}")
    print("channelWidth    =", f64(get(root, f"{base}/channelWidth")))
    for k in ("sf", "k0", "channelwidth", "polarity", "totalshots", "calibtype", "establ"):
        b = get(root, f"{base}/MassScale/{k}")
        print(f"MassScale/{k:12s} len={len(b) if not isinstance(b, str) else b}  "
              f"f64={f64(b) if not isinstance(b, str) else None}  i32={i32(b) if not isinstance(b, str) else None}")
    for k in ("numAnalysisShots", "numAnalysisCounts", "numPauseCycles", "seDataInfo", "shotDataInfo"):
        b = get(root, f"{base}/{k}")
        if not isinstance(b, str):
            print(f"{k:18s} len={len(b)} hex={hexs(b, 64)}")

    print("\n-- SIMSData blob")
    b = get(root, f"{base}/SIMSData")
    if isinstance(b, str):
        print(b)
        return
    n = len(b)
    print(f"length = {n:,} bytes; head = {hexs(b, 64)}")
    X, Y, Z, C = (dims.get(k) or 0 for k in ("sizeX", "sizeY", "sizeZ", "sizeChannel"))
    print(f"X*Y*Z*C = {X*Y*Z*C:,}  ->  bytes/value if dense = {n / (X*Y*Z*C):.4f}" if X*Y*Z*C else "dims missing")
    for sz in (1, 2, 4, 8):
        print(f"  dense {sz}-byte values would need {X*Y*Z*C*sz:,} bytes")
    print("compression:", try_zlib(b))

    print("\n-- MacroChannelInfo (channel binning table?)")
    m = get(root, f"{base}/MacroChannelInfo")
    if not isinstance(m, str):
        print(f"len={len(m)} head={hexs(m, 96)}")
        for sz, fmt in ((4, "<i"), (8, "<d"), (4, "<f")):
            if len(m) % sz == 0:
                vals = struct.unpack(f"<{min(12, len(m)//sz)}{fmt[1]}", m[:min(12, len(m)//sz) * sz])
                print(f"  first values as {fmt}: {vals}")
    print("\n-- EDRCorrection children")
    try:
        for x in root.goto(f"{base}/EDRCorrection"):
            print("  ", x.name, len(x.value) if x.value is not None else None)
    except Exception as e:
        print("  ", type(e).__name__, e)

    print("\n-- SI Image (top-level Meta) - a total-ion image is a handy sanity check")
    try:
        for x in root.goto("Meta"):
            if x.name == "SI Image":
                for y in x:
                    print("  ", y.name, len(y.value) if y.value is not None else None)
                break
    except Exception as e:
        print("  ", type(e).__name__, e)


def itm(root, path):
    print("\n### ITM MassScale")
    for k in ("sf", "k0", "channelwidth", "polarity", "totalshots", "calibtype", "establ"):
        b = get(root, f"MassScale/{k}")
        print(f"{k:13s} len={len(b) if not isinstance(b, str) else b}  "
              f"f64={f64(b) if not isinstance(b, str) else None}  i32={i32(b) if not isinstance(b, str) else None}")
    print("\n### ITM rawdata: first 40 children (id, size, head, compression)")
    try:
        for x in islice(root.goto("rawdata"), 40):
            v = x.value
            print(f"  id={x.name:>4s}  len={len(v) if v is not None else None:>7}  head={hexs(v, 24)}  "
                  f"{try_zlib(v) if v is not None and len(v) > 100 else ''}")
    except Exception:
        traceback.print_exc()
    print("\n### ITM via pySPM (full error if it fails)")
    try:
        import pySPM
        obj = pySPM.ITM(path)
        ch, s = obj.get_spectrum(time=True)
        print(f"OK: {len(ch)} channels, counts sum={s.sum():.0f}")
    except Exception:
        traceback.print_exc()


for p in sys.argv[1:]:
    print("=" * 100 + f"\n{p}")
    try:
        with open(p, "rb") as f:
            f.read(8)                      # 'ITStrF01' header, same as data_probe.probe
            root = Block(f)
            (itmx if p.lower().endswith("x") else lambda r: itm(r, p))(root)
    except Exception:
        traceback.print_exc()
