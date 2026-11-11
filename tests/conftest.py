import json
import math

import numpy as np
import pytest


def make_sample(directory, sf=66000.0, k0=-1100.0, w=8, h=8, peaks=((29.0391, 800), (41.0391, 1200), (43.0548, 2000)),
                seed=0):
    """Write a tiny synthetic ``raw_events_*`` sample (sorted ion channels) like ``data_load`` would."""
    rng = np.random.default_rng(seed)
    ch = []
    for m, n in peaks:
        t = sf * math.sqrt(m) + k0
        ch.append(np.round(rng.normal(t, 0.4, n)).astype(np.uint32))
    ch = np.sort(np.concatenate(ch))
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "raw_events_ch.npy", ch)
    np.save(directory / "raw_events_pix.npy", ((rng.integers(0, h, len(ch)) << 16) | rng.integers(0, w, len(ch))).astype(np.uint32))
    (directory / "raw_summary.json").write_text(json.dumps(dict(sf=sf, k0=k0, image_width=w, image_height=h)))
    return directory


@pytest.fixture
def dirs(tmp_path):
    ref, usr = tmp_path / "processed", tmp_path / "processed_uploads"
    make_sample(ref / "refA_pos1" / "itm", seed=1)
    make_sample(ref / "refB_neg1" / "itm", seed=2)
    return ref, usr
