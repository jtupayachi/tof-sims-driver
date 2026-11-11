import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from tof_sims_driver.pca import compute_pca
from tof_sims_driver.server import make_handler
from tof_sims_driver.store import Store, roi_key, split_roi


@pytest.fixture
def store(dirs):
    ref, usr = dirs
    return Store(ref, usr)


def test_roi_key_round_trip():
    k = roi_key("user:a_pos1", (1, 2, 3, 4))
    assert k == "user:a_pos1@roi:1,2,3,4" and split_roi(k) == ("user:a_pos1", (1, 2, 3, 4))
    assert split_roi("refA_pos1") == ("refA_pos1", None)
    with pytest.raises(ValueError):
        split_roi("a@roi:1,2,3")


def test_has_checks_the_image_bounds(store):
    assert store.has("refA_pos1") and store.has("refA_pos1@roi:0,0,7,7") and store.has("refA_pos1@roi:2,3,2,3")
    assert not store.has("refA_pos1@roi:0,0,8,7") and not store.has("refA_pos1@roi:5,0,2,7")
    assert not store.has("nope@roi:0,0,1,1") and not store.has("refA_pos1@roi:x")


def test_whole_image_roi_equals_the_sample(store):
    full = "refA_pos1@roi:0,0,7,7"
    assert store.total_ions(full) == store.total_ions("refA_pos1")
    assert np.array_equal(store.coarse_vector(full, 1.0, 900.0), store.coarse_vector("refA_pos1", 1.0, 900.0))
    w = [(28.5, 29.5), (40.5, 41.5)]
    assert np.array_equal(store.peak_counts(full, w), store.peak_counts("refA_pos1", w))


def test_disjoint_rois_add_up_to_the_sample(store):
    left, right = "refA_pos1@roi:0,0,3,7", "refA_pos1@roi:4,0,7,7"
    assert store.total_ions(left) + store.total_ions(right) == store.total_ions("refA_pos1")
    both = store.coarse_vector(left, 1.0, 900.0) + store.coarse_vector(right, 1.0, 900.0)
    assert np.array_equal(both, store.coarse_vector("refA_pos1", 1.0, 900.0))
    w = [(28.5, 29.5), (42.5, 43.5)]
    assert np.array_equal(store.peak_counts(left, w) + store.peak_counts(right, w), store.peak_counts("refA_pos1", w))
    assert 0 < store.total_ions(left) < store.total_ions("refA_pos1")


def test_pca_runs_on_regions_of_interest(store):
    names = ["refA_pos1@roi:0,0,3,7", "refA_pos1@roi:4,0,7,7", "refB_neg1@roi:0,0,3,7", "refB_neg1@roi:4,0,7,7", "refA_pos1"]
    labels = ["A", "A", "B", "B", "A"]
    res = compute_pca(store, names, labels, 3, "tic", True, "all", figures=False)
    assert res["names"] == names and len(res["scores"]) == 5
    peaks = [dict(name="", lo=28.5, hi=29.5), dict(name="", lo=40.5, hi=41.5), dict(name="", lo=42.5, hi=43.5)]
    assert compute_pca(store, names, labels, 2, "tic", True, "mine", peaks=peaks, figures=False)["mode"] == "mine"


def test_pca_endpoint_accepts_regions_and_rejects_bad_ones(store):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(store, None))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/api/pca"

    def post(samples):
        body = json.dumps(dict(samples=[dict(sample=s, label="x") for s in samples], n_components=2, bin=1.0, figures=False)).encode()
        req = urllib.request.Request(url, body, {"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req) as r:
                return r.status
        except urllib.error.HTTPError as e:
            return e.code

    try:
        assert post(["refA_pos1@roi:0,0,3,7", "refA_pos1@roi:4,0,7,7"]) == 200
        assert post(["refA_pos1@roi:0,0,3,7", "refA_pos1@roi:4,0,99,7"]) == 400
        assert post(["refA_pos1@roi:0,0,3,7", "ghost@roi:0,0,1,1"]) == 400
    finally:
        srv.shutdown()
