import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from conftest import make_sample
from tof_sims_driver.server import make_handler
from tof_sims_driver.store import Store
from tof_sims_driver.uploads import UploadManager


@pytest.fixture
def server(dirs):
    ref, usr = dirs
    make_sample(usr / "processed" / "mine_pos1" / "itm", seed=3)
    st = Store(ref, usr)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(st, UploadManager(st)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def call(url, data=None, method=None):
    req = urllib.request.Request(url, data=data, method=method)
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.read(), r.headers
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers


def test_samples_carry_their_source(server):
    st, body, _ = call(server + "/api/samples")
    src = {s["sample"]: s["source"] for s in json.loads(body)}
    assert src == {"refA_pos1": "reference", "refB_neg1": "reference", "user:mine_pos1": "user"}


def test_upload_endpoint_rejects_bad_files(server):
    assert call(server + "/api/upload?filename=x.sh", b"echo")[0] == 400
    st, resp, _ = call(server + "/api/upload?filename=../../x.itm&polarity=bogus", b"abc")
    assert st == 400


def test_only_uploads_can_be_deleted(server):
    assert call(server + "/api/upload?sample=refA_pos1", method="DELETE")[0] == 403
    assert call(server + "/api/upload?sample=user:mine_pos1", method="DELETE")[0] == 200
    assert call(server + "/api/upload?sample=user:mine_pos1", method="DELETE")[0] == 404


def test_config_and_page(server):
    cfg = json.loads(call(server + "/api/config")[1])
    assert cfg["uploads"] is True and ".itm" in cfg["exts"] and cfg["version"]
    st, page, _ = call(server + "/")
    assert st == 200 and b"Selected-peak PCA" in page and b'id="pane_docs"' in page


def test_export_endpoint(server):
    from test_pca import FakeStore
    from tof_sims_driver.pca import compute_pca
    store = FakeStore(n=6, p=40)
    res = compute_pca(store, list(store.vec), ["a"] * 3 + ["b"] * 3, 3, "tic", True, "all")
    st, body, h = call(server + "/api/export", json.dumps(dict(name="t", results=[res])).encode())
    import io
    import zipfile
    assert st == 200 and h["Content-Type"] == "application/zip"
    assert "all_peak_pca/scores.csv" in zipfile.ZipFile(io.BytesIO(body)).namelist()
    assert call(server + "/api/export", b"not json")[0] == 400
    assert call(server + "/api/export", json.dumps(dict(results=[])).encode())[0] == 400
    bad = dict(res, scores=[[1.0]])
    st, body, _ = call(server + "/api/export", json.dumps(dict(results=[bad])).encode())
    assert st == 400 and "cannot export" in json.loads(body)["error"]
