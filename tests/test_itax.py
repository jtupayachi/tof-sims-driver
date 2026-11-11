import io
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pandas as pd
import pytest

from conftest import make_sample
import tof_sims_driver.uploads as U
from tof_sims_driver.server import make_handler
from tof_sims_driver.store import Store
from tof_sims_driver.uploads import UploadManager


def write_itax(out, peaks=((29.04, 100.0), (41.04, 250.0), (43.05, 400.0))):
    """A stored spectrum like do_itax writes: channel, mass, counts."""
    out.mkdir(parents=True, exist_ok=True)
    m = np.array([p[0] for p in peaks] + [900.0])
    c = np.array([p[1] for p in peaks] + [0.0])
    pd.DataFrame({"channel": np.arange(len(m)) * 2, "mass": m, "counts": c}).to_parquet(out / "raw_spectrum.parquet")


def fake_process_file(path, out, keep_events=True, probed=None):
    ext = path.suffix.lower()
    if ext == ".itm":
        make_sample(out, seed=4)
    elif ext == ".itax":
        write_itax(out)
    else:
        return False, "no reader"
    return True, "ok"


@pytest.fixture
def up(dirs, monkeypatch):
    monkeypatch.setattr(U, "process_file", fake_process_file)
    ref, usr = dirs
    st = Store(ref, usr)
    return st, UploadManager(st)


def send(up, name, label="", polarity="auto"):
    job = up.receive(name, io.BytesIO(b"x" * 20), 20, label=label, polarity=polarity)
    for _ in range(200):
        j = [j for j in up.jobs() if j["id"] == job["id"]][0]
        if j["status"] in ("done", "error"):
            return j
        time.sleep(0.05)
    raise AssertionError("conversion did not finish")


def test_reference_sample_with_itax_has_a_stored_spectrum(dirs):
    ref, usr = dirs
    write_itax(ref / "refA_pos1" / "itax")
    st = Store(ref, usr)
    s = st.samples["refA_pos1"]
    assert s["files"] == ["itm", "itax"] and st.has_stored("refA_pos1") and not st.has_stored("refB_neg1")
    sp = st.stored_spectrum("refA_pos1", width=1.0, mass_max=100.0)
    assert sp["source"] == "stored" and sp["dx"] == 1.0 and len(sp["y"]) == 101
    assert sp["y"][29] == 100.0 and sp["y"][41] == 250.0 and sp["y"][43] == 400.0 and sum(sp["y"]) == 750.0
    with pytest.raises(KeyError):
        st.stored_spectrum("refB_neg1")


def test_itm_and_itax_of_one_measurement_become_one_sample(up):
    st, mgr = up
    assert send(mgr, "6PPD_pos1.itm", label="grp")["status"] == "done"
    j = send(mgr, "6PPD_pos1.itax")
    assert j["status"] == "done" and j["sample"] == "user:6PPD_pos1"
    s = st.samples["user:6PPD_pos1"]
    assert s["files"] == ["itm", "itax"] and st.has_stored("user:6PPD_pos1") and s["label"] == "grp"
    assert len([k for k in st.samples if k.startswith("user:")]) == 1
    again = send(mgr, "6PPD_pos1.itm")                       # same extension again = another upload, not the pair
    assert again["sample"] == "user:6PPD_pos1_2"


def test_itax_first_waits_for_its_itm(up):
    st, mgr = up
    j = send(mgr, "Si_neg1.itax", polarity="neg")
    assert j["status"] == "done" and "itm" in j["message"]
    assert "user:Si_neg1" not in st.samples
    part = st.partial_user()
    assert list(part) == ["user:Si_neg1"] and part["user:Si_neg1"]["files"] == ["itax"] and part["user:Si_neg1"]["polarity"] == "neg"
    assert send(mgr, "Si_neg1.itm")["status"] == "done"
    assert st.samples["user:Si_neg1"]["files"] == ["itm", "itax"] and st.samples["user:Si_neg1"]["polarity"] == "neg"
    assert not st.partial_user()


def test_failed_file_keeps_the_other_one(up, monkeypatch):
    st, mgr = up
    send(mgr, "A_pos1.itm")
    monkeypatch.setattr(U, "process_file", lambda *a, **k: (False, "broken"))
    assert send(mgr, "A_pos1.itax")["status"] == "error"
    assert "user:A_pos1" in st.samples and st.samples["user:A_pos1"]["files"] == ["itm"]
    assert not (mgr.root / "incoming" / "A_pos1.itax").exists()


def test_delete_removes_every_file_of_the_sample(up):
    st, mgr = up
    send(mgr, "B_pos1.itm"); send(mgr, "B_pos1.itax"); send(mgr, "C_pos1.itax")
    st.delete_user_sample("user:B_pos1")
    st.delete_user_sample("user:C_pos1")                       # a waiting .itax can be deleted too
    assert not list((mgr.root / "incoming").glob("*")) and not list((mgr.root / "processed").glob("*"))


def test_api_lists_files_partials_and_serves_the_stored_spectrum(up):
    st, mgr = up
    send(mgr, "D_pos1.itm"); send(mgr, "D_pos1.itax"); send(mgr, "E_pos1.itax")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(st, mgr))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    def get(path):
        try:
            with urllib.request.urlopen(base + path) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    try:
        rows = {r["sample"]: r for r in get("/api/samples")[1]}
        assert rows["user:D_pos1"]["files"] == ["itm", "itax"] and rows["user:D_pos1"]["stored"] and not rows["user:D_pos1"].get("partial")
        assert rows["user:E_pos1"]["partial"] and rows["user:E_pos1"]["events"] == 0 and rows["user:E_pos1"]["bytes"] > 0
        assert not rows["refA_pos1"]["stored"] and rows["refA_pos1"]["files"] == ["itm"]
        st_, sp = get("/api/spectrum?sample=user:D_pos1&source=stored&bin=1&mass=100")
        assert st_ == 200 and sp["source"] == "stored" and sp["y"][43] == 400.0
        assert get("/api/spectrum?sample=refA_pos1&source=stored")[0] == 404
        assert get("/api/spectrum?sample=user:D_pos1&bin=1&mass=100")[1].get("source") is None      # default: rebuilt from the ion events
    finally:
        srv.shutdown()
