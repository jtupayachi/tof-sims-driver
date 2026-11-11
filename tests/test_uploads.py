import io
import json
import time

import pytest

from conftest import make_sample
from tof_sims_driver.store import Store
from tof_sims_driver.uploads import UploadError, UploadManager, safe_stem


def test_datasets_are_separate(dirs):
    ref, usr = dirs
    make_sample(usr / "processed" / "mine_pos1" / "itm", seed=3)
    (usr / "processed" / "mine_pos1" / "meta.json").write_text(json.dumps(dict(label="grp", polarity="neg", original="mine_pos1.itm")))
    st = Store(ref, usr)
    assert set(st.samples) == {"refA_pos1", "refB_neg1", "user:mine_pos1"}
    assert st.samples["user:mine_pos1"]["source"] == "user" and st.samples["refA_pos1"]["source"] == "reference"
    assert st.samples["user:mine_pos1"]["polarity"] == "neg" and st.samples["user:mine_pos1"]["label"] == "grp"
    assert st.source_of("user:mine_pos1") == "user"


def test_uploads_dir_must_not_overlap_reference(dirs):
    ref, _ = dirs
    with pytest.raises(ValueError):
        Store(ref, ref / "uploads")
    with pytest.raises(ValueError):
        Store(ref, ref)


def test_delete_only_touches_uploads(dirs):
    ref, usr = dirs
    make_sample(usr / "processed" / "mine_pos1" / "itm")
    (usr / "incoming").mkdir()
    (usr / "incoming" / "mine_pos1.itm").write_bytes(b"x")
    (usr / "processed" / "mine_pos1" / "meta.json").write_text(json.dumps(dict(original="mine_pos1.itm")))
    st = Store(ref, usr)
    with pytest.raises(KeyError):
        st.delete_user_sample("refA_pos1")                 # reference data cannot be deleted
    assert (ref / "refA_pos1").exists()
    st.delete_user_sample("user:mine_pos1")
    assert "user:mine_pos1" not in st.samples and not (usr / "processed" / "mine_pos1").exists()
    assert not (usr / "incoming" / "mine_pos1.itm").exists() and len(st.samples) == 2


def test_safe_stem():
    assert safe_stem("../../etc/pass wd.itm") == "pass_wd"
    assert safe_stem("C:\\x\\My File (1).itm") == "My_File_1"
    with pytest.raises(UploadError):
        safe_stem("....itm")


def test_receive_validation_and_failed_conversion(dirs):
    ref, usr = dirs
    st = Store(ref, usr)
    up = UploadManager(st, max_bytes=1000)
    for name, size, pol in [("a.exe", 5, "auto"), ("a.itm", 0, "auto"), ("a.itm", 5000, "auto"), ("a.itm", 5, "weird")]:
        with pytest.raises(UploadError):
            up.receive(name, io.BytesIO(b"x" * size), size, polarity=pol)
    with pytest.raises(UploadError):                       # body shorter than announced
        up.receive("short.itm", io.BytesIO(b"abc"), 10)
    assert not list((usr / "incoming").glob("*"))           # nothing left behind
    job = up.receive("junk.itm", io.BytesIO(b"not an itm file"), 15, label="x")
    assert job["sample"] == "user:junk"
    for _ in range(100):
        j = [j for j in up.jobs() if j["id"] == job["id"]][0]
        if j["status"] in ("done", "error"):
            break
        time.sleep(0.1)
    assert j["status"] == "error" and j["message"]
    assert not (usr / "processed" / "junk").exists() and not (usr / "incoming" / "junk.itm").exists()
    assert set(st.samples) == {"refA_pos1", "refB_neg1"}


def test_pca_runs_across_reference_and_uploaded_samples(dirs):
    from tof_sims_driver.pca import compute_pca
    ref, usr = dirs
    make_sample(ref / "refC_pos2" / "itm", seed=5, peaks=((29.0391, 300), (41.0391, 1500), (43.0548, 900)))
    make_sample(usr / "processed" / "mine_pos1" / "itm", seed=3, peaks=((29.0391, 900), (41.0391, 400), (43.0548, 1800)))
    st = Store(ref, usr)
    res = compute_pca(st, ["refA_pos1", "refC_pos2", "user:mine_pos1"], ["a", "b", "c"], 2, "tic", True, "all",
                      bin_width=1.0, figures=False)
    assert res["names"] == ["refA_pos1", "refC_pos2", "user:mine_pos1"] and res["n_components"] == 2
