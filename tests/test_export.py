import csv
import io
import json
import zipfile

import numpy as np
import pytest

from conftest import make_sample  # noqa: F401  (shared fixtures)
from test_pca import FakeStore
from tof_sims_driver.export import results_zip, tables, validate_result
from tof_sims_driver.figures import ellipse_params
from tof_sims_driver.pca import compute_pca


@pytest.fixture(scope="module")
def res():
    store = FakeStore(n=8, p=60)
    return compute_pca(store, list(store.vec), ["a"] * 4 + ["b"] * 4, 4, "tic", True, "all", bin_width=1.0)


def read(z, name):
    return list(csv.reader(io.StringIO(z.read(name).decode())))


def test_result_carries_raw_counts(res):
    assert np.asarray(res["counts"]).shape == (8, res["n_features"])
    assert np.allclose(np.asarray(res["counts"]).sum(0), res["peak_counts"])      # the same numbers the filters use
    assert res["normalize"] == "tic" and res["sqrt"] is True


def test_zip_contents(res):
    z = zipfile.ZipFile(io.BytesIO(results_zip([res], "my run")))
    assert z.testzip() is None
    names = z.namelist()
    base = "all_peak_pca/"
    for f in ("counts_raw.csv", "scores.csv", "loadings.csv", "explained_variance.csv", "ellipses_90.csv", "files.csv", "settings.json"):
        assert base + f in names
    assert [n for n in names if n.startswith(base + "figures/")] == [base + f"figures/0{i}_{f['id']}.png" for i, f in enumerate(res["figures"], 1)]
    assert z.read(base + "figures/01_scree.png")[:8] == b"\x89PNG\r\n\x1a\n"
    assert "my run" in z.read("README.txt").decode()
    counts = read(z, base + "counts_raw.csv")
    assert len(counts) == 9 and len(counts[0]) == res["n_features"] + 1 and counts[1][0] == res["names"][0]
    assert [int(v) for v in counts[1][1:]] == res["counts"][0]
    scores = read(z, base + "scores.csv")
    assert scores[0][:3] == ["file", "label", "PC1"] and len(scores) == 9
    ev = read(z, base + "explained_variance.csv")
    assert abs(float(ev[-1][2]) - 100 * sum(res["explained"])) < 1e-9
    load = read(z, base + "loadings.csv")
    assert len(load) == res["n_features"] + 1 and load[0][-1] == "PC4_loading"
    assert json.loads(z.read(base + "settings.json"))["normalize"] == "tic"


def test_scores_can_be_redone_from_the_exported_counts(res):
    z = zipfile.ZipFile(io.BytesIO(results_zip([res])))
    X = np.array([[float(v) for v in r[1:]] for r in read(z, "all_peak_pca/counts_raw.csv")[1:]])
    L = np.array([[float(v) for v in r[-4:]] for r in read(z, "all_peak_pca/loadings.csv")[1:]])        # features x PCs
    S = np.array([[float(v) for v in r[2:]] for r in read(z, "all_peak_pca/scores.csv")[1:]])
    X = np.sqrt(X / X.sum(axis=1, keepdims=True))
    mine = (X - X.mean(0)) @ L
    assert np.allclose(mine, S, atol=1e-4)


def test_ellipse_table_matches_the_figure_maths(res):
    z = zipfile.ZipFile(io.BytesIO(results_zip([res])))
    rows = read(z, "all_peak_pca/ellipses_90.csv")
    assert rows[0][:3] == ["group", "pc_x", "pc_y"]
    groups = {r[0] for r in rows[1:]}
    assert groups <= {"a", "b"} and len(rows) > 1
    S = np.asarray(res["scores"])
    r = next(r for r in rows[1:] if r[0] == "a" and r[1] == "PC1" and r[2] == "PC2")
    e = ellipse_params(S[:4, 0], S[:4, 1])
    assert int(r[3]) == 4 and abs(float(r[6]) - e["semi_major"]) < 1e-9


def test_no_ellipse_rows_without_labels():
    store = FakeStore(n=6, p=40)
    r = compute_pca(store, list(store.vec), [""] * 6, 3, "tic", True, "all", figures=False)
    z = zipfile.ZipFile(io.BytesIO(results_zip([r])))
    assert len(read(z, "all_peak_pca/ellipses_90.csv")) == 1                # header only: no labels, no groups


def test_both_tabs_and_old_sessions_without_counts(res):
    store = FakeStore(n=8, p=60)
    sel = compute_pca(store, list(store.vec), [""] * 8, 3, "tic", True, "mine",
                      peaks=[dict(name="", lo=j, hi=j + 1) for j in (5, 10, 20, 30)], figures=False)
    old = {k: v for k, v in res.items() if k != "counts"}
    z = zipfile.ZipFile(io.BytesIO(results_zip([old, sel])))
    names = z.namelist()
    assert "all_peak_pca/scores.csv" in names and "selected_peak_pca/counts_raw.csv" in names
    assert "all_peak_pca/counts_raw.csv" not in names and "no raw counts" in z.read("README.txt").decode()


def test_spreadsheet_formulas_are_neutralised(res):
    evil = dict(res, names=["=HYPERLINK(\"http://x\")"] + res["names"][1:], labels=["@cmd"] + res["labels"][1:])
    z = zipfile.ZipFile(io.BytesIO(results_zip([evil])))
    rows = read(z, "all_peak_pca/scores.csv")
    assert rows[1][0].startswith("'=") and rows[1][1].startswith("'@")
    assert float(rows[2][2]) == float(rows[2][2])                           # numbers (including negatives) are untouched


@pytest.mark.parametrize("change", [
    {"scores": [[1.0]]}, {"comps": [[1.0]]}, {"mode": "x"}, {"names": []}, {"counts": [[1, 2]]}, {"peak_ratio": ["a"] * 60},
    {"figures": [{"png": "javascript:alert(1)"}]}, {"explained": ["1"]}, {"n_components": 0}])
def test_invalid_results_are_refused(res, change):
    with pytest.raises(ValueError):
        validate_result(dict(res, **change))
    with pytest.raises(ValueError):
        tables(dict(res, **change))


def test_nothing_to_export_and_duplicate_tabs(res):
    with pytest.raises(ValueError):
        results_zip([])
    with pytest.raises(ValueError):
        results_zip([res, res])
