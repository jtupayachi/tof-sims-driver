import numpy as np
import pytest

from tof_sims_driver.pca import compute_pca, pca_steps, selection_steps


class FakeStore:
    """Duck-typed stand-in for Store: two groups of subsamples with different peak patterns."""

    def __init__(self, n=8, p=60, seed=0):
        rng = np.random.default_rng(seed)
        base = rng.uniform(50, 500, p)
        self.vec = {}
        for i in range(n):
            v = base.copy()
            v[10 if i < n // 2 else 30] *= 6            # group marker peak
            self.vec[f"s{i}"] = rng.poisson(v).astype(float)

    def coarse_vector(self, name, width, mass_max):
        return self.vec[name]

    def peak_counts(self, name, windows):
        v = self.vec[name]
        return np.array([v[int(lo):int(hi)].sum() for lo, hi in windows], float)

    def total_ions(self, name):
        return float(self.vec[name].sum())


@pytest.fixture
def store():
    return FakeStore()


def test_all_peak_pca_is_mean_centered_and_reports_steps(store):
    names = list(store.vec)
    res = compute_pca(store, names, ["a"] * 4 + ["b"] * 4, 4, "tic", True, "all", bin_width=1.0)
    assert res["mean_centered"] and res["center_check"] < 1e-9
    titles = [s["title"] for s in res["steps"]]
    assert any("Mean centring" in t for t in titles)
    assert all(s["formula"] for s in res["steps"])
    assert len(res["peak_lo"]) == len(res["peak_mz"]) == len(res["peak_counts"]) == res["n_features"]
    # the two groups separate on PC1
    pc1 = np.asarray(res["scores"])[:, 0]
    assert np.sign(pc1[:4]).std() == 0 and np.sign(pc1[4:]).std() == 0 and np.sign(pc1[0]) != np.sign(pc1[4])
    assert not [k for k in res if "cluster" in k] and not any("k-means" in t for t in titles)   # no automatic grouping


def test_figures(store):
    names = list(store.vec)
    res = compute_pca(store, names, [""] * 8, 4, "tic", True, "all")
    ids = [f["id"] for f in res["figures"]]
    assert ids == ["scree", "scores", "pcloadings", "drivers"]
    assert "loadquad" not in ids and "loadings" not in ids and "labelbiplot" not in ids and "corr" not in ids
    assert all(f["png"].startswith("data:image/png;base64,") and f["formula"] for f in res["figures"])
    assert res["figures"][1]["caption"].startswith("Figure 2.")
    assert all(len(f["title"]) <= 32 for f in res["figures"])                            # short titles
    assert "90% confidence ellipse" in res["figures"][1]["caption"] and "k-means" not in res["figures"][1]["caption"]
    assert "No labels were given" in res["figures"][1]["caption"]                 # no labels: no group, so no ellipse around everything


def test_figures_per_label_loadings(store):
    names = list(store.vec)
    res = compute_pca(store, names, ["a"] * 4 + ["b"] * 4, 4, "tic", True, "all")
    ids = [f["id"] for f in res["figures"]]
    assert ids == ["scree", "scores", "pcloadings", "drivers", "labelpc", "labelload"]   # all PCs and all labels in each figure
    assert all(len(f["title"]) <= 32 for f in res["figures"])
    one = compute_pca(store, names, ["a"] * 8, 4, "tic", True, "all")                # one label = every file: nothing to contrast
    assert [f["id"] for f in one["figures"]] == ["scree", "scores", "pcloadings", "drivers"]
    sel = compute_pca(store, names, ["a"] * 4 + ["b"] * 4, 3, "tic", True, "mine",
                      peaks=[dict(name=f"P{j}", lo=j, hi=j + 1) for j in (5, 10, 20, 30, 40)])
    assert [f["id"] for f in sel["figures"]] == ids


def test_label_loadings_reconstruct_group_mean(store):
    from tof_sims_driver.figures import label_loadings
    names = list(store.vec)
    res = compute_pca(store, names, ["a"] * 4 + ["b"] * 4, 7, "tic", True, "all", figures=False)
    LL = label_loadings(res)
    assert LL["groups"] == ["a", "b"] and LL["n"] == [4, 4]
    assert np.allclose(LL["share"].sum(1), 1) and np.allclose(LL["d"], LL["parts"].sum(1))
    # with all PCs kept, d_g is the group's mean deviation from the grand mean of the processed (centred) matrix
    X = np.vstack([store.vec[n] for n in names])
    X = np.sqrt(X / X.sum(1, keepdims=True))
    keep = X.std(0) > 0
    Z = X[:, keep] - X[:, keep].mean(0)
    assert np.allclose(LL["d"][0], Z[:4].mean(0), atol=1e-8)
    assert np.sign(LL["tbar"][0][0]) != np.sign(LL["tbar"][1][0])                    # the groups sit on opposite sides of PC1


def _scores_caption(res):
    return next(f["caption"] for f in res["figures"] if f["id"] == "scores")


def test_figure_3_ellipses_follow_the_labels(store):
    names = list(store.vec)
    ok = _scores_caption(compute_pca(store, names, ["a"] * 4 + ["b"] * 4, 3, "tic", True, "all"))
    assert "No labels" not in ok and "no ellipse is drawn" not in ok
    partly = _scores_caption(compute_pca(store, names, ["a"] * 4 + [""] * 4, 3, "tic", True, "all"))
    assert "Files without a label have no ellipse" in partly
    tiny = _scores_caption(compute_pca(store, names, [f"g{i}" for i in range(8)], 3, "tic", True, "all"))
    assert "No labelled group has 3 or more files" in tiny


def test_selected_peak_pca(store):
    names = list(store.vec)
    peaks = [dict(name=f"P{j}", lo=j, hi=j + 1) for j in (5, 10, 20, 30, 40)]
    res = compute_pca(store, names, [""] * 8, 3, "tic", True, "mine", peaks=peaks, figures=False)
    assert res["mode"] == "mine" and res["n_features"] == 5
    assert res["peak_names"] == ["P5", "P10", "P20", "P30", "P40"]
    assert res["selection_steps"] and "figures" not in res


def test_selected_peak_pca_needs_peaks(store):
    with pytest.raises(ValueError):
        compute_pca(store, list(store.vec), [], 2, "tic", True, "mine", peaks=[])


def test_step_templates():
    assert len(pca_steps("all")) == len(pca_steps("mine")) == 7
    assert len(selection_steps()) == 6


class PeakStore(FakeStore):
    """Isolated peaks (every 5th bin) on a faint flat background; the first peak is under 3x its neighbour."""

    def __init__(self, n=8, p=60, seed=1):
        rng = np.random.default_rng(seed)
        self.vec = {}
        for i in range(n):
            v = np.full(p, 5.0)
            v[5::5] = 400.0
            v[5] = 10.0
            v[10 if i < n // 2 else 30] *= 4
            self.vec[f"s{i}"] = rng.poisson(v).astype(float)


def test_peak_filter_keeps_only_peaks_and_normalises_by_the_full_total():
    st = PeakStore()
    names = list(st.vec)
    off = compute_pca(st, names, [""] * 8, 3, "tic", True, "all", figures=False)
    on = compute_pca(st, names, [""] * 8, 3, "tic", True, "all", figures=False, filter_peaks=True,
                     ratio_min=3, frac_min=0.005)
    assert off["n_features"] == 60 and on["filter_peaks"] and not off["filter_peaks"]
    assert on["n_features"] == on["n_pass"] < off["n_features"]
    assert all(m % 5 == 0.5 and m > 5.5 for m in on["peak_mz"])      # background bins and the weak peak are gone
    assert all(r is None or r >= 3 for r in on["peak_ratio"]) and all(f >= 0.005 for f in on["peak_frac"])
    full = [st.vec[n].sum() for n in names]
    assert np.allclose(on["norm_totals"], full) and off["norm_totals"] is None
    assert np.asarray(on["counts"]).shape == (8, on["n_features"])


def test_peak_filter_errors_when_nothing_passes(store):
    with pytest.raises(ValueError, match="pass the filter"):
        compute_pca(store, list(store.vec), [""] * 8, 3, "tic", True, "all", figures=False,
                    filter_peaks=True, ratio_min=50, frac_min=0.5)


def test_peak_filter_is_ignored_for_selected_peaks(store):
    peaks = [{"name": "", "lo": 10, "hi": 11}, {"name": "", "lo": 30, "hi": 31}, {"name": "", "lo": 40, "hi": 42}]
    res = compute_pca(store, list(store.vec), [""] * 8, 2, "tic", True, "mine", peaks=peaks, figures=False,
                      filter_peaks=True, ratio_min=50)
    assert res["n_features"] == 3 and not res["filter_peaks"]


def test_tune_filter_uses_a_filter_and_checks_inputs():
    from tof_sims_driver.tune import tune_filter
    st = PeakStore(n=8)
    names, labels = list(st.vec), ["a"] * 4 + ["b"] * 4
    r = tune_filter(st, names, labels, ["a", "b"], bins=(1.0,), ratios=(0, 3), fracs=(0, 0.005), min_features=5, n_perm=50)
    assert r["best"]["filter"] and r["n_files"] == 8 and r["baseline"][0]["filter"] is False
    assert all(r_["n_features"] >= 5 for r_ in r["results"])
    assert r["figure"].startswith("data:image/png;base64,")
    with pytest.raises(ValueError, match="two different"):
        tune_filter(st, names, labels, ["a"])
    with pytest.raises(ValueError, match="at least 2"):
        tune_filter(st, names, ["a"] + ["b"] * 7, ["a", "b"])


def test_tune_closest_pc_reports_pc_loadings_and_shuffle_p():
    from tof_sims_driver.tune import tune_filter
    st = PeakStore(n=8)
    names, labels = list(st.vec), ["a"] * 4 + ["b"] * 4
    r = tune_filter(st, names, labels, ["a", "b"], bins=(1.0,), ratios=(0, 3), fracs=(0, 0.005), min_features=5,
                    min_var=0.0, n_perm=100)
    b = r["best"]
    assert 1 <= b["pc"] <= len(r["pcs"])
    assert b["dz"] == pytest.approx(r["pcs"][b["pc"] - 1]["dz"])
    assert 0 < r["p_corr"] <= 1 and r["n_compared"] >= 1
    assert 0 < len(r["loadings"]) <= 12 and r["figure"].startswith("data:image/png;base64,")
    # the A/B marker peak makes PC1 the separating PC, so it is not the closest one
    assert b["pc"] != 1 and r["pcs"][0]["dz"] > b["dz"]
