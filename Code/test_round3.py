"""Tests for the round-3 scripts -- run with:  python -m pytest Code/test_round3.py -q"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from dataset_preparing_allsubtypes import fair_split
from exp_allsub import disjoint_split
from exp_controls import FoldResampledMVS, ProportionalSMOTE
from exp_round3 import diversity_rows, meta_predict, view_outputs

VIEWS = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9, 10, 11], [12, 13], [14]]


def fitted_mvs():
    X, y = make_classification(n_samples=1500, n_features=15, n_informative=8, n_classes=4,
                               weights=[0.6, 0.25, 0.1, 0.05], random_state=0)
    m = FoldResampledMVS(views_indices=VIEWS,
                         first_level_learners=[RandomForestClassifier(n_estimators=20, random_state=1)
                                               for _ in VIEWS],
                         meta_learner=RandomForestClassifier(n_estimators=20, random_state=1),
                         k=3, random_state=1, resampler=ProportionalSMOTE(1000, 1))
    return m.fit(X[:1000], y[:1000]), X[1000:], y[1000:]


def test_meta_rebuild_matches_library_predict():
    m, X, _ = fitted_mvs()
    np.testing.assert_array_equal(meta_predict(m, view_outputs(m, X)), m.predict(X))


def test_diversity_identical_views_have_no_disagreement():
    y = np.array([0, 1, 1, 0, 1])
    out = (np.array([0, 1, 0, 0, 1]), None)
    row = diversity_rows([out, out], y, ["a", "b"])[0]
    assert row["disagreement"] == 0
    assert row["q_statistic"] == 1


def test_fair_split_respects_capacity_and_total():
    alloc = fair_split(100, [10, 200, 200])
    assert alloc[0] == 10 and sum(alloc) == 100 and abs(alloc[1] - alloc[2]) <= 1


def test_disjoint_split_keeps_captures_whole():
    labels = np.repeat([0, 1], 50)
    captures = np.array([f"c{i % 5}" for i in range(50)] + ["single"] * 50)
    t_rank = np.r_[np.zeros(50), np.linspace(0, 1, 50, endpoint=False)]
    test = disjoint_split(labels, captures, t_rank, seed=3)
    for c in np.unique(captures[:50]):
        assert len(set(test[:50][captures[:50] == c])) == 1      # never split
    assert test[50:].sum() == 15                                 # last 30% in time
    assert test[50:][-15:].all()


def test_each_lovo_model_drops_exactly_one_distinct_view():
    from exp_round3 import LOVO, build
    data = {"X_real": np.zeros((10, 15)), "y_real": np.zeros(10)}
    assert len(set(LOVO)) == 4
    dropped = []
    for name in LOVO:
        model, _ = build(name, data, seed=0, views=VIEWS)
        assert len(model.views_indices) == 3
        dropped.append(next(i for i, v in enumerate(VIEWS) if list(v) not in model.views_indices))
    assert sorted(dropped) == [0, 1, 2, 3]


def test_ordered_proba_matches_sklearn_and_is_bit_stable():
    from exp_controls import OrderedRandomForest
    X, y = make_classification(n_samples=3000, n_features=10, n_classes=3, n_informative=5,
                               random_state=0)
    ours = OrderedRandomForest(n_estimators=50, random_state=1, n_jobs=-1).fit(X[:2000], y[:2000])
    ref = RandomForestClassifier(n_estimators=50, random_state=1, n_jobs=-1).fit(X[:2000], y[:2000])
    p = ours.predict_proba(X[2000:])
    np.testing.assert_allclose(p, ref.predict_proba(X[2000:]), rtol=0, atol=1e-12)
    assert np.array_equal(p, ours.predict_proba(X[2000:]))
    np.testing.assert_array_equal(ours.predict(X[2000:]), ref.predict(X[2000:]))


def test_lgbm_uses_the_stabilised_settings():
    from exp_round3 import lgbm
    params = lgbm(0).get_params()
    assert params["min_child_samples"] == 100 and params["reg_lambda"] == 1.0
    assert params["deterministic"] is True


def test_hybrid_adds_full_feature_view():
    from exp_round3 import build
    data = {"X_real": np.zeros((10, 15)), "y_real": np.zeros(10)}
    model, which = build("HYB_FULL", data, seed=0, views=VIEWS)
    assert which == "real"
    assert len(model.views_indices) == 5
    assert model.views_indices[:4] == [list(v) for v in VIEWS]
    assert model.views_indices[4] == list(range(15))


def test_hybrid_random_control_keeps_view_sizes():
    from exp_round3 import build
    data = {"X_real": np.zeros((10, 15)), "y_real": np.zeros(10)}
    model, _ = build("HYB_RAND", data, seed=0, views=VIEWS)
    sizes = [len(v) for v in model.views_indices]
    assert sizes == [len(v) for v in VIEWS] + [15]
    assert sorted(c for v in model.views_indices[:4] for c in v) == list(range(15))


def test_allsub_builds_hybrid_on_real_flows():
    from exp_allsub import build
    data = {"X_real": np.zeros((10, 15)), "y_real": np.zeros(10),
            "X_bal": np.ones((12, 15)), "y_bal": np.zeros(12)}
    model, Xtr, _ = build("HYB_FULL", data, seed=0, views=VIEWS)
    assert len(model.views_indices) == 5 and Xtr is data["X_real"]
    _, Xtr, _ = build("SV69", data, seed=0, views=VIEWS)
    assert Xtr is data["X_bal"]
