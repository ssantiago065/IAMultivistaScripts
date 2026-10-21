"""Tests for exp_controls.py -- run with:  python -m pytest Code/test_controls.py -q"""

import numpy as np
from multiviewstacking import MultiViewStacking
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from exp_controls import FoldResampledMVS, ProportionalSMOTE, random_views, smote_strategy

VIEWS = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9, 10, 11], [12, 13], [14]]


def data(n=1500, seed=0):
    X, y = make_classification(n_samples=n, n_features=15, n_informative=8, n_classes=4,
                               weights=[0.6, 0.25, 0.1, 0.05], random_state=seed)
    return X, y


def build(cls, seed, meta, **kw):
    return cls(views_indices=[list(v) for v in VIEWS],
               first_level_learners=[RandomForestClassifier(n_estimators=20, random_state=seed)
                                     for _ in VIEWS],
               meta_learner=meta, k=3, random_state=seed, **kw)


def test_no_resampler_reproduces_library_exactly():
    X, y = data()
    for meta_factory in (lambda: LogisticRegression(max_iter=1000, random_state=1),
                         lambda: RandomForestClassifier(n_estimators=20, random_state=1)):
        lib = build(MultiViewStacking, 7, meta_factory()).fit(X[:1000], y[:1000])
        ours = build(FoldResampledMVS, 7, meta_factory(), resampler=None).fit(X[:1000], y[:1000])
        np.testing.assert_array_equal(lib.predict(X[1000:]), ours.predict(X[1000:]))
        np.testing.assert_allclose(lib.predict_proba(X[1000:]), ours.predict_proba(X[1000:]))


def test_meta_learner_sees_only_real_instances():
    X, y = data()
    seen = {}

    class SpyLR(LogisticRegression):
        def fit(self, Xm, ym, **kw):
            seen["n"] = len(ym)
            return super().fit(Xm, ym, **kw)

    m = build(FoldResampledMVS, 3, SpyLR(max_iter=1000), resampler=ProportionalSMOTE(1000, 3))
    m.fit(X[:1000], y[:1000])
    assert seen["n"] == 1000            # out-of-fold rows = real training rows, no synthetic ones


def test_proportional_smote_targets():
    X, y = data()
    rs = ProportionalSMOTE(n_full_train=1000, seed=0)
    import exp_controls
    old = exp_controls.SMOTE_LIMIT
    exp_controls.SMOTE_LIMIT = 300
    try:
        Xr, yr = rs(X[:600], y[:600], call_idx=0)          # 600/1000 -> limit 180
        counts = np.bincount(yr)
        orig = np.bincount(y[:600])
        for c in range(4):
            assert counts[c] == (180 if orig[c] < 180 else orig[c])
    finally:
        exp_controls.SMOTE_LIMIT = old


def test_smote_strategy_matches_experiment_A_rule():
    y = np.array([0] * 50 + [1] * 5 + [2] * 20)
    assert smote_strategy(y, 30) == {0: 50, 1: 30, 2: 30}


def test_random_views_keep_sizes_and_cover_all_columns():
    real_sizes = [27, 32, 7, 3]
    start, views = 0, []
    for s in real_sizes:
        views.append(list(range(start, start + s)))
        start += s
    rv = random_views(views, seed=11)
    assert [len(v) for v in rv] == real_sizes
    assert sorted(c for v in rv for c in v) == list(range(69))
    assert rv != views                                   # actually shuffled
    assert rv == random_views(views, seed=11)            # reproducible per seed
