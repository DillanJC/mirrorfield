# -*- coding: utf-8 -*-
"""Tests for the boundary-calibration analysis utilities and the persistence-fix
signal helper (added 2026-07-04 auto session — enforcement beats prose)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_module(relpath, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bc = _load_module("experiments/boundary_calibration/boundary_calibration.py", "bc")
syco = _load_module("experiments/sycophancy/sycophancy.py", "syco_mod")


class TestWilson:
    def test_half(self):
        lo, hi = bc.wilson(50, 100)
        assert lo < 0.5 < hi
        assert abs((lo + hi) / 2 - 0.5) < 0.01

    def test_bounds(self):
        for k, n in [(0, 10), (10, 10), (1, 3), (99, 100)]:
            lo, hi = bc.wilson(k, n)
            assert 0.0 <= lo <= hi <= 1.0

    def test_zero_n(self):
        assert bc.wilson(0, 0) == (0.0, 1.0)

    def test_shrinks_with_n(self):
        w_small = bc.wilson(5, 10)
        w_big = bc.wilson(500, 1000)
        assert (w_big[1] - w_big[0]) < (w_small[1] - w_small[0])

    def test_known_value(self):
        # k=58, n=100 (the §4y torn bin, seed 42): CI ≈ [0.482, 0.672]
        lo, hi = bc.wilson(58, 100)
        assert abs(lo - 0.482) < 0.005 and abs(hi - 0.672) < 0.005


class TestSignals:
    def test_empty_token_list(self):
        s = syco._signals([])
        assert set(s) == {"mm", "me", "br", "p_int"}
        assert all(np.isnan(v) for v in s.values())

    def test_finite_signals(self):
        # fake top-5 log-probs for 3 tokens: strongly peaked distribution
        tl = [{1: -0.05, 2: -4.0, 3: -5.0, 4: -6.0, 5: -7.0} for _ in range(3)]
        s = syco._signals(tl)
        assert np.isfinite(s["mm"]) and np.isfinite(s["me"]) and np.isfinite(s["br"])
        # p_int may be nan if calibrator unavailable, but if finite it's a probability
        if np.isfinite(s["p_int"]):
            assert 0.0 <= s["p_int"] <= 1.0

    def test_pint_delegate_matches(self):
        tl = [{1: -0.5, 2: -1.5, 3: -3.0, 4: -4.0, 5: -5.0} for _ in range(4)]
        s = syco._signals(tl)
        p = syco._pint(tl)
        assert (np.isnan(p) and np.isnan(s["p_int"])) or p == s["p_int"]


class TestBackwardCompatAssembly:
    def test_get_fallback_yields_nan(self):
        old_row = {"p1": 0.8, "p2": 0.7}  # pre-fix checkpoint row
        assert np.isnan(float(old_row.get("mm1", float("nan"))))
        new_row = {"p1": 0.8, "p2": 0.7, "mm1": 1.2}
        assert float(new_row.get("mm1", float("nan"))) == 1.2


class TestFrozenConstants:
    def test_taus_and_edges(self):
        # the locked §4y bin edges embed the frozen thresholds — drift here would
        # silently change every verdict
        assert bc.TAU_A == 0.7763 and bc.TAU_P == 0.8684
        assert bc.EDGES[1] == bc.TAU_A and bc.EDGES[5] == bc.TAU_P
        assert bc.MIN_N == 30


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
