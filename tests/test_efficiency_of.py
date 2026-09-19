"""Unit checks for Methods 6/7/8 helpers. PREDICTED formulas only."""

from __future__ import annotations

import math

from impulsecalc3.efficiency_of import (
    Y_rel,
    W_is_exit,
    delta_s,
    geom_hash,
    p0_rel,
    zeta_rel,
)


def test_p0_rel_stagnation_gt_static():
    p0 = p0_rel(1e5, 300.0, 400.0, 1.4, 287.0)
    assert p0 > 1e5


def test_Y_rel_zero_when_no_loss():
    p0 = 2e5
    p = 1e5
    assert abs(Y_rel(p0, p0, p) - 0.0) < 1e-12


def test_zeta_rel_zero_when_W_equals_Wis():
    assert abs(zeta_rel(500.0, 500.0) - 0.0) < 1e-12


def test_zeta_rel_positive_when_slow():
    z = zeta_rel(400.0, 500.0)
    assert z is not None and z > 0.0


def test_delta_s_zero_isentropic_same_state():
    assert abs(delta_s(300.0, 1e5, 300.0, 1e5, 1.4, 287.0) - 0.0) < 1e-12


def test_W_is_exit_positive():
    p01 = p0_rel(1.5e5, 1100.0, 900.0, 1.3, 320.0)
    T01 = 1100.0 + 0.5 * 900.0**2 / (1.3 * 320.0 / 0.3)
    w = W_is_exit(T01, p01, 1.2e5, 1.3, 320.0)
    assert w is not None and w > 0.0


def test_geom_hash_stable():
    job = {
        "geometry": {"chord_m": 0.01, "beta1_flow_deg": 72, "pitch_m": 0.008},
        "gas": {"p1_pa": 5.5e5, "t1_k": 1100, "w1_m_s": 950, "gamma": 1.3},
    }
    assert geom_hash(job) == geom_hash(job)
    assert len(geom_hash(job)) == 12
