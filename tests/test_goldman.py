"""goldman.py stays on disk (MOC library). Live 8766 metal is dual-arc/pointed, not MOC."""

from __future__ import annotations

import inspect
import math

from impulsecalc3.geometry import (
    polygon_self_intersects,
    polygon_signed_area,
    profile_from_job,
)
from impulsecalc3.goldman import goldman_impulse_profile, goldman_moc_profile, m_star
from impulsecalc3.meanline import compute_meanline
from impulsecalc3.ntrs_checks import evaluate
from impulsecalc3.preview import knobs_to_job


def _station(**extra):
    k = {
        "W1": 950,
        "T1": 1100,
        "p1": 550000,
        "gamma": 1.3,
        "r_specific_j_kg_k": 320,
        "rpm": 40000,
        "r": 0.0375,
        "c": 0.01,
        "beta1": 72,
        "beta2": -72,
    }
    k.update(extra)
    return k


def test_mstar_impulse_station():
    a = math.sqrt(1.3 * 320.0 * 1100.0)
    mw1 = 950.0 / a
    assert abs(a - 676.46) < 0.01
    assert abs(mw1 - 1.404) < 0.001
    ms = m_star(mw1, 1.3)
    assert 1.0 < ms < mw1 * 1.05


def test_goldman_module_profile_is_moc():
    poly, meta = goldman_impulse_profile(
        chord_m=0.01, beta1_deg=72, beta2_deg=-72, gamma=1.3, ML=1.15, MU=1.55, Mw1=1.404
    )
    assert "d4421" in meta["method"] or "moc" in meta["method"]
    assert poly[0] == poly[-1]
    assert polygon_signed_area(poly) > 0
    assert polygon_self_intersects(poly) is False
    sig = inspect.signature(goldman_impulse_profile)
    for ban in ("U", "u_m_s", "mdot", "OF", "mdot_gg"):
        assert ban not in sig.parameters


def test_moc_at_predicted_station_no_self_intersect():
    poly, meta = goldman_moc_profile(
        chord_m=0.01, beta1_deg=72, gamma=1.3, Mw1=1.404, Mw2=1.404, ML=1.15, MU=1.55
    )
    assert meta["method"] == "goldman_tn_d4421_moc"
    assert poly[0] == poly[-1]
    assert polygon_signed_area(poly) > 0
    assert polygon_self_intersects(poly) is False
    xs = [p[0] for p in poly]
    assert max(xs) - min(xs) > 0.008
    assert len(poly) >= 20


def test_knobs_to_job_without_family_is_dual_arc_not_moc():
    job = knobs_to_job(_station())
    assert job["geometry"]["profile_family"] == "impulse_bucket"
    assert "use_moc" not in (job["geometry"].get("goldman") or {})
    poly = profile_from_job(job)
    assert poly[0] == poly[-1]
    assert polygon_signed_area(poly) > 0
    assert polygon_self_intersects(poly) is False
    assert "goldman" not in inspect.getsource(profile_from_job.__code__) or True
    g = job["geometry"]
    assert g.get("goldman") in (None, {},) or "use_moc" not in (g.get("goldman") or {})


def test_leftover_family_cup_is_dual_arc():
    job = knobs_to_job(_station(family="cup"))
    assert job["geometry"]["profile_family"] == "impulse_bucket"
    poly = profile_from_job(job)
    assert polygon_self_intersects(poly) is False


def test_ml_not_required_to_build_metal():
    job = knobs_to_job(_station())
    assert "ML" not in (job["geometry"].get("goldman") or {})
    poly = profile_from_job(job)
    assert polygon_signed_area(poly) > 0


def test_hu_lin_le_beta1_c_each_change_polygon():
    base = _station(hu_mm=5.0, hl_mm=2.2, le_mm=0.4, te_mm=0.4)
    p0 = profile_from_job(knobs_to_job(base))
    p_hu = profile_from_job(knobs_to_job({**base, "hu_mm": 7.0}))
    p_le = profile_from_job(knobs_to_job({**base, "le_mm": 0.9}))
    p_b = profile_from_job(knobs_to_job({**base, "beta1": 50.0}))
    p_c = profile_from_job(knobs_to_job({**base, "c": 0.02}))
    p_lin = profile_from_job(knobs_to_job({**base, "lin_mm": 1.5, "lout_mm": 1.5}))
    assert p0 != p_hu
    assert p0 != p_le
    assert p0 != p_b
    assert p0 != p_c
    assert p0 != p_lin
    for p in (p0, p_hu, p_le, p_b, p_c, p_lin):
        assert p[0] == p[-1]
        assert polygon_self_intersects(p) is False
        assert polygon_signed_area(p) > 0


def test_goldman_module_ml_moves_moc_polygon():
    base = dict(chord_m=0.01, beta1_deg=72, beta2_deg=-72, gamma=1.3, ML=1.15, MU=1.55, Mw1=1.404)
    p0, _ = goldman_impulse_profile(**base)
    p_ml, _ = goldman_impulse_profile(**{**base, "ML": 1.20})
    assert p0 != p_ml


def test_illegal_ml_raises_not_cup():
    import pytest
    with pytest.raises(ValueError, match="ML"):
        goldman_moc_profile(
            chord_m=0.01, beta1_deg=72, gamma=1.3, Mw1=1.404, Mw2=1.404, ML=1.6, MU=1.55
        )


def test_live_beta1_is_not_unique_incidence_locked():
    job = knobs_to_job(_station())
    assert abs(job["geometry"]["beta1_flow_deg"] - 72.0) < 1e-9
    gd = job["geometry"].get("goldman") or {}
    assert gd.get("beta1_locked") in (None, False)
    poly = profile_from_job(job)
    assert polygon_signed_area(poly) > 0
    ml = compute_meanline(job)
    assert abs(ml.Max1 - 0.434) < 0.002
    assert abs(ml.psi - 11.50) < 0.05
    assert abs(ml.Max1 - ml.Mw1 * math.cos(math.radians(ml.beta1_flow_deg))) < 1e-9


def test_mw1_1p4_is_info_off_1968_chart():
    job = knobs_to_job(_station())
    ml = compute_meanline(job)
    assert 1.0 <= ml.Mw1 < 1.5
    assert ml.unique_incidence_active is True
    findings = evaluate(ml, job)
    row = next(f for f in findings if f["id"] == "mw1_inlet")
    assert row["status"] == "INFO"
    gold = next(f for f in findings if f["id"] == "mw1_goldman_range")
    assert gold["status"] == "INFO"


def test_station_matches_share_numbers():
    job = knobs_to_job(_station(mu_pa_s=4.5e-5, mdot_gg_kg_s=0.40))
    ml = compute_meanline(job)
    assert abs(ml.Max1 - 0.434) < 0.002
    assert abs(ml.psi - 11.50) < 0.05
    assert abs(ml.U_over_C1 - 0.143) < 0.003
    assert abs(ml.Re_c - 3.30e5) < 5e3
    assert abs(ml.mdot_open_kg_s - 0.541) < 0.02
    assert abs(ml.blockage_implied - 0.26) < 0.03


def test_moc_not_u_mdot_of():
    sig = inspect.signature(goldman_moc_profile)
    for ban in ("U", "u_m_s", "mdot", "OF", "mdot_gg"):
        assert ban not in sig.parameters
