"""PREDICTED station knobs. rpm×r → U. Do not rewrite Marlin JSON."""

from __future__ import annotations

import json
import math
from pathlib import Path

from impulsecalc3.meanline import compute_meanline
from impulsecalc3.preview import knobs_to_job, triangles_from_meanline

ROOT = Path(__file__).resolve().parent.parent
MARLIN = ROOT / "configs" / "marlin_v2_rotor.json"


def test_rpm_times_rm_sets_U():
    job = knobs_to_job({"rpm": 40000, "r": 0.0375, "family": "cup"})
    want = 40000.0 * 2.0 * math.pi / 60.0 * 0.0375
    assert abs(job["gas"]["blade_speed_u_m_s"] - want) < 0.05
    assert abs(want - 157.08) < 0.05
    assert job["gas"]["predicted"] is True
    assert job["predicted"] is True
    assert abs(job["geometry"]["mean_radius_m"] - 0.0375) < 1e-12


def test_typed_U_without_rpm_sticks():
    job = knobs_to_job({"U": 252.9, "r": 0.0375, "family": "cup"})
    assert abs(job["gas"]["blade_speed_u_m_s"] - 252.9) < 1e-9
    assert "rpm" not in job["gas"] or job["gas"].get("rpm") in (None, "")


def test_rpm_wins_when_both_sent():
    job = knobs_to_job({"rpm": 40000, "U": 450.0, "r": 0.0375, "family": "cup"})
    want = 40000.0 * 2.0 * math.pi / 60.0 * 0.0375
    assert abs(job["gas"]["blade_speed_u_m_s"] - want) < 0.05


def test_station_gas_and_mdot_copy():
    job = knobs_to_job(
        {
            "family": "cup",
            "W1": 1200,
            "p1": 4.0e5,
            "T1": 900,
            "mdot_engine_kg_s": 4.0,
            "mdot_gg_kg_s": 0.40,
            "OF": 1.20,
        }
    )
    assert abs(job["gas"]["w1_m_s"] - 1200) < 1e-9
    assert abs(job["gas"]["p1_pa"] - 4.0e5) < 1e-6
    assert abs(job["gas"]["t1_k"] - 900) < 1e-9
    assert abs(job["gas"]["mdot_engine_kg_s"] - 4.0) < 1e-12
    assert abs(job["gas"]["mdot_gg_kg_s"] - 0.40) < 1e-12
    assert abs(float(job["engine"]["OF"]) - 1.20) < 1e-12
    assert job["gas"]["predicted"] is True


def test_knobs_do_not_rewrite_marlin_json():
    before = MARLIN.read_bytes()
    gas0 = json.loads(before)["gas"]
    knobs_to_job({"rpm": 40000, "r": 0.0375, "W1": 1425.83, "U": 157.1, "family": "cup"})
    after = MARLIN.read_bytes()
    assert after == before
    assert gas0["w1_m_s"] == 950.0
    assert gas0["blade_speed_u_m_s"] == 450.0


def test_triangles_from_meanline_keys():
    job = knobs_to_job({"family": "cup", "W1": 950, "rpm": 40000, "r": 0.0375, "beta1": 72, "beta2": -72})
    ml = compute_meanline(job)
    tri = triangles_from_meanline(ml)
    for k in (
        "c1_m_s",
        "c2_m_s",
        "w1_m_s",
        "w2_m_s",
        "alpha1_abs_deg",
        "alpha2_abs_deg",
        "beta1_flow_deg",
        "beta2_flow_deg",
        "u_m_s",
        "euler_work_j_kg",
        "Mw1",
        "predicted",
    ):
        assert k in tri
    assert tri["predicted"] is True
    assert abs(tri["w1_m_s"] - 950) < 1e-9
    assert abs(tri["beta1_flow_deg"] - 72) < 1e-9
    assert abs(tri["u_m_s"] - job["gas"]["blade_speed_u_m_s"]) < 1e-9
    assert "eta" not in tri
