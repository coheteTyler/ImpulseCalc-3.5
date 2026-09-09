"""Off-nominal vs NTRS — cited bounds only. PREDICTED / SCOPING."""

from __future__ import annotations

import math

import pytest

from impulsecalc3.meanline import compute_meanline
from impulsecalc3.ntrs_checks import (
    LITERATURE,
    _finding,
    evaluate,
    turning_deg,
)
from impulsecalc3.preview import knobs_to_job


def _default_station(**extra):
    k = {
        "family": "cup",
        "W1": 950,
        "rpm": 40000,
        "r": 0.0375,
        "beta1": 72,
        "beta2": -72,
        "T1": 1100,
        "p1": 550000,
    }
    k.update(extra)
    return knobs_to_job(k)


def test_pm72_turning_is_high_vs_140_n94_23059():
    job = _default_station()
    ml = compute_meanline(job)
    turn = turning_deg(ml.beta1_flow_deg, ml.beta2_flow_deg)
    assert abs(turn - 144.0) < 1e-9
    findings = evaluate(ml, job)
    row = next(f for f in findings if f["id"] == "rotor_turning")
    assert row["status"] == "HIGH"
    assert row["ntrs_id"] == "19940018586"
    assert "N94-23059" in row["report"]
    assert abs(row["value"] - 144.0) < 1e-9
    assert abs(row["bound"] - 140.0) < 1e-12
    assert abs(row["pct_vs_bound"] - (144.0 / 140.0 - 1.0) * 100.0) < 1e-9
    assert row["predicted"] is True
    assert row["authority"] == "SCOPING"
    huber = next(f for f in findings if f["id"] == "huber_high_turning_context")
    assert huber["status"] == "INFO"
    assert huber["ntrs_id"] == "19920066198"


def test_mw1_computed_not_guessed():
    job = _default_station()
    ml = compute_meanline(job)
    a = math.sqrt(1.3 * 320.0 * 1100.0)
    want = 950.0 / a
    assert abs(ml.Mw1 - want) < 1e-12
    findings = evaluate(ml, job)
    row = next(f for f in findings if f["id"] == "mw1_inlet")
    assert abs(row["value"] - want) < 1e-12
    # not a round-number guess
    assert abs(want - 1.4) > 1e-3
    assert abs(want - 1.5) > 1e-3
    assert row["status"] == "INFO"  # 1.0 ≤ Mw1 < 1.5 off 1968 chart
    gold = next(f for f in findings if f["id"] == "mw1_goldman_range")
    assert gold["status"] == "INFO"  # 1.40 < 1.5, method valid, not NOMINAL
    assert gold["ntrs_id"] == "19680010807"


def test_literature_rows_have_ntrs_id():
    job = _default_station()
    ml = compute_meanline(job)
    findings = evaluate(ml, job)
    lit = [f for f in findings if f["id"] not in ("impulse_w2_over_w1", "of_w2_over_w1") and not str(f["id"]).startswith("cfd_")]
    assert lit
    for f in lit:
        assert f.get("ntrs_id"), f"missing ntrs_id for {f.get('id')}"
        assert f["url"] == f"https://ntrs.nasa.gov/citations/{f['ntrs_id']}"
        assert f["predicted"] is True
        assert f["authority"] == "SCOPING"
        assert "eta" not in f
        assert "FIRMA" not in (f.get("report") or "")
        assert "FIRMA" not in (f.get("note") or "")


def test_missing_ntrs_id_fails():
    with pytest.raises(ValueError, match="missing ntrs_id"):
        _finding(
            id="bogus",
            title="x",
            value=1,
            bound=1,
            unit="deg",
            pct_vs_bound=0,
            status="NOMINAL",
            ntrs_id="",
            report="no",
        )


def test_impulse_ratio_na_not_fake_nasa():
    job = _default_station()
    ml = compute_meanline(job)
    row = next(f for f in evaluate(ml, job) if f["id"] == "impulse_w2_over_w1")
    assert row["status"] == "N/A"
    assert row["ntrs_id"] is None
    assert "locked" in (row.get("note") or "").lower() or "construction" in (row.get("report") or "").lower()


def test_cfd_flags_extra_rows_do_not_upgrade_literature():
    job = _default_station()
    ml = compute_meanline(job)
    findings = evaluate(ml, job, cfd_flags={"mesh_ok": True, "force_plateau": False})
    lit = [f for f in findings if not str(f["id"]).startswith("cfd_")]
    assert all(f["authority"] == "SCOPING" for f in lit)
    mesh = next(f for f in findings if f["id"] == "cfd_mesh_ok")
    plat = next(f for f in findings if f["id"] == "cfd_force_plateau")
    assert mesh["status"] == "NOMINAL"
    assert plat["status"] == "HIGH"
    assert mesh["authority"] == "SCOPING"


def test_cited_ids_are_the_hardcoded_set():
    ids = {v["ntrs_id"] for v in LITERATURE.values()}
    assert ids == {"19940018586", "19920066198", "19740025363", "19680010807", "19890012364"}


def test_write_preview_includes_ntrs_checks(tmp_path):
    from impulsecalc3.preview import outline_from_knobs

    info = outline_from_knobs(
        {"family": "cup", "W1": 950, "rpm": 40000, "r": 0.0375, "beta1": 72, "beta2": -72},
        dest=tmp_path / "ntrs",
    )
    assert info["ntrs_checks"]
    turn = next(f for f in info["ntrs_checks"] if f["id"] == "rotor_turning")
    assert turn["status"] == "HIGH"
    assert abs(info["triangles"]["u_m_s"] - job_u(info)) < 1.0


def job_u(info):
    # default PREDICTED station U from rpm×r ≈ 157
    return 40000.0 * 2.0 * math.pi / 60.0 * 0.0375


def test_of_station_adds_mw1_and_w_ratio_rows():
    job = _default_station()
    ml = compute_meanline(job)
    of_st = {
        "Mw1_OF": 1.25,
        "w_inlet_m_s": 937.0,
        "w_outlet_m_s": 378.0,
        "w2_over_w1": 378.0 / 937.0,
        "probe": "owner cells of inlet/outlet patches",
        "predicted": True,
    }
    findings = evaluate(ml, job, cfd_flags={"mesh_ok": True, "force_plateau": True}, of_station=of_st)
    of_in = next(f for f in findings if f["id"] == "mw1_of_inlet")
    of_g = next(f for f in findings if f["id"] == "mw1_of_goldman_range")
    of_r = next(f for f in findings if f["id"] == "of_w2_over_w1")
    assert of_in["ntrs_id"] == "19680010807"
    assert of_g["ntrs_id"] == "19680010807"
    assert of_in["status"] == "HIGH"  # 1.25 >= 1 sonic
    assert of_g["status"] == "NOMINAL"  # 1.25 < 1.5
    assert of_in["authority"] == "SCOPING"
    assert of_in["predicted"] is True
    assert of_r["ntrs_id"] is None
    assert of_r["status"] == "INFO"
    lit = [f for f in findings if f["id"] in ("mw1_inlet", "mw1_goldman_range", "rotor_turning")]
    assert all(f["authority"] == "SCOPING" for f in lit)
    assert all(f["predicted"] is True for f in lit)


def test_extract_of_stations_from_finished_knobs_preview():
    from impulsecalc3.job import ROOT
    from impulsecalc3.post import extract_of_stations

    case = ROOT / "output/geom_tests/knobs_preview/openfoam_cases/knobs_preview"
    if not (case / "log.rhoCentralFoam").is_file():
        return
    job = _default_station()
    st = extract_of_stations(case, job)
    assert st is not None
    assert st.get("eta_from_cfd") is None
    assert st["Mw1_OF"] is not None
    assert st["w_inlet_m_s"] is not None
    assert st["w_outlet_m_s"] is not None
    assert st["w2_over_w1"] is not None
    assert abs(st["w2_over_w1"] - 1.0) > 0.05  # OF is not the meanline lock
    assert "inlet" in (st.get("probe") or "").lower() or "12%" in (st.get("probe") or "")
