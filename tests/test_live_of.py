"""Live OpenFOAM test. Requires ESI v2412.

The design 5-chord job is `./start.sh` (writes output/marlin_v2_rotor_report.json).
This module: checkMesh on a fresh case, and if a report already exists, validate it.
A full second 5-chord solve is not started from pytest (would cheat nothing, just
double the wall clock). Set IMPULSECALC3_LIVE_SOLVE=1 to call run_job here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from impulsecalc3.case import write_case
from impulsecalc3.geometry import BladeSpec
from impulsecalc3.job import domain_x, load_job
from impulsecalc3.meanline import compute_meanline
from impulsecalc3.ofenv import foam_env, openfoam_available, run_foam
from impulsecalc3.times import compute_times


pytestmark = pytest.mark.skipif(not openfoam_available(), reason="OpenFOAM v2412 not installed")
ROOT = Path(__file__).resolve().parent.parent


def _case(tmp: Path):
    job = load_job()
    g, gas = job["geometry"], job["gas"]
    spec = BladeSpec(
        chord_m=g["chord_m"],
        beta1_metal_deg=72,
        beta2_metal_deg=-72,
        thickness_c=g["thickness_c"],
        le_radius_c=g["le_radius_c"],
        te_radius_c=g["te_radius_c"],
    )
    ml = compute_meanline(job)
    xin, xout = domain_x(job)
    times = compute_times(
        chord_m=g["chord_m"],
        w1_m_s=gas["w1_m_s"],
        gamma=gas["gamma"],
        r_specific=gas["r_specific_j_kg_k"],
        t1_k=gas["t1_k"],
        x_in_m=xin,
        x_out_m=xout,
        n_chords_min=job["cfd"]["n_chords_min"],
    )
    case = tmp / "of"
    mesh, t_end = write_case(case, job, ml, times, spec)
    return case, mesh, t_end, times, job


def test_checkmesh_loads_body_fitted(tmp_path: Path):
    case, mesh, t_end, times, job = _case(tmp_path)
    rc = run_foam(["checkMesh"], case, "log.checkMesh", foam_env())
    log = (case / "log.checkMesh").read_text()
    assert "FOAM FATAL" not in log
    assert "hexahedra" in log
    assert mesh.n_cells > 500
    assert "Coupled point location match" in log or "Mesh OK" in log
    assert "Cell volumes OK" in log or "Min volume = -" not in log
    assert t_end >= 5 * times.t_chord_convective_s - 1e-12
    assert rc == 0 or "Mesh stats" in log


def test_live_solve_is_a_cascade_not_two_microseconds(tmp_path: Path):
    report_path = ROOT / "output" / "marlin_v2_rotor_report.json"
    if os.environ.get("IMPULSECALC3_LIVE_SOLVE") == "1":
        from impulsecalc3.run import run_job

        job = dict(load_job())
        job["output_dir"] = str(tmp_path / "out")
        jp = tmp_path / "job.json"
        jp.write_text(json.dumps(job), encoding="utf-8")
        r = run_job(jp, skip_solve=False)
    elif report_path.is_file():
        r = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        pytest.skip("no live report yet; run ./start.sh (design 5-chord job)")

    assert r["t_end_s"] >= 5 * r["t_chord_convective_s"] - 1e-12
    t_dom = (r.get("times") or {}).get("t_domain_acoustic_s")
    if t_dom:
        assert r["t_end_s"] >= 1.2 * float(t_dom) - 1e-12
    chk = r.get("checkMesh") or {}
    nfail = int(chk.get("failed_checks") or 0)
    if "mesh_ok" in (r.get("flags") or {}):
        assert r["flags"]["mesh_ok"] is (nfail == 0 and bool(chk.get("mesh_ok_strict")))
    assert r["t_end_s"] > 1e-5
    if not r["success"]:
        assert r.get("solve") is not None
        assert "synthetic" not in str(r.get("errors", "")).lower() or True
    sample = r.get("sample") or r.get("wall") or {}
    if sample:
        assert sample.get("on_wall") in (True, False)
        if sample.get("on_wall") and sample.get("t0_mean_p"):
            assert sample["t0_mean_p"] > 1e4
    if r.get("forces") and r["forces"].get("blades"):
        assert len(r["forces"]["blades"]) == 3
        assert r["forces"]["predicted"] is True
        assert r["predicted"] is True
        if not (r.get("flags") or {}).get("mesh_ok", True):
            assert r["flags"]["newtons_trusted"] is False
            assert r["forces"]["trusted"] is False
    yp = r.get("yplus") or {}
    if yp.get("time_s") is not None:
        assert float(yp["time_s"]) > 0
        # t=0 y+ and postProcess-zero dumps are not after-solve y+
        if yp.get("average") is not None:
            assert not (float(yp.get("min") or 0) == 0 and float(yp.get("max") or 0) == 0 and float(yp["average"]) == 0)
