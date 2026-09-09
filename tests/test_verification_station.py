"""PREDICTED verification station. mesh_ok is failed_checks==0, not positive volumes."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from impulsecalc3.case import write_case
from impulsecalc3.geometry import spec_from_job
from impulsecalc3.job import domain_x, load_job
from impulsecalc3.meanline import compute_meanline
from impulsecalc3.ofenv import foam_env, openfoam_available, run_foam
from impulsecalc3.preview import knobs_to_job
from impulsecalc3.times import compute_times
from viewer.ofio import parse_checkmesh

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"
VERIF = CONFIGS / "verification_station.json"
MARLIN = CONFIGS / "marlin_v2_rotor.json"


def _write(tmp: Path, job_path: Path):
    job = load_job(job_path)
    spec = spec_from_job(job)
    ml = compute_meanline(job)
    spec.beta1_metal_deg = ml.beta1_metal_deg
    spec.beta2_metal_deg = ml.beta2_metal_deg
    g, gas = job["geometry"], job["gas"]
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
    case = tmp / job["name"]
    mesh, t_end = write_case(case, job, ml, times, spec)
    return job, mesh, t_end, case


def test_verification_station_json_is_predicted_not_marlin():
    raw = json.loads(VERIF.read_text(encoding="utf-8"))
    job = load_job(VERIF)
    assert job["name"] == "verification_station"
    assert job["name"] != "marlin_v2_rotor"
    assert job["predicted"] is True
    assert job["geometry_test"] is True
    assert job["geometry"]["profile_family"] == "impulse_bucket"
    assert abs(job["gas"]["w1_m_s"] - 950.0) < 1e-9
    want_u = 40000.0 * 2.0 * math.pi / 60.0 * 0.0375
    assert abs(job["gas"]["blade_speed_u_m_s"] - want_u) < 0.05
    assert abs(want_u - 157.08) < 0.05
    assert abs(job["gas"]["p1_pa"] - 550000.0) < 1e-6
    assert abs(job["gas"]["t1_k"] - 1100.0) < 1e-9
    assert abs(float(job["engine"]["OF"]) - 1.20) < 1e-12
    assert "geom_tests" in str(job["output_dir"])
    assert "253" not in json.dumps(raw["gas"])
    assert "1426" not in json.dumps(raw["gas"])
    marlin = json.loads(MARLIN.read_text(encoding="utf-8"))
    assert marlin["gas"]["w1_m_s"] == 950.0
    assert marlin["gas"]["blade_speed_u_m_s"] == 450.0
    assert marlin["name"] == "marlin_v2_rotor"


def test_verification_station_writes_positive_ogrid(tmp_path: Path):
    job, mesh, t_end, case = _write(tmp_path, VERIF)
    assert mesh.min_area_2d > 0
    assert mesh.n_cells > 500
    assert mesh.mesh_kind == "body_fitted_OH"
    assert mesh.patches["bottom"] == mesh.patches["top"]
    notes = " ".join(mesh.check_notes).lower()
    assert "stair" in notes
    assert (case / "constant" / "polyMesh" / "points").is_file()


def test_knobs_preview_cup_at_floor_station_writes(tmp_path: Path):
    job = knobs_to_job(
        {
            "family": "cup",
            "rpm": 40000,
            "r": 0.0375,
            "W1": 950,
            "p1": 550000,
            "T1": 1100,
            "mdot_engine_kg_s": 4.0,
            "mdot_gg_kg_s": 0.40,
            "OF": 1.20,
        }
    )
    job["output_dir"] = str(tmp_path / "knobs_preview")
    job["name"] = "knobs_preview"
    spec = spec_from_job(job)
    ml = compute_meanline(job)
    g, gas = job["geometry"], job["gas"]
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
    mesh, _ = write_case(tmp_path / "knobs_case", job, ml, times, spec)
    assert mesh.min_area_2d > 0
    assert abs(job["gas"]["blade_speed_u_m_s"] - 157.08) < 0.05
    assert job["name"] != "marlin_v2_rotor"


@pytest.mark.skipif(not openfoam_available(), reason="OpenFOAM v2412 not installed")
def test_verification_checkmesh_failed_checks_zero(tmp_path: Path):
    """mesh_ok tracks failed_checks==0. Positive volumes are not a pass."""
    job, mesh, t_end, case = _write(tmp_path, VERIF)
    rc = run_foam(["checkMesh"], cwd=case, log_name="log.checkMesh", env=foam_env())
    log = (case / "log.checkMesh").read_text(encoding="utf-8", errors="replace")
    chk = parse_checkmesh(log)
    nfail = int(chk.get("failed_checks") or 0)
    mesh_ok = bool(chk.get("mesh_ok_strict")) and nfail == 0
    assert "FOAM FATAL" not in log
    assert mesh.min_area_2d > 0
    assert mesh_ok, (
        f"checkMesh failed_checks={chk.get('failed_checks')} "
        f"wrong_oriented={chk.get('wrong_oriented_faces')} "
        f"nonortho_max={chk.get('nonortho_max')} skew_max={chk.get('skew_max')} "
        f"rc={rc} (positive volumes are not mesh_ok)"
    )
    assert nfail == 0
    assert chk.get("wrong_oriented_faces", 0) == 0


@pytest.mark.skipif(not openfoam_available(), reason="OpenFOAM v2412 not installed")
def test_knobs_preview_cup_checkmesh_failed_checks_zero(tmp_path: Path):
    job = knobs_to_job({"family": "cup", "rpm": 40000, "r": 0.0375, "W1": 950})
    job["output_dir"] = str(tmp_path / "out")
    spec = spec_from_job(job)
    ml = compute_meanline(job)
    g, gas = job["geometry"], job["gas"]
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
    case = tmp_path / "of"
    write_case(case, job, ml, times, spec)
    rc = run_foam(["checkMesh"], cwd=case, log_name="log.checkMesh", env=foam_env())
    log = (case / "log.checkMesh").read_text(encoding="utf-8", errors="replace")
    chk = parse_checkmesh(log)
    nfail = int(chk.get("failed_checks") or 0)
    assert nfail == 0, chk
    assert bool(chk.get("mesh_ok_strict"))
    assert rc == 0 or "Mesh OK" in log
