"""Knobs write the metal: sagitta, t/c, beta, Z/σ change polygon/pitch. Families build."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from impulsecalc3.geometry import (
    polygon_signed_area,
    profile_from_job,
    profile_from_points,
    spec_from_job,
)
from impulsecalc3.job import (
    apply_geometry_aliases,
    apply_packing,
    load_job,
    pitch_m,
    set_n_blades_machine,
    set_solidity,
    validate_job,
)
from impulsecalc3.preview import knobs_to_job, write_preview

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"


def _ymax(poly):
    return max(p[1] for p in poly)


def _area(poly):
    return polygon_signed_area(poly)


def test_upper_sagitta_changes_cup_polygon():
    job = load_job(CONFIGS / "geom_impulse_bucket.json")
    p0 = profile_from_job(job)
    a0, y0 = _area(p0), _ymax(p0)
    job["geometry"]["upper_sagitta_c"] = 0.58
    p1 = profile_from_job(job)
    assert p0 != p1
    assert _ymax(p1) > y0 + 1e-5
    assert _area(p1) > a0


def test_lower_sagitta_changes_cup_polygon():
    job = load_job(CONFIGS / "geom_impulse_bucket.json")
    p0 = profile_from_job(job)
    a0 = _area(p0)
    job["geometry"]["lower_sagitta_c"] = 0.12
    p1 = profile_from_job(job)
    assert p0 != p1
    assert abs(_area(p1) - a0) > 1e-8


def test_v2_alias_upper_h_lower_h():
    job = load_job(CONFIGS / "geom_impulse_bucket.json")
    g = job["geometry"]
    g.pop("upper_sagitta_c")
    g.pop("lower_sagitta_c")
    g["upper_h"] = 0.42
    g["lower_h"] = 0.18
    apply_geometry_aliases(g)
    assert g["upper_sagitta_c"] == 0.42
    assert g["lower_sagitta_c"] == 0.18
    job = validate_job(job)
    poly = profile_from_job(job)
    assert poly[0] == poly[-1]
    assert _area(poly) > 0


def test_thickness_and_beta_change_foil_polygon():
    job = load_job(CONFIGS / "geom_foil.json")
    assert "circular_arc" in job["geometry"]["profile_family"]
    p0 = profile_from_job(job)
    a0 = _area(p0)
    job["geometry"]["thickness_c"] = 0.20
    p1 = profile_from_job(job)
    assert p0 != p1
    assert _area(p1) > a0
    job["geometry"]["beta1_flow_deg"] = 50.0
    spec = spec_from_job(job)
    assert abs(spec.beta1_metal_deg - 50.0) < 1e-9
    p2 = profile_from_job(job, spec)
    assert p2 != p1


def test_solidity_changes_pitch():
    job = load_job(CONFIGS / "geom_impulse_bucket.json")
    p0 = pitch_m(job)
    set_solidity(job["geometry"], 1.50)
    job = validate_job(job)
    p1 = pitch_m(job)
    assert abs(p1 - p0) > 1e-6
    assert abs(p1 - job["geometry"]["chord_m"] / 1.50) < 1e-12


def test_n_blades_machine_changes_pitch():
    job = load_job(CONFIGS / "geom_impulse_bucket.json")
    p0 = pitch_m(job)
    poly0 = profile_from_job(job)
    set_n_blades_machine(job["geometry"], 40)
    job = validate_job(job)
    p1 = pitch_m(job)
    assert abs(p1 - p0) > 1e-6
    # Z=40 at rm=0.0375 → s = 2π rm / 40
    expect = 2.0 * 3.141592653589793 * 0.0375 / 40.0
    assert abs(p1 - expect) < 1e-9
    poly1 = profile_from_job(job)
    # fillet cap uses pitch, so the closed polygon can move; pitch always moves.
    assert poly1[0] == poly1[-1]


def test_foil_cup_points_all_build():
    for name in ("geom_foil.json", "geom_impulse_bucket.json", "geom_points.json"):
        job = load_job(CONFIGS / name)
        poly = profile_from_job(job)
        assert poly[0] == poly[-1]
        assert _area(poly) > 0


def test_open_polyline_fails_loud():
    open_poly = [(0.0, 0.0), (0.01, 0.0), (0.01, 0.002), (0.0, 0.002)]
    with pytest.raises(ValueError, match="open polyline"):
        profile_from_points(open_poly)


def test_preview_png_moves_when_sagitta_moves(tmp_path: Path):
    knobs = {
        "profile_family": "impulse_bucket",
        "upper_sagitta_c": 0.45,
        "lower_sagitta_c": 0.22,
        "chord_m": 0.01,
        "solidity": 1.13688,
        "n_blades_machine": 25,
        "mean_radius_m": 0.0375,
        "rotor_tip_radius_m": 0.04,
        "hub_radius_m": 0.035,
        "span_m": 0.005,
        "beta1_flow_deg": 72.0,
        "beta2_flow_deg": -72.0,
        "le_fillet_r_c": 0.04,
        "te_fillet_r_c": 0.04,
    }
    job_a = knobs_to_job(knobs)
    job_a["name"] = "knobs_a"
    job_a["output_dir"] = str(tmp_path / "out_a")
    info_a = write_preview(job_a, dest=tmp_path / "a")
    knobs["upper_sagitta_c"] = 0.58
    job_b = knobs_to_job(knobs)
    job_b["name"] = "knobs_b"
    job_b["output_dir"] = str(tmp_path / "out_b")
    info_b = write_preview(job_b, dest=tmp_path / "b")
    assert info_a["ok"] and info_b["ok"]
    assert info_b["poly_ymax_m"] > info_a["poly_ymax_m"] + 1e-5
    pa, pb = Path(info_a["png"]), Path(info_b["png"])
    assert pa.is_file() and pb.is_file()
    assert pa.read_bytes() != pb.read_bytes()


def test_preview_foil_png_moves_when_thickness_moves(tmp_path: Path):
    knobs = {
        "profile_family": "circular_arc_camber_metal_angles",
        "thickness_c": 0.10,
        "chord_m": 0.01,
        "solidity": 1.13688,
        "n_blades_machine": 25,
        "mean_radius_m": 0.0375,
        "rotor_tip_radius_m": 0.04,
        "hub_radius_m": 0.035,
        "span_m": 0.005,
        "beta1_flow_deg": 72.0,
        "beta2_flow_deg": -72.0,
    }
    job_a = knobs_to_job(knobs)
    job_a["output_dir"] = str(tmp_path / "out_fa")
    info_a = write_preview(job_a, dest=tmp_path / "fa")
    knobs["thickness_c"] = 0.18
    job_b = knobs_to_job(knobs)
    job_b["output_dir"] = str(tmp_path / "out_fb")
    info_b = write_preview(job_b, dest=tmp_path / "fb")
    assert info_b["poly_area_m2"] > info_a["poly_area_m2"]
    assert Path(info_a["png"]).read_bytes() != Path(info_b["png"]).read_bytes()


def test_skip_solve_remeshes_geom_not_marlin(tmp_path: Path):
    from impulsecalc3.run import run_job

    src = json.loads((CONFIGS / "geom_impulse_bucket.json").read_text())
    src["output_dir"] = str(tmp_path / "out")
    src["name"] = "knobs_skip"
    jp = tmp_path / "job.json"
    jp.write_text(json.dumps(src), encoding="utf-8")
    r = run_job(jp, skip_solve=True, post_only=False)
    case = Path(r["case_dir"])
    preview = case / "mesh_preview.png"
    assert preview.is_file()
    assert (case / "constant" / "polyMesh" / "points").is_file()
    # Live Marlin report not touched by this geom job.
    live = ROOT / "output" / "marlin_v2_rotor_report.json"
    assert live.is_file()


def test_knobs_to_job_writes_impulsecalc3_json():
    job = knobs_to_job({"upper_h": 0.48, "lower_h": 0.22, "Z": 30, "packing_driver": "z"})
    assert job["format"] == "impulsecalc3_job_v1"
    assert job["geometry"]["upper_sagitta_c"] == 0.48
    assert job["geometry"]["n_blades_machine"] == 30
    assert job["name"] != "marlin_v2_rotor"
    assert job["geometry_test"] is True
