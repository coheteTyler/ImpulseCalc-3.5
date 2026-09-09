"""Knob change must move the outline PNG from the real mesh/geometry writer.

No OpenFOAM. No live Marlin tree. Authority of a preview is SCOPING.
"""

from __future__ import annotations

import sys
from pathlib import Path

from impulsecalc3.geometry import profile_from_job
from impulsecalc3.preview import (
    AUTHORITY_FIELD_CFD,
    AUTHORITY_SCOPING,
    authority_from_report,
    knobs_to_job,
    outline_from_knobs,
)


def test_cup_hu_change_moves_polygon_and_png(tmp_path: Path):
    a = knobs_to_job({"family": "cup", "hu_c": 0.50, "hl_c": 0.18, "c": 0.01})
    b = knobs_to_job({"family": "cup", "hu_c": 0.36, "hl_c": 0.18, "c": 0.01})
    assert a["name"] != "marlin_v2_rotor"
    assert a["geometry_test"] is True
    pa = profile_from_job(a)
    pb = profile_from_job(b)
    ya = max(p[1] for p in pa)
    yb = max(p[1] for p in pb)
    assert abs(ya - yb) > 5e-4
    ra = outline_from_knobs({"family": "cup", "hu_c": 0.50, "hl_c": 0.18}, dest=tmp_path / "a")
    rb = outline_from_knobs({"family": "cup", "hu_c": 0.36, "hl_c": 0.18}, dest=tmp_path / "b")
    assert ra["authority"] == AUTHORITY_SCOPING
    assert rb["authority"] == AUTHORITY_SCOPING
    assert ra["cfd_ran"] is False
    tri = ra["triangles"]
    for k in ("c1_m_s", "c2_m_s", "w1_m_s", "w2_m_s", "alpha1_abs_deg", "alpha2_abs_deg",
              "beta1_flow_deg", "beta2_flow_deg", "u_m_s", "euler_work_j_kg", "Mw1"):
        assert k in tri
    assert tri["predicted"] is True
    assert ra["png"] and rb["png"]
    ba = Path(ra["png"]).read_bytes()
    bb = Path(rb["png"]).read_bytes()
    assert ba[:8] == b"\x89PNG\r\n\x1a\n"
    assert ba != bb
    assert abs(float(ra["poly_ymax_m"]) - float(rb["poly_ymax_m"])) > 5e-4
    assert "marlin_v2_rotor" not in ra["case_dir"]


def test_foil_tc_change_moves_outline(tmp_path: Path):
    ra = outline_from_knobs({"family": "foil", "t_c": 0.10, "beta1": 72, "beta2": -72}, dest=tmp_path / "fa")
    rb = outline_from_knobs({"family": "foil", "t_c": 0.20, "beta1": 72, "beta2": -72}, dest=tmp_path / "fb")
    assert ra["family"] == "circular_arc_camber_metal_angles"
    assert Path(ra["png"]).read_bytes() != Path(rb["png"]).read_bytes()
    assert abs(float(ra["poly_area_m2"]) - float(rb["poly_area_m2"])) > 1e-8


def test_authority_never_hardware_from_cfd():
    scoping = authority_from_report({"name": "knobs_preview", "flags": {"solve_ok": False}, "solve": {}})
    assert scoping == AUTHORITY_SCOPING
    field = authority_from_report(
        {"name": "knobs_preview", "flags": {"solve_ok": True, "mesh_ok": True}, "solve": {"rc": 0, "fatal": False}}
    )
    assert field == AUTHORITY_FIELD_CFD
    dirty = authority_from_report(
        {"name": "knobs_preview", "flags": {"solve_ok": True, "mesh_ok": False}, "solve": {"rc": 0, "fatal": False}}
    )
    assert dirty == AUTHORITY_SCOPING
    # Live Marlin numbers are not the UI label.
    assert authority_from_report({"name": "marlin_v2_rotor", "flags": {"solve_ok": True, "mesh_ok": True}, "solve": {"rc": 0}}) == AUTHORITY_SCOPING


def test_serve_does_not_import_impulsecalc_or_lpre():
    import impulsecalc3.serve  # noqa: F401

    bad = [m for m in sys.modules if m == "impulsecalc" or m.startswith("impulsecalc.") or m == "lpre" or m.startswith("lpre.")]
    assert all(m.startswith("impulsecalc3") for m in bad)
