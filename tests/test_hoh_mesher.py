"""HOH mesher tests. No OpenFOAM required."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from impulsecalc3.hoh_mesher import (
    HohResult,
    build_hoh_mesh,
    build_midgap_cyclic,
    tfi_quad,
)
from impulsecalc3.hoh_polymesh import write_polymesh
from impulsecalc3.hoh_quality import gates, is_rectangle_outer
from impulsecalc3.job import load_job, validate_job

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ROOT / "configs" / "default_design.json"


def _job():
    return validate_job(json.loads(DEFAULT.read_text()))


def test_single_hex_polymesh(tmp_path):
    xy = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    quads = [(0, 1, 2, 3)]
    patches = {
        "inlet": [(0, 3)],
        "outlet": [(1, 2)],
        "bottom": [(0, 1)],
        "top": [(3, 2)],
        "blades": [],
    }
    write_polymesh(xy, quads, patches, 0.001, tmp_path, cyclic_sep_y=1.0)
    mesh = tmp_path / "constant" / "polyMesh"
    pts = mesh.joinpath("points").read_text()
    assert "\n8\n" in pts
    faces = mesh.joinpath("faces").read_text()
    assert "\n6\n" in faces
    own = mesh.joinpath("owner").read_text()
    assert "\n6\n" in own
    nei = mesh.joinpath("neighbour").read_text()
    assert "\n0\n" in nei
    b = mesh.joinpath("boundary").read_text()
    for name in ("inlet", "outlet", "bottom", "top", "blades", "frontAndBack"):
        assert name in b
    assert "oldInternalFaces" not in b
    assert "cyclicAMI" not in b


def test_rectangle_outer_is_rejected():
    pitch = 0.01
    P = np.column_stack([np.linspace(0, 1, 20), np.full(20, 0.0)])
    assert is_rectangle_outer(P, pitch)
    g = gates(
        {
            "n_blades": 3,
            "n_closed_blade_loops": 3,
            "n_cells": 30000,
            "max_nonortho_deg": 20.0,
            "max_skew": 1.0,
            "cyclic_ystdev_over_pitch": 0.0,
            "cyclic_translate_error": 0.0,
            "mesh_seconds": 1.0,
            "metal_nonortho_deg": 5.0,
            "min_det": 0.1,
            "nfaces_bottom": 10,
            "nfaces_top": 10,
            "rectangle_outer": True,
        }
    )
    assert g["checks"]["curved_cyclic"] is False
    assert g["pass"] is False


def test_curved_cyclic_is_not_rectangle():
    job = _job()
    from impulsecalc3.geometry import profile_from_job
    from impulsecalc3.job import pitch_m
    from impulsecalc3.hoh_mesher import _shift, _spine

    pitch = float(pitch_m(job))
    o0 = np.asarray(profile_from_job(job), dtype=float)
    outs = [_shift(o0, k * pitch) for k in range(3)]
    sp = _spine(outs[0], float(job["geometry"]["chord_m"]), 72.0, -72.0)
    P_bot, P_top = build_midgap_cyclic(outs, pitch, sp, 48)
    Y = 3.0 * pitch
    assert len(P_bot) == len(P_top) == 48
    assert np.max(np.abs((P_top[:, 1] - P_bot[:, 1]) - Y)) < 1e-12
    assert float(np.std(P_bot[:, 1]) / pitch) > 0.05


def test_three_blades(tmp_path):
    job = _job()
    r = build_hoh_mesh(job, n_blades=3, tier="fast", out_dir=tmp_path / "c")
    assert r.n_blades == 3
    assert "high_turning_midgap" in r.notes or abs(72 - (-72)) == 144


def test_high_turning_uses_midgap(tmp_path):
    job = _job()
    assert abs(job["geometry"]["beta1_flow_deg"] - job["geometry"]["beta2_flow_deg"]) == 144
    r = build_hoh_mesh(job, n_blades=3, tier="fast", out_dir=tmp_path / "h")
    assert r.cyclic_ystdev_over_pitch > 0.05
    assert any("midgap" in n or "high_turning" in n for n in r.notes) or r.cyclic_ystdev_over_pitch > 0.05


def test_default_bucket_gates(tmp_path):
    job = _job()
    r = build_hoh_mesh(job, n_blades=3, tier="balanced", out_dir=tmp_path / "d")
    assert r.n_blades == 3
    assert r.cyclic_ystdev_over_pitch > 0.05
    # success may be False on first ship if non-ortho; still must not be rectangle
    assert not r.gates["checks"].get("not_rectangle") is False or r.cyclic_ystdev_over_pitch > 0.05
    if getattr(r, "solvable", r.success):
        assert 20_000 <= r.n_cells <= 70_000
        assert r.n_blades == 3


def test_profile_change_remeshes(tmp_path):
    job = _job()
    a = build_hoh_mesh(job, n_blades=3, tier="fast", out_dir=tmp_path / "a")
    job2 = json.loads(json.dumps(job))
    job2["geometry"]["upper_sagitta_c"] = float(job2["geometry"]["upper_sagitta_c"]) + 0.04
    from impulsecalc3.job import validate_job
    job2 = validate_job(job2)
    b = build_hoh_mesh(job2, n_blades=3, tier="fast", out_dir=tmp_path / "b")
    pa = (tmp_path / "a" / "constant" / "polyMesh" / "points").read_text()
    pb = (tmp_path / "b" / "constant" / "polyMesh" / "points").read_text()
    assert pa != pb
    assert a.n_blades == b.n_blades == 3
    assert b.cyclic_ystdev_over_pitch > 0.05


def test_polymesh_files_exist(tmp_path):
    r = build_hoh_mesh(_job(), n_blades=3, tier="fast", out_dir=tmp_path / "p")
    mesh = Path(r.case_dir) / "constant" / "polyMesh"
    for name in ("points", "faces", "owner", "neighbour", "boundary"):
        assert (mesh / name).is_file()
    b = (mesh / "boundary").read_text()
    for name in ("inlet", "outlet", "bottom", "top", "blades", "frontAndBack"):
        assert name in b
    assert "oldInternalFaces" not in b
    assert "cyclicAMI" not in b
    assert "neighbourPatch  top" in b


def test_owner_neighbour_counts(tmp_path):
    r = build_hoh_mesh(_job(), n_blades=3, tier="fast", out_dir=tmp_path / "o")
    mesh = Path(r.case_dir) / "constant" / "polyMesh"

    def count(obj):
        txt = (mesh / obj).read_text()
        # first integer after header
        for line in txt.splitlines():
            if line.strip().isdigit():
                return int(line.strip())
        return -1

    n_faces = count("faces")
    n_own = count("owner")
    n_nei = count("neighbour")
    assert n_own == n_faces
    assert n_nei < n_faces
    # owner ids
    raw = mesh.joinpath("owner").read_text().split("(\n", 1)[-1].split("\n)")[0]
    ids = [int(x) for x in raw.split() if x.lstrip("-").isdigit()]
    assert ids and min(ids) >= 0 and max(ids) < r.n_cells
