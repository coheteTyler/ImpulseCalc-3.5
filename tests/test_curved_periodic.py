"""Curved-periodic O/H mesher: writer unit test (one legal hex) + builder invariants.

These tests write only into pytest tmp dirs (never output/geom_tests/knobs_preview).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from impulsecalc3.curved_periodic import build_curved_periodic, write_polymesh_2d
from impulsecalc3.ofenv import BASHRC, openfoam_available

ROOT = Path(__file__).resolve().parents[1]


def _read_list_len(p: Path) -> int:
    txt = p.read_text()
    body = txt.split("// * * *")[-1] if "// * * *" in txt else txt
    for line in body.splitlines():
        s = line.strip()
        if s.isdigit():
            return int(s)
    raise AssertionError(f"no count in {p}")


def _one_hex(case: Path) -> dict:
    xy = np.array([[0.0, 0.0], [1e-3, 0.0], [1e-3, 1e-3], [0.0, 1e-3]])
    quads = np.array([[0, 1, 2, 3]])
    patches = {"bottom": [(0, 1)], "outlet": [(1, 2)], "top": [(2, 3)], "inlet": [(3, 0)]}
    return write_polymesh_2d(case, xy, quads, patches, 1e-3, patch_types={"bottom": "wall", "top": "wall"})


def test_writer_emits_one_legal_hex(tmp_path):
    case = tmp_path / "hex1"
    info = _one_hex(case)
    pm = case / "constant" / "polyMesh"
    assert info["n_points"] == 8
    assert info["n_cells"] == 1
    assert info["n_faces"] == 6
    assert info["n_internal"] == 0
    assert _read_list_len(pm / "points") == 8
    assert _read_list_len(pm / "faces") == 6
    assert _read_list_len(pm / "owner") == 6


@pytest.mark.skipif(not (openfoam_available() and BASHRC.is_file()), reason="native OpenFOAM v2412 not installed")
def test_one_hex_checkmesh_ok(tmp_path):
    case = tmp_path / "hex1"
    _one_hex(case)
    sysd = case / "system"
    sysd.mkdir(parents=True, exist_ok=True)
    (sysd / "controlDict").write_text(
        "FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }\n"
        "application checkMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1;\n"
        "deltaT 1; writeControl timeStep; writeInterval 1;\n"
    )
    (sysd / "fvSchemes").write_text(
        "FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }\n"
        "ddtSchemes { default Euler; } gradSchemes { default Gauss linear; }\n"
        "divSchemes { default none; } laplacianSchemes { default Gauss linear corrected; }\n"
        "interpolationSchemes { default linear; } snGradSchemes { default corrected; }\n"
    )
    (sysd / "fvSolution").write_text(
        "FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\n"
    )
    out = subprocess.run(
        ["bash", "-lc", f"set +u; . {BASHRC} >/dev/null 2>&1; checkMesh -case {case}"],
        capture_output=True, text=True, check=False,
    )
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    assert "Mesh OK." in out.stdout, out.stdout[-3000:]


LOOP = Path("/workspace/rotor_stage/loop.json")


@pytest.mark.skipif(not LOOP.is_file(), reason="rotor loop points not staged on this machine")
def test_curved_builder_invariants():
    from impulsecalc3.geometry import profile_from_points

    loop = profile_from_points(json.loads(LOOP.read_text()))
    s = 0.00632246
    r = build_curved_periodic(loop, pitch=s, chord=0.01, x_in=-0.0095, x_out=0.035,
                              y1=6.063e-7, growth=1.12, n_wall=20)
    xy = r.xy
    bot = sorted({i for e in r.patches["bottom"] for i in e}, key=lambda i: xy[i, 0])
    top = sorted({i for e in r.patches["top"] for i in e}, key=lambda i: xy[i, 0])
    assert len(bot) == len(top)
    d = xy[top] - xy[bot]
    assert np.max(np.abs(d[:, 0])) <= 1e-12
    assert np.max(np.abs(d[:, 1] - s)) <= 1e-12
    assert r.metrics["periodic_clearance_min_m"] >= r.metrics["O_thickness_m"]
    assert r.metrics["n_cells"] < 70000
    # all quads strictly positive (CCW) area
    P = xy[r.quads]
    a = 0.5 * np.sum(P[:, :, 0] * np.roll(P[:, :, 1], -1, 1) - np.roll(P[:, :, 0], -1, 1) * P[:, :, 1], axis=1)
    assert a.min() > 0
