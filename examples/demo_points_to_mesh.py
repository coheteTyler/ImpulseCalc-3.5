#!/usr/bin/env python3
"""Blade surface points -> labelled 2D cascade mesh (OpenFOAM polyMesh).

Demonstrates the ImpulseCalc3 cascade mesher from raw blade geometry:

  1. Geometry: upper and lower surface point lists (CSV, x,y columns, each
     ordered leading edge -> trailing edge), a unit scale into metres, and the
     blade pitch, given either directly (--pitch) or as number of blades Z at
     a mean radius r_m (pitch = 2*pi*r_m/Z, via impulsecalc3.job.apply_packing).
  2. Resolution: --n-around (cells around the blade O-ring), --n-radial
     (wall-normal layers) and --growth (wall-layer growth ratio, cfd.stretch).
  3. Output: an OpenFOAM case under <out>/openfoam_cases/demo_points with
     patches inlet (patch), outlet (patch), bottom/top (translational cyclic
     pair, one pitch apart = periodic), blade0 (wall), frontAndBack (empty).

The two surfaces are joined into one closed loop (upper LE->TE, then lower
TE->LE, closed back to the first point) and passed as geometry.profile_points
to the existing mesher. Gas/engine/rotor come from configs/geom_points.json;
the cfd (mesh) block is the live tuned one (LIVE_CFD below, copied from the
app's knobs_preview job) unless --cfd-from is given. CLI knobs override it
only when passed. This script only builds the job and calls
impulsecalc3.run.run_job in mesh-only mode (renumberMesh + checkMesh
-meshQuality, no solver).

First-cell height: there is no explicit first-cell knob. A profile_points
blade whose metal is a down-opening cup (mesh.profile_has_cavity) is meshed
on the high-def Goldman O-H path: wall first cell y1 = yplus_target * mu /
(rho * u_tau_frac_w1 * W1) (impulsecalc3.oh_shock.y1_wall_m), n_radial held in
15..25, and the O-collar thickness d_o grown from y1 at --growth (cfd.stretch,
capped 1.25). Use --yplus-target to move y1. A non-cup profile_points blade
falls back to the padded-AABB O-grid, where the first cell is derived from
d_o, n_radial and growth instead. The achieved first cell is printed from the
mesh report.

Example (mm input):
  .venv/bin/python examples/demo_points_to_mesh.py \
      --upper examples/demo_upper.csv --lower examples/demo_lower.csv \
      --scale 1e-3 --n-blades 25 --mean-radius 0.0375 --out output/demo_points
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE_JOB = ROOT / "configs" / "geom_points.json"
JOB_NAME = "demo_points"

# Live tuned cfd block, copied from output/geom_tests/knobs_preview/knobs_preview.json
# ['cfd'] (smoke_end_s dropped: mesh only). Embedded so a fresh clone meshes with the
# same knobs as the live app. configs/geom_points.json carries a stale cfd block
# (n_pitch_fill 7, n_radial 12, stretch 1.25, ...) and is used for gas/engine only.
LIVE_CFD: dict = {
    "solver": "rhoCentralFoam",
    "openfoam": "ESI-v2412",
    "openfoam_bashrc": "/usr/lib/openfoam/openfoam2412/etc/bashrc",
    "wall": "noSlip",
    "turbulence": "laminar",
    "n_chords_min": 5.0,
    "mesh": "body_fitted_OH",
    "z_thick_m": 0.001,
    "n_around": 56,
    "n_radial": 20,
    "n_inlet": 10,
    "n_outlet": 28,
    "n_cyclic": 16,
    "x_up_c": 0.95,
    "x_dn_c": 2.5,
    "outlet_p": "waveTransmissive",
    "max_co": 0.15,
    "stretch": 1.12,
    "n_pitch_fill": 40,
    "inlet_bc": "static_rel",
    "inlet_stretch": 1.12,
    "le_cluster": 2.5,
    "n_le": 14,
    "h_le": None,
    "h_pass": None,
    "h_far": None,
    "growth": 1.25,
    "x_dense_c": 0.25,
    "n_pitchwise_throat": 40,
    "dump_rx": 1.18,
    "dump_dx_last_m": 0.0014,
    "wake_cx": 1.0,
    "yplus_target": 1.0,
    "u_tau_frac_w1": 0.05,
    "te_angular_min": 10,
    "stack_after_child_ok": False,
}


def load_cfd(path: Path | None) -> dict:
    """Base cfd block: embedded LIVE_CFD, or ['cfd'] (or the whole dict) of a JSON file."""
    if path is None:
        cfd = copy.deepcopy(LIVE_CFD)
    else:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        cfd = copy.deepcopy(data["cfd"] if isinstance(data.get("cfd"), dict) else data)
    cfd.pop("smoke_end_s", None)  # mesh only; no endTime patching
    cfd["outlet_p"] = "waveTransmissive"
    return cfd


def read_xy_csv(path: Path) -> list[tuple[float, float]]:
    """Read (x, y) from the first two columns of a CSV. A text header row is skipped."""
    pts: list[tuple[float, float]] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if not row or not row[0].strip() or row[0].lstrip().startswith("#"):
                continue
            try:
                pts.append((float(row[0]), float(row[1])))
            except (ValueError, IndexError):
                if pts:
                    raise ValueError(f"{path}: bad row {row!r}")
                continue  # header
    if len(pts) < 2:
        raise ValueError(f"{path}: need at least 2 points, got {len(pts)}")
    return pts


def _same(a: tuple[float, float], b: tuple[float, float], tol: float) -> bool:
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


def closed_loop(
    upper: list[tuple[float, float]],
    lower: list[tuple[float, float]],
    scale: float,
) -> list[list[float]]:
    """Upper (LE->TE) + reversed lower (TE->LE), scaled to metres, first point repeated last.

    A shared TE point is kept once; a shared LE point becomes the closing point.
    Orientation is not forced here: profile_from_points reverses a CW loop to CCW.
    """
    up = [(x * scale, y * scale) for x, y in upper]
    lo = [(x * scale, y * scale) for x, y in lower]
    xs = [p[0] for p in up + lo]
    tol = 1e-9 * max(max(xs) - min(xs), 1e-30)
    loop = list(up)
    lo_rev = lo[::-1]
    if _same(loop[-1], lo_rev[0], tol):
        lo_rev = lo_rev[1:]
    loop.extend(lo_rev)
    if _same(loop[-1], loop[0], tol):
        loop[-1] = loop[0]
    else:
        loop.append(loop[0])
    return [[float(x), float(y)] for x, y in loop]


def build_job(args: argparse.Namespace, loop: list[list[float]], out_dir: Path) -> dict:
    job = json.loads(BASE_JOB.read_text(encoding="utf-8"))
    job = copy.deepcopy(job)
    job["name"] = JOB_NAME
    job["output_dir"] = str(out_dir)
    job["article"] = "DEMO: blade surface points -> labelled 2D cascade mesh (examples/demo_points_to_mesh.py)"
    job["predicted_reason"] = "Mesh demonstration only. Gas state copied from configs/geom_points.json; no solve."
    job["predicted"] = True
    job["geometry_test"] = True

    g = job["geometry"]
    g["profile_family"] = "profile_points"
    g["profile_points"] = loop
    g["n_profile_points"] = len(loop) - 1
    # Chord = straight LE->TE distance of the input (upper[0] -> upper[-1]).
    le = (args.upper_pts[0][0] * args.scale, args.upper_pts[0][1] * args.scale)
    te = (args.upper_pts[-1][0] * args.scale, args.upper_pts[-1][1] * args.scale)
    g["chord_m"] = math.hypot(te[0] - le[0], te[1] - le[1])
    g.pop("pitch_m", None)
    g.pop("solidity", None)
    if args.pitch is not None:
        g["pitch_m"] = float(args.pitch)
        g["packing_driver"] = "pitch"
        if args.mean_radius is not None:
            g["mean_radius_m"] = float(args.mean_radius)
    else:
        g["n_blades_machine"] = int(args.n_blades)
        g["mean_radius_m"] = float(args.mean_radius)
        g["packing_driver"] = "z"
    # Keep hub/tip consistent with the (possibly new) mean radius; span unchanged.
    span = float(g.get("span_m") or 0.005)
    rm = float(g["mean_radius_m"])
    g["rotor_tip_radius_m"] = rm + 0.5 * span
    g["hub_radius_m"] = rm - 0.5 * span
    g["note"] = (
        "DEMO metal: closed profile_points built from --upper/--lower CSVs "
        f"(scale {args.scale:g} -> m)."
    )

    job["cfd"] = load_cfd(args.cfd_from)
    cfd = job["cfd"]
    if args.n_around is not None:
        cfd["n_around"] = int(args.n_around)
    if args.n_radial is not None:
        cfd["n_radial"] = int(args.n_radial)
    if args.growth is not None:
        cfd["stretch"] = float(args.growth)
    if args.yplus_target is not None:
        cfd["yplus_target"] = float(args.yplus_target)
    return job


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Blade upper/lower surface points -> labelled 2D cascade mesh (mesh only, no solve).",
    )
    p.add_argument("--upper", required=True, type=Path, help="CSV x,y of the upper surface, LE -> TE")
    p.add_argument("--lower", required=True, type=Path, help="CSV x,y of the lower surface, LE -> TE")
    p.add_argument("--scale", type=float, default=1.0,
                   help="multiplies CSV coordinates into metres (1e-3 for mm). Default 1.0")
    pk = p.add_argument_group("blade pitch (give --pitch, or --n-blades with --mean-radius)")
    pk.add_argument("--n-blades", type=int, help="number of blades Z on the wheel")
    pk.add_argument("--mean-radius", type=float, help="mean radius r_m [m]; pitch = 2*pi*r_m/Z")
    pk.add_argument("--pitch", type=float, help="blade pitch at mean radius [m] (overrides Z / r_m)")
    p.add_argument("--cfd-from", type=Path, default=None,
                   help="JSON whose ['cfd'] block replaces the embedded live knobs "
                        "(e.g. output/geom_tests/knobs_preview/knobs_preview.json)")
    rs = p.add_argument_group("resolution (omit to keep the live tuned knobs)")
    rs.add_argument("--n-around", type=int, help="cells around the blade O-ring (cfd.n_around)")
    rs.add_argument("--n-radial", type=int, help="wall-normal O-grid layers (cfd.n_radial)")
    rs.add_argument("--growth", type=float, help="wall-layer growth ratio (cfd.stretch)")
    rs.add_argument("--yplus-target", type=float,
                    help="cfd.yplus_target: sets the wall first cell y1 on the high-def (cup) path; "
                         "no explicit first-cell knob exists")
    p.add_argument("--out", required=True, type=Path, help="output directory (case goes in <out>/openfoam_cases/demo_points)")
    a = p.parse_args(argv)
    if a.pitch is None and (a.n_blades is None or a.mean_radius is None):
        p.error("give --pitch, or both --n-blades and --mean-radius")
    if a.scale <= 0:
        p.error("--scale must be > 0")
    return a


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sys.path.insert(0, str(ROOT))
    from impulsecalc3.geometry import profile_from_points
    from impulsecalc3.run import run_job

    args.upper_pts = read_xy_csv(args.upper)
    args.lower_pts = read_xy_csv(args.lower)
    loop = closed_loop(args.upper_pts, args.lower_pts, args.scale)
    profile_from_points(loop)  # fail loud: open, self-intersecting or zero-area loop

    out_dir = args.out if args.out.is_absolute() else (Path.cwd() / args.out)
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    job = build_job(args, loop, out_dir)
    job_path = out_dir / f"{JOB_NAME}_job.json"
    job_path.write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
    print(f"job written: {job_path}")

    rep = run_job(job_path, skip_solve=True, run_solve=False, run_post=False)

    mesh = rep.get("mesh") or {}
    chk = rep.get("checkMesh") or {}
    g = json.loads(job_path.read_text(encoding="utf-8"))["geometry"]
    summary = {
        "case_dir": rep.get("case_dir"),
        "n_cells": mesh.get("n_cells"),
        "mesh_kind": mesh.get("kind"),
        "first_cell_m": mesh.get("first_cell_m"),
        "patches": mesh.get("patches"),
        "checkMesh_failed_checks": chk.get("failed_checks"),
        "checkMesh_note": chk.get("note"),
        "errors": rep.get("errors"),
        "predicted": True,
        "input_n_points": len(loop) - 1,
        "chord_m": g.get("chord_m"),
    }
    print(json.dumps(summary, indent=2, default=str))
    ok = bool(mesh.get("n_cells")) and not int(chk.get("failed_checks") or 0) and bool(chk.get("mesh_ok_strict"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
