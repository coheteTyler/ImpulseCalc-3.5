"""Write case, checkMesh, rhoCentralFoam, parse forces + wall p(s), viz, JSON report.

Fail loud. No cartoon JSON. No eta from CFD. No synthetic Cp. predicted:true locked.
"""

from __future__ import annotations

import argparse
import sys
import json
import math
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .case import write_case
from .forces import load_blade_forces
from .geometry import BladeSpec, profile_from_job, spec_from_job
from .job import (
    DEFAULT_PATH,
    domain_x,
    is_live_marlin_job,
    load_job,
    remesh_live_marlin_allowed,
)
from .meanline import compute_meanline
from .ntrs_checks import evaluate as ntrs_evaluate
from .ofenv import foam_env, openfoam_available, run_foam
from .sample import load_wall_pressure
from .times import compute_times


def _spec(job: dict[str, Any], ml) -> BladeSpec:
    spec = spec_from_job(job)
    spec.beta1_metal_deg = float(ml.beta1_metal_deg)
    spec.beta2_metal_deg = float(ml.beta2_metal_deg)
    return spec


def _latest_time(case_dir: Path) -> float:
    latest = 0.0
    for p in case_dir.iterdir():
        if not p.is_dir():
            continue
        try:
            t = float(p.name)
        except ValueError:
            continue
        if (p / "p").is_file():
            latest = max(latest, t)
    return latest


def _strip_eta(d: dict[str, Any]) -> dict[str, Any]:
    banned = ("eta_design_proxy", "force_synthetic", "allow_synthetic", "eta_cfd", "eta")
    return {k: v for k, v in d.items() if k not in banned}


class FoamLogMissing(RuntimeError):
    """checkMesh/rhoCentralFoam log missing — command never started or case wiped."""


def read_foam_log(case_dir: Path, name: str, command: str) -> str:
    """Read a foam log. Never FileNotFoundError — fail loud with a sentence."""
    p = Path(case_dir) / name
    if not p.is_file():
        raise FoamLogMissing(
            f"{command} never started or case was wiped mid-run (missing {name})."
        )
    return p.read_text(encoding="utf-8", errors="replace")


def run_job(
    job_path: str | Path | None = None,
    *,
    skip_solve: bool = False,
    post_only: bool = False,
    run_mesh: bool = True,
    run_solve: bool | None = None,
    run_post: bool = True,
    **_kwargs: Any,
) -> dict[str, Any]:
    if run_solve is False:
        skip_solve = True
    if post_only:
        skip_solve = True
    _ = run_mesh, run_post
    errors: list[str] = []
    flags = {
        "mesh_ok": False,
        "solve_ok": False,
        "sample_on_wall": False,
        "force_plateau": False,
        "force_climbing": True,
        "newtons_trusted": False,
        "eta_from_cfd": False,
        "synthetic": False,
        "predicted": True,
    }
    root = Path(__file__).resolve().parent.parent
    job = load_job(job_path)
    out_root = Path(job.get("output_dir", "output"))
    if not out_root.is_absolute():
        out_root = (root / out_root).resolve()
    else:
        out_root = out_root.resolve()
    case_dir = out_root / "openfoam_cases" / str(job["name"])
    fig_dir = out_root / "plots"
    out_root.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    ml = compute_meanline(job)
    g = job["geometry"]
    gas = job["gas"]
    xin, xout = domain_x(job)
    times = compute_times(
        chord_m=float(g["chord_m"]),
        w1_m_s=float(gas["w1_m_s"]),
        gamma=float(gas["gamma"]),
        r_specific=float(gas["r_specific_j_kg_k"]),
        t1_k=float(gas["t1_k"]),
        x_in_m=xin,
        x_out_m=xout,
        n_chords_min=float(job["cfd"].get("n_chords_min", 10.0)),
        n_lx_a_min=float(job["cfd"].get("n_lx_a_min", 1.2)),
    )
    spec = _spec(job, ml)
    p1 = float(gas["p1_pa"])
    q_dyn = 0.5 * ml.rho1_kg_m3 * float(gas["w1_m_s"]) ** 2

    def _has_live_times(d: Path) -> bool:
        if not d.is_dir():
            return False
        n = 0
        for pth in d.iterdir():
            if not pth.is_dir():
                continue
            try:
                ft = float(pth.name)
            except ValueError:
                continue
            if ft > 0 and (pth / "p").is_file():
                n += 1
        return n >= 1

    live_marlin = is_live_marlin_job(job, out_root)
    poly_exists = (case_dir / "constant" / "polyMesh" / "points").is_file()
    # --post-only: reuse. --skip-solve: remesh (knobs must write the mesh) unless
    # this is the live 80 µs Marlin tree, which we never remesh/re-solve here.
    if live_marlin and not remesh_live_marlin_allowed():
        if not skip_solve:
            raise RuntimeError(
                "refusing to remesh/re-solve the live Marlin 80 us case "
                "(output/marlin_v2_rotor_report.json). Use configs/geom_*.json or set "
                "IMPULSECALC3_REMESH_MARLIN=1"
            )
        reuse = bool(poly_exists)
        if reuse:
            mesh_notes_extra = [
                "live Marlin 80 us case reused (skip_solve); polyMesh not rewritten"
            ]
        else:
            reuse = False
            mesh_notes_extra = []
    else:
        mesh_notes_extra = []
        reuse = bool(
            post_only
            and _has_live_times(case_dir)
            and poly_exists
        )
        if (not skip_solve) and case_dir.exists():
            for child in list(case_dir.iterdir()):
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                elif child.is_file():
                    child.unlink()
        elif skip_solve and not post_only and case_dir.exists():
            # Remesh: drop leftover time dirs AND stale postProcessing so a 3-blade
            # sample cannot be scored against a 2-wall passage mesh.
            for child in list(case_dir.iterdir()):
                if child.is_dir():
                    try:
                        float(child.name)
                    except ValueError:
                        if child.name in ("postProcessing", "VTK"):
                            shutil.rmtree(child, ignore_errors=True)
                        continue
                    shutil.rmtree(child, ignore_errors=True)

    if reuse:
        from .geometry import center_in_pitch
        from .job import pitch_m
        from .mesh import MeshBuild

        meta_p = case_dir / "impulsecalc3_case_meta.json"
        meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.is_file() else {}
        pitch = pitch_m(job)
        poly0, y_shift = center_in_pitch(profile_from_job(job, spec), pitch)
        n_blades = int(job["geometry"]["n_blades_cascade"])
        blade_polys = [[(xy[0], xy[1] + k * pitch) for xy in poly0] for k in range(n_blades)]
        mesh = MeshBuild(
            n_cells=int(meta.get("mesh_cells", 0)),
            n_points=int(meta.get("n_points", 0) or 0),
            n_faces=0,
            patches=meta.get("patches") or {"blade0": 1, "blade1": 1, "blade2": 1},
            first_cell_m=float(meta.get("first_cell_m", 1e-4)),
            min_area_2d=0.0,
            check_notes=["reused existing live case (skip_solve); polyMesh not rewritten"],
            y_shift_m=y_shift,
            pitch_m=pitch,
            z_thick_m=float(job["cfd"]["z_thick_m"]),
            blade_polys=blade_polys,
            x_in=xin,
            x_out=xout,
            n_around=int(job["cfd"].get("n_around", 0) or 0),
            n_radial=int(job["cfd"].get("n_radial", 0) or 0),
            mesh_kind="body_fitted_OH",
        )
        t_end = float(meta.get("t_end_s", times.t_end_floor_s))
    else:
        mesh, t_end = write_case(case_dir, job, ml, times, spec)
    is_geom = bool(job.get("geometry_test"))
    smoke_s = (job.get("cfd") or {}).get("smoke_end_s")
    if is_geom and smoke_s is not None and not reuse:
        import re as _re
        t_smoke = float(smoke_s)
        cd = case_dir / "system" / "controlDict"
        txt = cd.read_text(encoding="utf-8")
        txt = _re.sub(r"endTime\s+\S+;", f"endTime         {t_smoke:.8g};", txt)
        txt = _re.sub(r"writeInterval\s+\S+;", f"writeInterval   {t_smoke:.8g};", txt)
        cd.write_text(txt, encoding="utf-8")
        t_end = t_smoke
        mesh.check_notes.append(
            f"geometry_test smoke: controlDict endTime patched to {t_smoke:.8g} s "
            f"(design floor {times.t_end_floor_s:.8g} s not required)"
        )
    if (not is_geom) and t_end + 1e-18 < times.t_end_floor_s:
        errors.append(f"endTime {t_end} below max(5*c/W1, 1.2*Lx/a) floor {times.t_end_floor_s}")

    env = foam_env() if openfoam_available() else None
    check = {
        "mesh_ok_strict": False,
        "rc": None,
        "log": str(case_dir / "log.checkMesh"),
        "note": "checkMesh not run",
    }
    solve = {"rc": None, "log": str(case_dir / "log.rhoCentralFoam"), "note": "not run"}

    if env is None:
        errors.append("OpenFOAM ESI v2412 not available")
    else:
        # upperTriangularFace is a checkMesh fail on our hex dump; Cuthill-McKee is the fix.
        # Never renumber on skip_solve: time-dir fields would map to the wrong cells.
        if not reuse:
            run_foam(["renumberMesh", "-overwrite"], cwd=case_dir, log_name="log.renumberMesh", env=env)
        rc_m = run_foam(["checkMesh"], cwd=case_dir, log_name="log.checkMesh", env=env)
        try:
            log_m = read_foam_log(case_dir, "log.checkMesh", "checkMesh")
        except FoamLogMissing as exc:
            msg = str(exc)
            errors.append(msg)
            check = {
                "mesh_ok_strict": False,
                "rc": int(rc_m),
                "log": str(case_dir / "log.checkMesh"),
                "note": msg,
                "failed_checks": None,
            }
            flags["mesh_ok"] = False
            log_m = ""
            _mesh_log_missing = True
        else:
            _mesh_log_missing = False
        from viewer.ofio import parse_checkmesh

        if _mesh_log_missing:
            mesh_usable = False
        else:
            check = parse_checkmesh(log_m)
            check["rc"] = int(rc_m)
            check["log"] = str(case_dir / "log.checkMesh")
            check["kind"] = mesh.mesh_kind
            nfail = int(check.get("failed_checks") or 0)
            flags["mesh_ok"] = bool(check.get("mesh_ok_strict")) and nfail == 0
            vol_ok = (not check.get("open_cells")) and not check.get("negative_volume") and (
                check.get("min_volume") is None or float(check["min_volume"]) > 0
            )
            if not vol_ok:
                errors.append("checkMesh: negative/open cells")
            if nfail:
                errors.append(f"checkMesh failed {nfail} checks (hard fail; not a painted cascade)")
            if check.get("wrong_oriented_faces"):
                check["note"] = (
                    f"{check['wrong_oriented_faces']} pyramid-orientation flags; "
                    "not subsetMesh stairs."
                )
            elif nfail:
                check["note"] = f"checkMesh failed {nfail} quality checks"
            else:
                check["note"] = "Mesh OK (strict)"
            mesh_usable = bool(flags["mesh_ok"])
        if not skip_solve and mesh_usable:
            rc_s = run_foam(["rhoCentralFoam"], cwd=case_dir, log_name="log.rhoCentralFoam", env=env)
            try:
                log_s = read_foam_log(case_dir, "log.rhoCentralFoam", "rhoCentralFoam")
            except FoamLogMissing as exc:
                msg = str(exc)
                errors.append(msg)
                solve = {
                    "rc": int(rc_s),
                    "log": str(case_dir / "log.rhoCentralFoam"),
                    "end_seen": False,
                    "fatal": True,
                    "note": msg,
                }
                flags["solve_ok"] = False
                log_s = ""
            else:
                solve = {
                    "rc": int(rc_s),
                    "log": str(case_dir / "log.rhoCentralFoam"),
                    "end_seen": "End" in log_s[-4000:] or "Finalising" in log_s[-4000:],
                    "fatal": "FOAM FATAL" in log_s or "Foam::error" in log_s,
                }
                flags["solve_ok"] = int(rc_s) == 0 and not solve["fatal"]
                if not flags["solve_ok"]:
                    errors.append(f"rhoCentralFoam rc={rc_s}")
            # foamToVTK if present
            vtk_dir = case_dir / "VTK"
            if shutil.which("foamToVTK", path=env.get("PATH", "")):
                run_foam(["foamToVTK"], cwd=case_dir, log_name="log.foamToVTK", env=env)
                solve["foamToVTK"] = str(vtk_dir) if vtk_dir.exists() else None
            # y+ is the in-run FO at last writeTime (postProcessing/yPlus1).
            # Do NOT run `postProcess -func yPlus`: no turbulence model in the
            # database, so it overwrites the last-time field with zeros.
        elif skip_solve:
            lp = case_dir / "log.rhoCentralFoam"
            log_s = lp.read_text(encoding="utf-8", errors="replace") if lp.is_file() else ""
            t_have = _latest_time(case_dir)
            fatal = "FOAM FATAL" in log_s or "Foam::error" in log_s
            solve = {
                "rc": 0 if t_have > 0 and not fatal else None,
                "note": "skip_solve: reused existing time directories",
                "log": str(lp),
                "end_seen": ("End" in log_s[-4000:] or "Finalising" in log_s[-4000:]) if log_s else False,
                "fatal": fatal,
            }
            flags["solve_ok"] = t_have > 0 and not fatal
            vtk_dir = case_dir / "VTK"
            if shutil.which("foamToVTK", path=env.get("PATH", "")):
                run_foam(["foamToVTK"], cwd=case_dir, log_name="log.foamToVTK", env=env)
                solve["foamToVTK"] = str(vtk_dir) if vtk_dir.exists() else None
            # y+ from in-run FO last writeTime. Do not postProcess-zero the field.

    t_reached = _latest_time(case_dir)
    n_chords = t_reached / times.t_chord_convective_s if times.t_chord_convective_s else 0.0
    wall_is_noslip = str(job["cfd"].get("wall", "noSlip")) == "noSlip"

    forces = load_blade_forces(
        case_dir,
        span_m=float(g["span_m"]),
        z_thick_m=mesh.z_thick_m,
        t_chord_s=times.t_chord_convective_s,
        n_chords_run=n_chords,
        wall_is_noslip=wall_is_noslip,
    )
    flags["force_plateau"] = bool(forces.get("plateau"))
    flags["force_climbing"] = bool(forces.get("climbing", True))
    if forces.get("error") and not is_geom:
        errors.append(str(forces["error"]))

    wall = load_wall_pressure(
        case_dir,
        p1_pa=p1,
        chord_m=float(g["chord_m"]),
        q_dyn_pa=q_dyn,
        blade_polys=mesh.blade_polys,
        first_cell_m=mesh.first_cell_m,
    )
    flags["sample_on_wall"] = bool(wall.get("on_wall"))
    if not flags["sample_on_wall"] and not is_geom:
        errors.append(wall.get("error") or "sample not on wall")

    # yPlus + outlet dump
    log_s_txt = ""
    lp = case_dir / "log.rhoCentralFoam"
    if lp.is_file():
        log_s_txt = lp.read_text(encoding="utf-8", errors="replace")
    from viewer.ofio import parse_yplus, patch_mean_from_field, time_dirs, cell_centres, read_scalar_field, read_vector_field

    yplus = parse_yplus(case_dir, log_s_txt)
    outlet_p = None
    tds = time_dirs(case_dir)
    if tds:
        outlet_p = patch_mean_from_field(tds[-1][1] / "p", "outlet")
    outlet = {
        "type": job["cfd"].get("outlet_p"),
        "fieldInf": p1,
        "NOT": "0.95 p1",
        "measured_mean_p": outlet_p,
        "measured_over_p1": (outlet_p / p1) if outlet_p else None,
    }

    # Viz from real time dirs. Do not paint a wedge that failed checkMesh.
    viz: dict[str, Any] = {"pngs": [], "note": "matplotlib from real OF time dirs; no fake encoder"}
    try:
        from viewer.plots import field_png, force_history_png, sequence_from_times, wall_cp_png

        if not flags.get("mesh_ok"):
            viz["note"] = "field contours withheld: checkMesh hard fail (not a cascade)"
            preview = case_dir / "mesh_preview.png"
            if preview.is_file():
                viz["pngs"].append(str(preview))
            raise RuntimeError("skip field paint")
        cc = cell_centres(case_dir) if tds else None
        if tds and cc is not None:
            last = tds[-1][1]
            pv = read_scalar_field(last / "p", cc.shape[0])
            uv = read_vector_field(last / "U", cc.shape[0])
            if pv:
                png = field_png(
                    fig_dir / "p_latest.png",
                    cc,
                    __import__("numpy").array(pv, dtype=float),
                    title=f"p  t={tds[-1][0]:.4g} s",
                    cbar="p [Pa]",
                    blade_polys=mesh.blade_polys,
                )
                if png:
                    viz["pngs"].append(str(png))
            if uv:
                mag = __import__("numpy").array(
                    [(v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5 for v in uv], dtype=float
                )
                png = field_png(
                    fig_dir / "Umag_latest.png",
                    cc,
                    mag,
                    title=f"|U|  t={tds[-1][0]:.4g} s",
                    cbar="|U| [m/s]",
                    blade_polys=mesh.blade_polys,
                )
                if png:
                    viz["pngs"].append(str(png))
            seq = sequence_from_times(fig_dir, case_dir, cc, mesh.blade_polys, p1)
            viz["sequence"] = seq
            viz["pngs"].extend(seq)
        wpng = wall_cp_png(
            fig_dir / "wall_cp_blade0.png",
            wall,
            p1_pa=p1,
            q_dyn_pa=q_dyn,
        )
        if wpng:
            viz["pngs"].append(str(wpng))
        fpng = force_history_png(fig_dir / "ft_history.png", forces)
        if fpng:
            viz["pngs"].append(str(fpng))
        preview = case_dir / "mesh_preview.png"
        if preview.is_file():
            viz["pngs"].append(str(preview))
    except Exception as exc:
        viz["error"] = f"{type(exc).__name__}: {exc}"
    try:
        from .post import estimate_yplus, write_plots

        if not flags.get("mesh_ok"):
            for name in (
                "contour_p.png", "contour_U.png", "contour_stream.png",
                "contour_M.png", "contour_shock.png", "contour_T.png", "p_latest.png",
                "Umag_latest.png",
            ):
                fp = fig_dir / name
                if fp.is_file():
                    fp.unlink()
            extra = {}
        else:
            extra = write_plots(fig_dir, case_dir, job, forces, wall, ml.to_dict())
        viz["post_plots"] = extra
        if isinstance(extra, dict):
            viz["pngs"].extend(str(v) for v in extra.values() if isinstance(v, str) and v.endswith(".png"))
        if not yplus or yplus.get("average") is None:
            yplus = estimate_yplus(
                case_dir,
                first_cell_m=mesh.first_cell_m,
                rho=ml.rho1_kg_m3,
                mu=float(gas["mu_pa_s"]),
                forces=forces,
                z_thick_m=mesh.z_thick_m,
            )
    except Exception as exc:
        viz["post_error"] = f"{type(exc).__name__}: {exc}"

    # Newtons trusted only if plateau AND 5 chords AND on wall AND solve ok.
    # Inlet state is unsigned ⇒ predicted always remains True.
    climbing_at_end = flags["force_climbing"]
    short = n_chords < 4.99
    is_geom = bool(job.get("geometry_test"))
    if climbing_at_end and not is_geom:
        errors.append("force still climbing at endTime — do not treat Newtons as closed")
    if short and t_reached > 0 and not is_geom:
        errors.append(f"t_reached={t_reached:.4g} s is only {n_chords:.3g} chords (< 5)")

    flags["newtons_trusted"] = (
        (not is_geom)
        and flags["solve_ok"]
        and flags["sample_on_wall"]
        and flags["force_plateau"]
        and flags["mesh_ok"]
        and not climbing_at_end
        and not short
        and not flags["synthetic"]
    )
    # Unsigned GG station: even a plateau is PREDICTED.
    predicted = True

    if is_geom:
        # Geometry test: no 8-chord plateau. mesh_ok still tracks failed_checks.
        success = flags["mesh_ok"] and flags["solve_ok"] and not errors
    else:
        success = (
            flags["mesh_ok"]
            and flags["solve_ok"]
            and flags["sample_on_wall"]
            and not climbing_at_end
            and not short
            and not errors
        )
    # mesh_ok already tracks failed_checks. Inverted/open cells also kill success.
    if check.get("open_cells") or check.get("negative_volume") or int(check.get("failed_checks") or 0) > 0:
        success = False
        flags["mesh_ok"] = bool(check.get("mesh_ok_strict")) and int(check.get("failed_checks") or 0) == 0
        flags["newtons_trusted"] = False

    blades_out = []
    for b in forces.get("blades") or []:
        rec = dict(b)
        rec["trusted"] = bool(flags["newtons_trusted"])
        rec["predicted"] = True
        rec["climbing"] = flags["force_climbing"]
        blades_out.append(rec)

    report = {
        "app": "ImpulseCalc3",
        "name": job["name"],
        "article": job.get("article"),
        "success": bool(success),
        "predicted": predicted,
        "predicted_reason": job.get("predicted_reason"),
        "flags": flags,
        "errors": errors,
        "times": {
            "t_end_s": t_end,
            "t_end_floor_s": times.t_end_floor_s,
            "t_chord_convective_s": times.t_chord_convective_s,
            "t_chord_acoustic_s": times.t_chord_acoustic_s,
            "t_domain_acoustic_s": times.t_domain_acoustic_s,
            "t_reached_s": t_reached,
            "n_chords": n_chords,
            "n_chords_min": float(job["cfd"].get("n_chords_min", 5.0)),
            "Mw1": times.Mw1,
            "notes": times.notes,
        },
        "mesh": {
            "kind": mesh.mesh_kind,
            "n_cells": mesh.n_cells,
            "n_points": mesh.n_points,
            "n_around": mesh.n_around,
            "n_radial": mesh.n_radial,
            "first_cell_m": mesh.first_cell_m,
            "min_area_2d": mesh.min_area_2d,
            "z_thick_m": mesh.z_thick_m,
            "patches": mesh.patches,
            "notes": mesh.check_notes,
        },
        "checkMesh": check,
        "solve": solve,
        "wall": {
            "on_wall": wall.get("on_wall"),
            "success": wall.get("success"),
            "p1_pa": p1,
            "t0_mean_p": wall.get("t0_mean_p"),
            "mean_p": (wall["blades"][0]["mean_p"] if wall.get("blades") else None),
            "mean_p_over_p1": (
                (wall["blades"][0]["mean_p"] / p1) if wall.get("blades") else None
            ),
            "q_dyn_pa": q_dyn,
            "parser": wall.get("parser"),
            "notes": wall.get("notes"),
            "error": wall.get("error"),
            "n_points": [b.get("n_points") for b in wall.get("blades") or []],
        },
        "yplus": yplus,
        "forces": {
            "predicted": True,
            "climbing": flags["force_climbing"],
            "plateau": flags["force_plateau"],
            "trusted": flags["newtons_trusted"],
            "frame": forces.get("frame"),
            "scale": forces.get("scale"),
            "z_thick_m": mesh.z_thick_m,
            "span_m": float(g["span_m"]),
            "source": forces.get("source"),
            "blades": blades_out,
            "notes": forces.get("notes"),
            "plateau_check": forces.get("plateau_check"),
            "viscous_included": wall_is_noslip,
        },
        "outlet": outlet,
        "meanline": {
            "predicted": True,
            "euler_work_j_kg": ml.euler_work_j_kg,
            "power_w": ml.power_w,
            "u_m_s": ml.u_m_s,
            "ainley_mathieson": ml.ainley_mathieson,
            "loss_scoping": ml.loss_scoping,
            "Mw1": ml.Mw1,
            "Re_c": ml.Re_c,
            "wx1": ml.wx1,
            "wy1": ml.wy1,
            "notes": ml.notes,
            "eta_from_cfd": None,
        },
        "euler_work_j_kg": ml.euler_work_j_kg,
        "power_w": ml.power_w,
        "u_m_s": ml.u_m_s,
        "Mw1": ml.Mw1,
        "loss_scoping": ml.loss_scoping,
        "eta_from_cfd": None,
        "gas": {
            "p1_pa": p1,
            "t1_k": float(gas["t1_k"]),
            "w1_m_s": float(gas["w1_m_s"]),
            "predicted": True,
            "note": gas.get("note"),
        },
        "engine_note": job["engine"].get("note"),
        "viz": viz,
        "case_dir": str(case_dir),
        "report_path": str(out_root / f"{job['name']}_report.json"),
        "geometry_test": bool(job.get("geometry_test")),
        "t_end_s": t_end,
        "t_chord_convective_s": times.t_chord_convective_s,
        "sample": None,  # filled below
    }
    report["sample"] = report["wall"]
    of_st = None
    if not skip_solve:
        try:
            from .post import extract_of_stations

            of_st = extract_of_stations(case_dir, job)
        except Exception as exc:
            of_st = {"error": f"{type(exc).__name__}: {exc}", "predicted": True, "eta_from_cfd": None}
    report["of_station"] = of_st
    report["Mw1_OF"] = (of_st or {}).get("Mw1_OF") if isinstance(of_st, dict) else None
    ft0 = blades_out[0].get("Ft_N") if blades_out else None
    try:
        report["of_power_w"] = (float(ft0) * float(ml.u_m_s)) if (ft0 is not None and not skip_solve) else None
    except (TypeError, ValueError):
        report["of_power_w"] = None
    if skip_solve:
        report["of_station"] = None
        report["Mw1_OF"] = None
        report["of_power_w"] = None
    cfd_for_ntrs = flags if (not skip_solve) else None
    of_for_ntrs = of_st if (not skip_solve and isinstance(of_st, dict) and of_st.get("Mw1_OF") is not None) else None
    report["ntrs_checks"] = ntrs_evaluate(ml, job, cfd_flags=cfd_for_ntrs, of_station=of_for_ntrs)
    report = _strip_eta(report)
    assert "eta_design_proxy" not in json.dumps(report)
    (out_root / f"{job['name']}_report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (out_root / "job_result.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    # Thin viewer copies. Geometry tests must not overwrite Marlin last_plots.
    vp = root / "viewer" / "last_plots"
    if str(job["name"]) != "marlin_v2_rotor":
        vp = out_root / "viewer_plots"
    vp.mkdir(parents=True, exist_ok=True)
    aliases = {
        "p_latest.png": "contour_p.png",
        "Umag_latest.png": "contour_Umag.png",
        "ft_history.png": "force_history.png",
        "wall_cp_blade0.png": "wall_cp_blade0.png",
        "contour_p.png": "contour_p.png",
        "contour_Umag.png": "contour_Umag.png",
        "force_history.png": "force_history.png",
    }
    for png in list(fig_dir.glob("*.png")):
        shutil.copy2(png, vp / png.name)
        dest_name = aliases.get(png.name)
        if dest_name and dest_name != png.name:
            shutil.copy2(png, vp / dest_name)
            shutil.copy2(png, fig_dir / dest_name)
    preview = case_dir / "mesh_preview.png"
    if preview.is_file():
        shutil.copy2(preview, fig_dir / "mesh_preview.png")
        shutil.copy2(preview, vp / "mesh_preview.png")
    # compact stdout
    print(
        json.dumps(
            {
                "success": report["success"],
                "predicted": True,
                "t_end_s": t_end,
                "t_reached_s": t_reached,
                "n_chords": n_chords,
                "checkMesh_failed_checks": check.get("failed_checks"),
                "solve_rc": solve.get("rc"),
                "wall_mean_p": report["wall"]["mean_p"],
                "p1_pa": p1,
                "climbing": flags["force_climbing"],
                "newtons_trusted": flags["newtons_trusted"],
                "errors": errors,
                "report": report["report_path"],
            },
            default=str,
        )
    )
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ImpulseCalc3 cascade (default: Marlin V2 rotor)")
    ap.add_argument("job", nargs="?", default=str(DEFAULT_PATH))
    ap.add_argument("--skip-solve", action="store_true", help="remesh, do not solve (writes mesh_preview.png)")
    ap.add_argument("--no-solve", action="store_true", help="alias of --skip-solve")
    ap.add_argument("--mesh-only", action="store_true", help="alias of --skip-solve")
    ap.add_argument("--post-only", action="store_true", help="reuse existing time dirs; do not remesh or resolve")
    args = ap.parse_args(argv)
    run_job(
        args.job,
        skip_solve=args.skip_solve or args.no_solve or args.mesh_only,
        post_only=args.post_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
