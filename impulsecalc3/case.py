"""Write a runnable ESI v2412 rhoCentralFoam case. No 0.95 p1. noSlip. per-blade walls."""

from __future__ import annotations

import json
import math
import textwrap
import threading
from pathlib import Path
from typing import Any

from .geometry import BladeSpec, profile_from_job
from .job import domain_x, pitch_m
from .meanline import Meanline
from .mesh import MeshBuild, write_mesh_preview_png, write_polymesh
from .times import TimeScales

# One writer on a case tree. Mesh/solve holds the tree so /outline cannot
# unlink log.checkMesh / log.rhoCentralFoam mid-run (FileNotFoundError).
_REGISTRY = threading.Lock()
_TREE_IO: dict[str, threading.RLock] = {}
_JOB_HOLDERS: dict[str, int] = {}  # resolved case path -> thread ident


def _case_key(case_dir: Path) -> str:
    return str(Path(case_dir).resolve())


def _io_lock(case_dir: Path) -> threading.RLock:
    key = _case_key(case_dir)
    with _REGISTRY:
        lock = _TREE_IO.get(key)
        if lock is None:
            lock = threading.RLock()
            _TREE_IO[key] = lock
        return lock


def acquire_job_tree(case_dir: Path) -> None:
    """Wait for any in-flight write_case, then mark this thread as the job owner."""
    case_dir = Path(case_dir)
    _io_lock(case_dir).acquire()
    with _REGISTRY:
        _JOB_HOLDERS[_case_key(case_dir)] = threading.get_ident()


def release_job_tree(case_dir: Path) -> None:
    case_dir = Path(case_dir)
    key = _case_key(case_dir)
    with _REGISTRY:
        if _JOB_HOLDERS.get(key) == threading.get_ident():
            _JOB_HOLDERS.pop(key, None)
    try:
        _io_lock(case_dir).release()
    except RuntimeError:
        pass


def job_tree_held_by_other(case_dir: Path) -> bool:
    with _REGISTRY:
        owner = _JOB_HOLDERS.get(_case_key(case_dir))
    return owner is not None and owner != threading.get_ident()


def clear_job_trees() -> None:
    """Test helper. Does not release RLocks held by other threads."""
    with _REGISTRY:
        _JOB_HOLDERS.clear()


def _refuse_busy_tree(case_dir: Path) -> None:
    if job_tree_held_by_other(case_dir):
        raise RuntimeError(
            f"refusing to rewrite {case_dir}: mesh/solve holds this tree"
        )


def _hdr(cls: str, obj: str) -> str:
    return textwrap.dedent(
        f"""\
        /*--------------------------------*- C++ -*----------------------------------*\\
        | ImpulseCalc3 generated OpenFOAM case                                       |
        \\*---------------------------------------------------------------------------*/
        FoamFile
        {{
            version     2.0;
            format      ascii;
            class       {cls};
            object      {obj};
        }}
        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
        """
    )


def _cyclic_empty_walls(
    n_blades: int = 3, cyclic: bool = True, lid_walls: bool = False
) -> tuple[str, str]:
    """U vs scalar patch blocks for the shared topology."""
    blades_u = "\n".join(
        f"            blade{k} {{ type noSlip; }}" for k in range(n_blades)
    )
    blades_s = "\n".join(
        f"            blade{k} {{ type zeroGradient; }}" for k in range(n_blades)
    )
    if cyclic:
        cyc = "            bottom { type cyclic; }\n            top    { type cyclic; }\n"
    elif lid_walls:
        cyc = (
            "            bottom { type noSlip; }\n"
            "            top    { type noSlip; }\n"
        )
        # scalars: zeroGradient on lid walls
    else:
        cyc = ""
    shared_u = textwrap.dedent(
        f"""\
{cyc}            frontAndBack {{ type empty; }}
        """
    )
    if lid_walls and not cyclic:
        cyc_s = (
            "            bottom { type zeroGradient; }\n"
            "            top    { type zeroGradient; }\n"
        )
        shared_s = textwrap.dedent(
            f"""\
{cyc_s}            frontAndBack {{ type empty; }}
        """
        )
    else:
        shared_s = shared_u
    return shared_u + blades_u, shared_s + blades_s



def write_thermophysical(case_dir: Path, job: dict[str, Any]) -> None:
    gas = job["gas"]
    gamma = float(gas["gamma"])
    r = float(gas["r_specific_j_kg_k"])
    mu = float(gas["mu_pa_s"])
    pr = float(gas.get("pr", 0.7))
    cp = gamma * r / (gamma - 1.0)
    mol = 8314.462618 / r
    body = _hdr("dictionary", "thermophysicalProperties") + textwrap.dedent(
        f"""\
        thermoType
        {{
            type            hePsiThermo;
            mixture         pureMixture;
            transport       const;
            thermo          hConst;
            equationOfState perfectGas;
            specie          specie;
            energy          sensibleInternalEnergy;
        }}
        mixture
        {{
            specie {{ molWeight {mol:.6g}; }}
            thermodynamics {{ Cp {cp:.6g}; Hf 0; hf 0; }}
            transport {{ mu {mu:.6g}; Pr {pr:.6g}; }}
        }}
        // ************************************************************************* //
        """
    )
    const = case_dir / "constant"
    const.mkdir(parents=True, exist_ok=True)
    (const / "thermophysicalProperties").write_text(body, encoding="utf-8")
    # v2412 also accepts physicalProperties name; write a pointer-style copy
    (const / "physicalProperties").write_text(body.replace("thermophysicalProperties", "physicalProperties"), encoding="utf-8")
    (const / "momentumTransport").write_text(
        _hdr("dictionary", "momentumTransport")
        + "simulationType laminar;\n// ************************************************************************* //\n",
        encoding="utf-8",
    )
    (const / "turbulenceProperties").write_text(
        _hdr("dictionary", "turbulenceProperties")
        + "simulationType laminar;\n// ************************************************************************* //\n",
        encoding="utf-8",
    )


def write_schemes(case_dir: Path) -> None:
    text = _hdr("dictionary", "fvSchemes") + textwrap.dedent(
        """\
        fluxScheme      Tadmor;
        ddtSchemes { default Euler; }
        gradSchemes { default cellLimited Gauss linear 1; }
        divSchemes {
            default none;
            div(tauMC) Gauss linear;
        }
        laplacianSchemes { default Gauss linear corrected; }
        interpolationSchemes {
            default linear;
            reconstruct(rho) Minmod;
            reconstruct(U) MinmodV;
            reconstruct(T) Minmod;
        }
        snGradSchemes { default uncorrected; }
        // ************************************************************************* //
        """
    )
    (case_dir / "system").mkdir(parents=True, exist_ok=True)
    (case_dir / "system" / "fvSchemes").write_text(text, encoding="utf-8")


def write_solution(case_dir: Path) -> None:
    text = _hdr("dictionary", "fvSolution") + textwrap.dedent(
        """\
        solvers {
            "(rho|rhoU|rhoE).*" { solver diagonal; }
            "U.*" {
                solver smoothSolver;
                smoother GaussSeidel;
                nSweeps 2;
                tolerance 1e-09;
                relTol 0.01;
            }
            "e.*" {
                solver smoothSolver;
                smoother GaussSeidel;
                nSweeps 2;
                tolerance 1e-10;
                relTol 0;
            }
            "h.*" {
                solver smoothSolver;
                smoother GaussSeidel;
                nSweeps 2;
                tolerance 1e-10;
                relTol 0;
            }
        }
        // ************************************************************************* //
        """
    )
    (case_dir / "system" / "fvSolution").write_text(text, encoding="utf-8")


def write_control_dict(
    case_dir: Path,
    job: dict[str, Any],
    times: TimeScales,
) -> float:
    cfd = job["cfd"]
    t_end = float(times.t_end_floor_s)
    # Floor = max(n_chords_min*c/W1, 1.2*Lx/a) from times.py (default n_chords_min=10).
    t_min = float(times.t_end_floor_s)
    t_max = max(2.6609057819508715e-04, 30.0 * float(times.t_chord_convective_s))
    raw = cfd.get("end_time_s")
    if raw not in (None, ""):
        t_end = float(raw)
    t_end = min(t_max, max(t_min, t_end))
    if t_end < t_min - 1e-18:
        raise ValueError(f"endTime {t_end} below floor {t_min} s (max(10 c/W1, 1.2 Lx/a))")
    write_iv = t_end / 8.0
    max_co = float(cfd.get("max_co", 0.12))
    max_dt = min(t_end / 40.0, 5e-7)
    dt0 = min(1e-11, max_dt * 0.05)
    gamma = float(job["gas"]["gamma"])
    n_blades = int(job.get("_n_blades_patches") or job.get("geometry", {}).get("n_blades_cascade") or 3)
    force_blocks = []
    surf_blocks = []
    for k in range(n_blades):
        force_blocks.append(
            f"""
            forces_blade{k}
            {{
                type            forces;
                libs            ("libforces.so");
                writeControl    timeStep;
                writeInterval   20;
                executeControl  timeStep;
                executeInterval 20;
                patches         (blade{k});
                rho             rho;
                CofR            (0 0 0);
                log             true;
                writeFields     no;
            }}"""
        )
        surf_blocks.append(
            f"""
            surfaces_blade{k}
            {{
                type            surfaces;
                libs            ("libsampling.so");
                executeAtStart  true;
                writeControl    writeTime;
                surfaceFormat   raw;
                fields          (p);
                interpolationScheme cellPoint;
                surfaces
                {{
                    blade{k}Wall
                    {{
                        type        patch;
                        patches     (blade{k});
                        interpolate false;
                    }}
                }}
            }}"""
        )
    yplus = """
            yPlus1
            {
                type            yPlus;
                libs            ("libfieldFunctionObjects.so");
                executeControl  writeTime;
                writeControl    writeTime;
                executeAtStart  false;
                log             true;
            }"""
    text = _hdr("dictionary", "controlDict") + textwrap.dedent(
        f"""\
        application     rhoCentralFoam;
        startFrom       startTime;
        startTime       0;
        stopAt          endTime;
        endTime         {t_end:.8g};
        deltaT          {dt0:.8g};
        writeControl    adjustableRunTime;
        writeInterval   {write_iv:.8g};
        purgeWrite      0;
        writeFormat     ascii;
        writePrecision  8;
        writeCompression off;
        timeFormat      general;
        timePrecision   8;
        runTimeModifiable true;
        adjustTimeStep  yes;
        maxCo           {max_co:.4g};
        maxDeltaT       {max_dt:.8g};
        functions
        {{
        {"".join(force_blocks)}
        {"".join(surf_blocks)}
        {yplus}
        }}
        // ************************************************************************* //
        """
    )
    (case_dir / "system" / "controlDict").write_text(text, encoding="utf-8")
    # gamma is used by waveTransmissive in 0/p, not here
    _ = gamma
    return t_end


def _resolve_inlet_thermo(job: dict[str, Any], ml: Meanline) -> dict[str, float]:
    """Map locked inlet spec → static state for 0/ fields.

    Locked law: relative Mrel1, β1, Pt,rel, Tt into rotor-only foam.
    - cfd.inlet_bc == "total_rel" (default for new jobs): gas p1_pa=Pt,rel, t1_k=Tt;
      M from gas m_rel1 or ml.Mw1; derive static p,T and |W|=M a(T).
    - inlet_bc == "static_rel": legacy fixedValue W1 + static p,T (v3 cases).
    """
    import math
    gas = job["gas"]
    cfd = job.get("cfd") or {}
    gamma = float(gas["gamma"])
    # R_specific from gas if present else from meanline path via a/T later
    r_sp = float(gas.get("r_specific") or gas.get("R") or 0.0)
    if r_sp <= 0.0:
        # fall back: a^2/(γ T) once we have T; use job meanline note gas
        r_sp = float(gas.get("r_j_kg_k") or 320.0)
    mode = str(cfd.get("inlet_bc") or gas.get("inlet_bc") or ("total_rel" if gas.get("pt_rel_pa") not in (None, "") else "static_rel"))
    beta = float(ml.beta1_flow_deg)
    rad = math.radians(beta)
    if mode == "static_rel":
        p_s = float(gas["p1_pa"])
        t_s = float(gas["t1_k"])
        wx, wy = float(ml.wx1), float(ml.wy1)
        w = math.hypot(wx, wy)
        a = math.sqrt(max(gamma * r_sp * t_s, 1e-18))
        m = w / a
        pt = p_s * (1.0 + 0.5 * (gamma - 1.0) * m * m) ** (gamma / (gamma - 1.0))
        tt = t_s * (1.0 + 0.5 * (gamma - 1.0) * m * m)
        return {
            "mode": mode,
            "p_static": p_s,
            "t_static": t_s,
            "pt_rel": pt,
            "tt": tt,
            "m_rel": m,
            "wx": wx,
            "wy": wy,
            "w": w,
        }
    # total_rel
    pt = float(gas.get("pt_rel_pa") or gas["p1_pa"])
    tt = float(gas.get("tt_k") or gas["t1_k"])
    if gas.get("m_rel1") not in (None, ""):
        m = float(gas["m_rel1"])
    else:
        m = float(ml.Mw1)
    m = max(m, 1e-9)
    fac = 1.0 + 0.5 * (gamma - 1.0) * m * m
    t_s = tt / fac
    p_s = pt / (fac ** (gamma / (gamma - 1.0)))
    a = math.sqrt(max(gamma * r_sp * t_s, 1e-18))
    w = m * a
    wx, wy = w * math.cos(rad), w * math.sin(rad)
    return {
        "mode": "total_rel",
        "p_static": p_s,
        "t_static": t_s,
        "pt_rel": pt,
        "tt": tt,
        "m_rel": m,
        "wx": wx,
        "wy": wy,
        "w": w,
    }



def write_fields(case_dir: Path, job: dict[str, Any], ml: Meanline) -> None:
    gas = job["gas"]
    gamma = float(gas["gamma"])
    cfd = job["cfd"]
    inlet = _resolve_inlet_thermo(job, ml)
    p1 = float(inlet["p_static"])
    t1 = float(inlet["t_static"])
    wx, wy = float(inlet["wx"]), float(inlet["wy"])
    job["_inlet_resolved"] = inlet
    x_in, x_out = domain_x(job)
    l_inf = max(x_out - float(job["geometry"]["chord_m"]), 0.02)
    outlet_kind = str(cfd.get("outlet_p", "waveTransmissive"))
    n_blades = int(job.get("_n_blades_patches") or 3)
    cyclic = bool(job.get("_cyclic_pitch", True))
    lid_walls = bool(job.get("_lid_walls", False))
    u_shared, s_shared = _cyclic_empty_walls(n_blades=n_blades, cyclic=cyclic, lid_walls=lid_walls)

    if outlet_kind == "inletOutlet":
        p_out = (
            f"            outlet {{ type inletOutlet; inletValue uniform {p1:.8g}; "
            f"value uniform {p1:.8g}; }}"
        )
    elif outlet_kind == "zeroGradient":
        p_out = "            outlet { type zeroGradient; }"
    else:
        # waveTransmissive: far-field reference is p1, NOT 0.95 p1. Dump is measured.
        p_out = (
            "            outlet {\n"
            "                type            waveTransmissive;\n"
            f"                gamma           {gamma:.8g};\n"
            f"                fieldInf        {p1:.8g};\n"
            f"                lInf            {l_inf:.8g};\n"
            f"                value           uniform {p1:.8g};\n"
            "            }"
        )

    zero = case_dir / "0"
    zero.mkdir(parents=True, exist_ok=True)
    u = _hdr("volVectorField", "U") + textwrap.dedent(
        f"""\
        dimensions [0 1 -1 0 0 0 0];
        internalField uniform ({wx:.8g} {wy:.8g} 0);
        boundaryField {{
            inlet  {{ type fixedValue; value uniform ({wx:.8g} {wy:.8g} 0); }}
            outlet {{ type inletOutlet; inletValue uniform (0 0 0); value uniform ({wx:.8g} {wy:.8g} 0); }}
        {u_shared}
        }}
        // ************************************************************************* //
        """
    )
    p = _hdr("volScalarField", "p") + textwrap.dedent(
        f"""\
        dimensions [1 -1 -2 0 0 0 0];
        internalField uniform {p1:.8g};
        boundaryField {{
            inlet  {{ type fixedValue; value uniform {p1:.8g}; }}
        {p_out}
        {s_shared}
        }}
        // ************************************************************************* //
        """
    )
    t = _hdr("volScalarField", "T") + textwrap.dedent(
        f"""\
        dimensions [0 0 0 1 0 0 0];
        internalField uniform {t1:.8g};
        boundaryField {{
            inlet  {{ type fixedValue; value uniform {t1:.8g}; }}
            outlet {{ type inletOutlet; inletValue uniform {t1:.8g}; value uniform {t1:.8g}; }}
        {s_shared}
        }}
        // ************************************************************************* //
        """
    )
    (zero / "U").write_text(u, encoding="utf-8")
    (zero / "p").write_text(p, encoding="utf-8")
    (zero / "T").write_text(t, encoding="utf-8")


def write_case_readme(
    case_dir: Path,
    job: dict[str, Any],
    ml: Meanline,
    times: TimeScales,
    mesh: MeshBuild,
    t_end: float,
) -> None:
    g = job["geometry"]
    gas = job["gas"]
    lines = [
        f"ImpulseCalc3 OpenFOAM case — {job.get('article') or job.get('name')}",
        "NOT flight. NOT CFX. CFD checks the blade; it does not invent efficiency.",
        "",
        "TIME SCALES (required)",
        *("  " + n for n in times.notes),
        f"  endTime written = {t_end:.8g} s",
        f"  n_chords = endTime / (c/W1) = {t_end / times.t_chord_convective_s:.3g}",
        "  Stop: force plateau (dFt/dt small) AND/OR t > max(5 c/W1, 1.2 Lx/a). Never 2e-6 s.",
        "",
        "MESH",
        *("  " + n for n in mesh.check_notes),
        f"  cells={mesh.n_cells} points={mesh.n_points} first_cell={mesh.first_cell_m:.3g} m",
        f"  patches={mesh.patches}",
        "",
        "WALL",
        "  noSlip on blade0, blade1, blade2. Laminar. Viscous Ft is meaningful only because noSlip.",
        "",
        "BCs",
        f"  inlet: relative W1=({ml.wx1:.6g}, {ml.wy1:.6g}, 0) m/s at flow β1={ml.beta1_flow_deg}°",
        f"  p1={gas['p1_pa']} Pa  T1={gas['t1_k']} K  (PREDICTED; not Pc={job['engine'].get('Pc_bar')} bar)",
        f"  outlet p: {job['cfd'].get('outlet_p')} with fieldInf=p1 (NOT 0.95 p1). Dump is measured.",
        "",
        "GEOMETRY",
        f"  metal β* = ({ml.beta1_metal_deg}, {ml.beta2_metal_deg}) deg  stagger={ml.stagger_deg} deg",
        f"  chord={g['chord_m']} m  pitch={pitch_m(job):.6g} m  solidity={g['solidity']}",
        f"  n_blades_machine={g['n_blades_machine']}  cascade blades=3  span={g['span_m']} m",
        "",
        "FORCES",
        "  OpenFOAM forces FO per blade patch. Ft=Fy tangential, Fd=Fx axial.",
        f"  Scale slab {mesh.z_thick_m} m → span {g['span_m']} m. NOT total/n_blades clones.",
        "",
        "MEANLINE (not CFD η)",
        f"  Euler work (PREDICTED) = {ml.euler_work_j_kg:.6g} J/kg",
        f"  AM book Yp (PREDICTED) = {ml.ainley_mathieson['Yp_profile']:.4g}  [{ml.ainley_mathieson['book']}]",
        "  Do not write eta_design_proxy into this folder.",
        "",
        "FAIL LOUD",
        "  mesh fail / solve crash / sample-not-on-wall / force still climbing with t < 1 chord → success=false.",
        "",
    ]
    (case_dir / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    meta = {
        "app": "ImpulseCalc3",
        "solver": "rhoCentralFoam",
        "openfoam": "ESI-v2412",
        "wall": "noSlip",
        "mesh": mesh.mesh_kind,
        "predicted": True,
        "t_end_s": t_end,
        "t_chord_convective_s": times.t_chord_convective_s,
        "t_chord_acoustic_s": times.t_chord_acoustic_s,
        "Mw1": times.Mw1,
        "n_chords": t_end / times.t_chord_convective_s,
        "mesh_cells": mesh.n_cells,
        "first_cell_m": mesh.first_cell_m,
        "patches": mesh.patches,
        "euler_work_j_kg": ml.euler_work_j_kg,
        "ainley_mathieson_Yp": ml.ainley_mathieson["Yp_profile"],
        "engine_note": job["engine"]["note"],
    }
    (case_dir / "impulsecalc3_case_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    # Explicitly do not write eta_design_proxy
    assert "eta_design_proxy" not in meta


def write_case(
    case_dir: Path,
    job: dict[str, Any],
    ml: Meanline,
    times: TimeScales,
    spec: BladeSpec,
) -> tuple[MeshBuild, float]:
    case_dir = Path(case_dir)
    _refuse_busy_tree(case_dir)
    lock = _io_lock(case_dir)
    lock.acquire()
    try:
        _refuse_busy_tree(case_dir)
        return _write_case_unlocked(case_dir, job, ml, times, spec)
    finally:
        lock.release()


def _write_case_unlocked(
    case_dir: Path,
    job: dict[str, Any],
    ml: Meanline,
    times: TimeScales,
    spec: BladeSpec,
) -> tuple[MeshBuild, float]:
    case_dir = Path(case_dir)
    if case_dir.exists():
        # only wipe generated trees we own
        import shutil

        for sub in ("0", "constant", "system"):
            p = case_dir / sub
            if p.exists():
                shutil.rmtree(p)
        for name in ("README.txt", "impulsecalc3_case_meta.json", "log.checkMesh", "log.rhoCentralFoam"):
            p = case_dir / name
            if p.exists():
                p.unlink()
    (case_dir / "system").mkdir(parents=True, exist_ok=True)
    (case_dir / "constant").mkdir(parents=True, exist_ok=True)
    (case_dir / "0").mkdir(parents=True, exist_ok=True)

    poly = profile_from_job(job, spec)
    mesh = write_polymesh(case_dir, job, spec, poly=poly)
    n_b = len([k for k in mesh.patches if str(k).startswith("blade")])
    job["_n_blades_patches"] = n_b
    job["_lid_walls"] = mesh.mesh_kind == "cassette_OH"
    job["_cyclic_pitch"] = (mesh.mesh_kind == "body_fitted_OH") and bool(mesh.patches.get("bottom"))
    write_thermophysical(case_dir, job)
    write_schemes(case_dir)
    write_solution(case_dir)
    t_end = write_control_dict(case_dir, job, times)
    write_fields(case_dir, job, ml)
    write_case_readme(case_dir, job, ml, times, mesh, t_end)
    write_mesh_preview_png(case_dir / "mesh_preview.png", job, spec, mesh)
    # boundary_conditions.json — what was actually written
    bc = {
        "inlet_U": "fixedValue from Mrel1+β1 (total_rel→static) or legacy W1",
        "outlet_U": "inletOutlet inletValue (0 0 0) — not W1, not a second stator",
        "inlet_bc": (job.get("cfd") or {}).get("inlet_bc", "total_rel"),
        "inlet_p_static": float((job.get("_inlet_resolved") or {}).get("p_static", job["gas"]["p1_pa"])),
        "inlet_pt_rel": float((job.get("_inlet_resolved") or {}).get("pt_rel", job["gas"]["p1_pa"])),
        "inlet_p": float((job.get("_inlet_resolved") or {}).get("p_static", job["gas"]["p1_pa"])),
        "inlet_T": float(job["gas"]["t1_k"]),
        "outlet_p": job["cfd"].get("outlet_p"),
        "outlet_p_fieldInf": float(job["gas"]["p1_pa"]),
        "outlet_p_NOT": "0.95 p1",
        "blades": "noSlip on blade0 blade1 blade2",
        "frontAndBack": "empty",
        "top_bottom": ("lid walls" if mesh.mesh_kind == "cassette_OH" else "cyclic pitch-periodic"),
        "turbulence": "laminar",
        "predicted": True,
    }
    (case_dir / "boundary_conditions.json").write_text(json.dumps(bc, indent=2) + "\n", encoding="utf-8")
    return mesh, t_end
