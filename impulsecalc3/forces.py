"""Parse ESI v2412 forces function-object output. Per-blade. No total/n_blades clone."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any


def parse_force_dat(path: Path) -> list[dict[str, float]]:
    """v2412 10-column force.dat: Time tot_x y z  p_x y z  visc_x y z."""
    rows: list[dict[str, float]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.replace("(", " ").replace(")", " ").split()
        nums = []
        for p in parts:
            try:
                nums.append(float(p))
            except ValueError:
                continue
        if len(nums) < 10:
            continue
        rows.append(
            {
                "t": nums[0],
                "Fx": nums[1],
                "Fy": nums[2],
                "Fz": nums[3],
                "Fpx": nums[4],
                "Fpy": nums[5],
                "Fpz": nums[6],
                "Fvx": nums[7],
                "Fvy": nums[8],
                "Fvz": nums[9],
            }
        )
    return rows


def _find_force_dat(case_dir: Path, blade: int) -> Path | None:
    root = case_dir / "postProcessing"
    if not root.is_dir():
        return None
    # forces_bladeK/0/force.dat  or nested time dirs
    hits = sorted(root.glob(f"forces_blade{blade}/**/force.dat"))
    return hits[-1] if hits else None


def plateau_flags(
    times: list[float],
    ft: list[float],
    t_chord: float,
    rel_tol: float = 0.04,
) -> dict[str, Any]:
    if len(times) < 4 or t_chord <= 0:
        return {"plateau": False, "climbing": True, "dFt_over_Ft": None, "window_s": None}
    t_end = times[-1]
    t0 = t_end - max(t_chord, 0.2 * t_end)
    window = [(t, f) for t, f in zip(times, ft) if t >= t0]
    if len(window) < 3:
        window = list(zip(times[-5:], ft[-5:]))
    fvals = [f for _, f in window]
    fmean = sum(fvals) / len(fvals)
    fspan = max(fvals) - min(fvals)
    rel = abs(fspan) / max(abs(fmean), 1e-9)
    dt = window[-1][0] - window[0][0]
    dft_dt = (window[-1][1] - window[0][1]) / max(dt, 1e-16)
    climbing = rel > rel_tol
    return {
        "plateau": not climbing,
        "climbing": climbing,
        "dFt_over_Ft": rel,
        "dFt_dt": dft_dt,
        "window_s": dt,
        "Ft_window_mean": fmean,
    }


def load_blade_forces(
    case_dir: Path,
    *,
    span_m: float,
    z_thick_m: float,
    t_chord_s: float,
    n_chords_run: float,
    wall_is_noslip: bool,
) -> dict[str, Any]:
    scale = span_m / max(z_thick_m, 1e-16)
    blades = []
    histories = []
    missing = []
    n_force = 0
    root_pp = case_dir / "postProcessing"
    if root_pp.is_dir():
        n_force = sum(1 for d in root_pp.iterdir() if d.is_dir() and d.name.startswith("forces_blade"))
    n_blades = n_force if n_force else 3
    for k in range(n_blades):
        p = _find_force_dat(case_dir, k)
        if p is None:
            missing.append(k)
            continue
        rows = parse_force_dat(p)
        if not rows:
            missing.append(k)
            continue
        last = rows[-1]
        # Ft = Fy tangential, Fd = Fx axial. Scale slab → span.
        fd_slab = last["Fx"]
        ft_slab = last["Fy"]
        fdv_slab = last["Fvx"]
        ftv_slab = last["Fvy"]
        blades.append(
            {
                "blade_index": k,
                "patch": f"blade{k}",
                "Ft_N": ft_slab * scale,
                "Fd_N": fd_slab * scale,
                "Ft_N_per_m": ft_slab / z_thick_m,
                "Fd_N_per_m": fd_slab / z_thick_m,
                "Ft_pressure_N": last["Fpy"] * scale,
                "Fd_pressure_N": last["Fpx"] * scale,
                "Ft_viscous_N": ftv_slab * scale if wall_is_noslip else 0.0,
                "Fd_viscous_N": fdv_slab * scale if wall_is_noslip else 0.0,
                "viscous_included": bool(wall_is_noslip),
                "span_m": span_m,
                "source": "openfoam_forces_fo",
            }
        )
        histories.append(
            {
                "blade": k,
                "t": [r["t"] for r in rows],
                "Ft_N": [r["Fy"] * scale for r in rows],
                "Fd_N": [r["Fx"] * scale for r in rows],
            }
        )
    if missing:
        return {
            "success": False,
            "predicted": True,
            "climbing": True,
            "plateau": False,
            "error": f"missing force.dat for blade(s) {missing}",
            "blades": blades,
            "frame": {"x": "axial Fd", "y": "tangential Ft"},
            "notes": ["FAIL LOUD: no cartoon forces"],
        }
    plat = plateau_flags(histories[0]["t"], histories[0]["Ft_N"], t_chord_s)
    climbing = bool(plat["climbing"])
    # Inlet state is unsigned ⇒ predicted always. Plateau only clears climbing.
    predicted = True
    usable = bool(plat["plateau"]) and n_chords_run >= 5.0
    notes = [
        "Ft=Fy tangential, Fd=Fx axial, relative cascade chord frame.",
        f"Scale OF slab {z_thick_m} m → span {span_m} m. Per-blade patches, not total/n.",
        "PREDICTED until ORBIT signs the GG-to-rotor station (and until plateau).",
    ]
    if climbing:
        notes.append("Force still climbing at endTime. Do not treat as a design load.")
    if not wall_is_noslip:
        notes.append("Viscous columns zeroed: wall is not noSlip.")
    return {
        "success": True,
        "predicted": predicted,
        "climbing": climbing,
        "plateau": bool(plat["plateau"]),
        "usable_as_design_load": False,  # unsigned station
        "n_chords_run": n_chords_run,
        "plateau_check": plat,
        "blades": blades,
        "history": histories,
        "span_m": span_m,
        "z_thick_m": z_thick_m,
        "scale": scale,
        "frame": {"x": "axial / drag Fd", "y": "tangential Ft", "z": "empty span"},
        "units": {"force": "N at span_m", "force_per_span": "N/m"},
        "viscous_included": wall_is_noslip,
        "source": "openfoam_forces_fo",
        "notes": notes,
    }
