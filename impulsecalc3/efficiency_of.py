"""OpenFOAM relative-cascade efficiency load-path (Marlin full-admission 2D).

Primary from field quantities — NOT 1D meanline / Ainley–Mathieson.
Those stay SCOPING comparison only; never mint stage η as CFD truth.

Methods (group-chat + voice stamp):
  8  Y_rel, ζ_rel from mass-weighted p,T,W → p0,rel
  7  Δs from mass-avg T,p on the same planes
  6  work identities w_Ft = Ft·U/ṁ  and  w_gas = U·(Wy1 − Wy3); honesty gate
Static p3/p2 and W3/W1 are impulse checks only.
η_tt / η_ts are optional derived PREDICTED (only if U mapped) — not primary CSV.
eta_from_cfd stays None.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

# Honesty gate: |w_Ft − w_gas| / max(|w|,eps) must be below this to publish η columns.
WORK_MATCH_TOL = 0.03


def geom_hash(job: dict[str, Any] | None) -> str:
    """Stable short hash of geometry + OP gas knobs (not mesh cell count)."""
    if not job:
        return "nojob"
    g = job.get("geometry") or {}
    gas = job.get("gas") or {}
    payload = {
        "family": g.get("profile_family"),
        "chord_m": g.get("chord_m"),
        "pitch_m": g.get("pitch_m"),
        "solidity": g.get("solidity"),
        "beta1_flow_deg": g.get("beta1_flow_deg"),
        "beta2_flow_deg": g.get("beta2_flow_deg"),
        "span_m": g.get("span_m"),
        "mean_radius_m": g.get("mean_radius_m"),
        "p1_pa": gas.get("p1_pa"),
        "t1_k": gas.get("t1_k"),
        "w1_m_s": gas.get("w1_m_s"),
        "gamma": gas.get("gamma"),
        "r_specific_j_kg_k": gas.get("r_specific_j_kg_k"),
        "blade_speed_u_m_s": gas.get("blade_speed_u_m_s"),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


def _cp_r(gamma: float, rspec: float) -> tuple[float, float]:
    g = max(float(gamma), 1.0001)
    r = max(float(rspec), 1e-12)
    cp = g * r / (g - 1.0)
    return cp, r


def p0_rel(p: float, T: float, Wmag: float, gamma: float, rspec: float) -> float:
    """Relative stagnation pressure from static p,T and |W|."""
    cp, _ = _cp_r(gamma, rspec)
    T = max(float(T), 1e-9)
    T0 = T + 0.5 * float(Wmag) ** 2 / cp
    return float(p) * (T0 / T) ** (gamma / (gamma - 1.0))


def T0_rel(T: float, Wmag: float, gamma: float, rspec: float) -> float:
    cp, _ = _cp_r(gamma, rspec)
    return float(T) + 0.5 * float(Wmag) ** 2 / cp


def Y_rel(p01: float, p03: float, p3: float) -> float | None:
    """Method 8: Y = (p0,rel,in − p0,rel,out) / (p0,rel,out − p_out)."""
    den = float(p03) - float(p3)
    if abs(den) < 1e-12:
        return None
    return (float(p01) - float(p03)) / den


def zeta_rel(W3: float, W3_is: float) -> float | None:
    """Method 8: ζ_rel = 1 − (W3/W3_is)²  (KE loss vs isentropic relative exit)."""
    if W3_is is None or abs(float(W3_is)) < 1e-12:
        return None
    r = float(W3) / float(W3_is)
    return 1.0 - r * r


def W_is_exit(T01_rel: float, p01_rel: float, p3: float, gamma: float, rspec: float) -> float | None:
    """Isentropic relative exit speed to static p3 from inlet relative stagnation."""
    cp, _ = _cp_r(gamma, rspec)
    if p01_rel <= 0 or p3 <= 0 or T01_rel <= 0:
        return None
    pr = min(max(float(p3) / float(p01_rel), 1e-12), 1.0)
    T3s = float(T01_rel) * pr ** ((gamma - 1.0) / gamma)
    dT = float(T01_rel) - T3s
    if dT < 0:
        return None
    return math.sqrt(max(2.0 * cp * dT, 0.0))


def delta_s(T1: float, p1: float, T3: float, p3: float, gamma: float, rspec: float) -> float | None:
    """Method 7: Δs = Cp ln(T3/T1) − R ln(p3/p1) on mass-avg states."""
    if min(T1, T3, p1, p3) <= 0:
        return None
    cp, r = _cp_r(gamma, rspec)
    return cp * math.log(float(T3) / float(T1)) - r * math.log(float(p3) / float(p1))


def _blade_le_te(case_dir: Path, chord_m: float) -> tuple[float, float]:
    """LE/TE x from latest blade0 wall sample; fallback chord frame 0…c."""
    pp = Path(case_dir) / "postProcessing"
    hits = sorted(pp.glob("surfaces_blade0/**/p_blade0Wall.raw")) if pp.is_dir() else []
    xs: list[float] = []
    if hits:
        for line in hits[-1].read_text(encoding="utf-8", errors="replace").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 3:
                continue
            try:
                xs.append(float(parts[0]))
            except ValueError:
                continue
    if xs:
        return min(xs), max(xs)
    return 0.0, float(chord_m)


def _load_fields(case_dir: Path) -> dict[str, Any] | None:
    from viewer.ofio import read_scalar_field, read_vector_field, time_dirs

    case_dir = Path(case_dir)
    tds = [(t, d) for t, d in time_dirs(case_dir) if t > 0]
    if not tds:
        return None
    t_latest, tdir = tds[-1]
    C = read_vector_field(tdir / "C")
    if not C:
        try:
            from viewer.ofio import cell_centres

            cc = cell_centres(case_dir)
            C = [tuple(map(float, row)) for row in cc]
        except Exception:
            return None
    n = len(C)
    U = read_vector_field(tdir / "U", n)
    p = read_scalar_field(tdir / "p", n)
    T = read_scalar_field(tdir / "T", n)
    rho = read_scalar_field(tdir / "rho", n)
    if not U or not p or not T or not rho:
        return None
    if not (len(U) == len(p) == len(T) == len(rho) == n):
        return None
    return {"t": float(t_latest), "C": C, "U": U, "p": p, "T": T, "rho": rho, "n": n}


def _station_ids(C: list, x_st: float, band: float) -> list[int]:
    ids = [i for i, c in enumerate(C) if abs(float(c[0]) - x_st) <= band]
    if len(ids) >= 8:
        return ids
    # widen once
    band2 = band * 2.5
    ids = [i for i, c in enumerate(C) if abs(float(c[0]) - x_st) <= band2]
    return ids


def _mass_avg_plane(
    ids: list[int],
    C: list,
    U: list,
    p: list,
    T: list,
    rho: list,
    *,
    z_thick: float,
) -> dict[str, float | None]:
    """Mass-flux averages on a near-constant-x slab: weight = ρ |Wx| A, A = z·Δy."""
    if not ids:
        return {
            "n": 0,
            "p": None,
            "T": None,
            "Wx": None,
            "Wy": None,
            "W": None,
            "mdot_slab": None,
        }
    # sort pitchwise for Δy areas
    order = sorted(ids, key=lambda i: float(C[i][1]))
    ys = [float(C[i][1]) for i in order]
    areas: list[float] = []
    for k, i in enumerate(order):
        if len(order) == 1:
            dy = abs(z_thick)  # degenerate
        elif k == 0:
            dy = abs(ys[1] - ys[0])
        elif k == len(order) - 1:
            dy = abs(ys[-1] - ys[-2])
        else:
            dy = 0.5 * abs(ys[k + 1] - ys[k - 1])
        areas.append(max(dy, 1e-16) * max(z_thick, 1e-16))

    wsum = 0.0
    mdot = 0.0
    acc = {"p": 0.0, "T": 0.0, "Wx": 0.0, "Wy": 0.0, "W": 0.0}
    for k, i in enumerate(order):
        ux, uy = float(U[i][0]), float(U[i][1])
        Wmag = math.hypot(ux, uy)
        wi = max(float(rho[i]), 0.0) * abs(ux) * areas[k]
        if wi <= 0.0:
            continue
        wsum += wi
        mdot += wi  # ρ|Wx|A
        acc["p"] += wi * float(p[i])
        acc["T"] += wi * float(T[i])
        acc["Wx"] += wi * ux
        acc["Wy"] += wi * uy
        acc["W"] += wi * Wmag
    if wsum <= 0.0:
        return {
            "n": len(ids),
            "p": None,
            "T": None,
            "Wx": None,
            "Wy": None,
            "W": None,
            "mdot_slab": 0.0,
        }
    inv = 1.0 / wsum
    return {
        "n": len(ids),
        "p": acc["p"] * inv,
        "T": acc["T"] * inv,
        "Wx": acc["Wx"] * inv,
        "Wy": acc["Wy"] * inv,
        "W": acc["W"] * inv,
        "mdot_slab": mdot,
    }


def _plateau_and_ft(case_dir: Path, job: dict[str, Any], forces: dict[str, Any] | None) -> dict[str, Any]:
    """Reuse existing forces / sample_on_wall gates when provided; else parse case."""
    out: dict[str, Any] = {
        "plateau": False,
        "sample_on_wall": False,
        "Ft_N": None,
        "climbing": True,
    }
    if forces and isinstance(forces, dict):
        out["plateau"] = bool(forces.get("plateau"))
        out["climbing"] = bool(forces.get("climbing", not out["plateau"]))
        blades = forces.get("blades") or []
        if blades:
            try:
                out["Ft_N"] = float(blades[0].get("Ft_N"))
            except (TypeError, ValueError):
                out["Ft_N"] = None
    wall = None
    if forces is None:
        try:
            from .forces import load_blade_forces
            from .times import compute_times

            g = job.get("geometry") or {}
            gas = job.get("gas") or {}
            cfd = job.get("cfd") or {}
            times = compute_times(job)
            fr = load_blade_forces(
                Path(case_dir),
                span_m=float(g.get("span_m") or 0.005),
                z_thick_m=float(cfd.get("z_thick_m") or 0.001),
                t_chord_s=float(times.t_chord_convective_s),
                n_chords_run=float(getattr(times, "n_chords", 0.0) or 0.0),
                wall_is_noslip=str(cfd.get("wall") or "noSlip").lower() == "noslip",
            )
            out["plateau"] = bool(fr.get("plateau"))
            out["climbing"] = bool(fr.get("climbing", True))
            blades = fr.get("blades") or []
            if blades:
                out["Ft_N"] = float(blades[0]["Ft_N"])
        except Exception as exc:
            out["forces_error"] = f"{type(exc).__name__}: {exc}"
    # sample_on_wall from wall raw if present
    pp = Path(case_dir) / "postProcessing"
    hits = list(pp.glob("surfaces_blade0/**/p_blade0Wall.raw")) if pp.is_dir() else []
    out["sample_on_wall"] = bool(hits)
    if forces and "sample_on_wall" in (forces.get("notes") or []):
        pass
    return out


def compute_efficiency_of(
    case_dir: Path | str,
    job: dict[str, Any] | None = None,
    *,
    forces: dict[str, Any] | None = None,
    flags: dict[str, Any] | None = None,
    work_match_tol: float = WORK_MATCH_TOL,
) -> dict[str, Any]:
    """Full-admission Marlin 2D relative cascade efficiency row (PREDICTED)."""
    case_dir = Path(case_dir)
    job = job or {}
    g = job.get("geometry") or {}
    gas = job.get("gas") or {}
    cfd = job.get("cfd") or {}
    gamma = float(gas.get("gamma") or 1.3)
    rspec = float(gas.get("r_specific_j_kg_k") or 320.0)
    chord = float(g.get("chord_m") or 0.01)
    z_thick = float(cfd.get("z_thick_m") or 0.001)
    span = float(g.get("span_m") or 0.005)
    U_blade = gas.get("blade_speed_u_m_s")
    try:
        U_blade = float(U_blade) if U_blade is not None else None
    except (TypeError, ValueError):
        U_blade = None
    u_mapped = U_blade is not None and abs(U_blade) > 1e-9

    gh = geom_hash(job)
    row: dict[str, Any] = {
        "geom_hash": gh,
        "p1_pa": gas.get("p1_pa"),
        "T1_K": gas.get("t1_k"),
        "W1_m_s": gas.get("w1_m_s"),
        "beta1_deg": g.get("beta1_flow_deg"),
        "gas_gamma": gamma,
        "gas_R": rspec,
        "Y_rel": None,
        "zeta_rel": None,
        "delta_s": None,
        "w_Ft": None,
        "w_gas": None,
        "residual": None,
        "p3_over_p2": None,
        "W3_over_W1": None,
        "plateau": False,
        "sample_on_wall": False,
        "work_match_ok": False,
        "gate_ok": False,
        "predicted": True,
        "eta_from_cfd": None,
        "eta_tt_PREDICTED": None,
        "eta_ts_PREDICTED": None,
        "publish_eta": False,
        "authority": "PREDICTED",
        "note": (
            "Field CFD efficiency load-path (Methods 6/7/8). "
            "1D meanline/AM are SCOPING only. eta_from_cfd=None."
        ),
    }

    gate_flags = _plateau_and_ft(case_dir, job, forces)
    if flags:
        if "force_plateau" in flags:
            gate_flags["plateau"] = bool(flags["force_plateau"])
        if "sample_on_wall" in flags:
            gate_flags["sample_on_wall"] = bool(flags["sample_on_wall"])
    row["plateau"] = bool(gate_flags.get("plateau"))
    row["sample_on_wall"] = bool(gate_flags.get("sample_on_wall"))
    Ft = gate_flags.get("Ft_N")

    fields = _load_fields(case_dir)
    if fields is None:
        row["error"] = "no positive-time OpenFOAM fields (p,T,U,rho,C)"
        return {"row_efficiency_PREDICTED": row, "eta_from_cfd": None, "predicted": True}

    x_le, x_te = _blade_le_te(case_dir, chord)
    # Stations: interior near LE (off inlet face) + TE+1c (off dump)
    x1 = x_le - 0.10 * chord
    x3 = x_te + 1.0 * chord
    xs = [float(c[0]) for c in fields["C"]]
    xmin, xmax = min(xs), max(xs)
    # keep interior: nudge off domain faces
    eps = 0.02 * chord
    x1 = min(max(x1, xmin + eps), xmax - eps)
    x3 = min(max(x3, xmin + eps), xmax - eps)
    if x3 <= x1:
        x3 = min(xmax - eps, x1 + chord)

    band = max(0.04 * chord, 2.5e-4)
    ids1 = _station_ids(fields["C"], x1, band)
    ids3 = _station_ids(fields["C"], x3, band)
    st1 = _mass_avg_plane(
        ids1, fields["C"], fields["U"], fields["p"], fields["T"], fields["rho"], z_thick=z_thick
    )
    st3 = _mass_avg_plane(
        ids3, fields["C"], fields["U"], fields["p"], fields["T"], fields["rho"], z_thick=z_thick
    )
    row["stations"] = {
        "x1_m": x1,
        "x3_m": x3,
        "x_le_m": x_le,
        "x_te_m": x_te,
        "band_m": band,
        "n1": st1["n"],
        "n3": st3["n"],
        "time_s": fields["t"],
        "averaging": "mass_flux rho*|Wx|*A — median banned for efficiency block",
    }

    p1a, T1a, W1a = st1["p"], st1["T"], st1["W"]
    p3a, T3a, W3a = st3["p"], st3["T"], st3["W"]
    Wy1, Wy3 = st1["Wy"], st3["Wy"]

    if None not in (p1a, T1a, W1a, p3a, T3a, W3a):
        p01 = p0_rel(p1a, T1a, W1a, gamma, rspec)
        p03 = p0_rel(p3a, T3a, W3a, gamma, rspec)
        T01 = T0_rel(T1a, W1a, gamma, rspec)
        row["Y_rel"] = Y_rel(p01, p03, p3a)
        Wis = W_is_exit(T01, p01, p3a, gamma, rspec)
        row["zeta_rel"] = zeta_rel(W3a, Wis) if Wis is not None else None
        row["delta_s"] = delta_s(T1a, p1a, T3a, p3a, gamma, rspec)
        row["p3_over_p2"] = (float(p3a) / float(p1a)) if abs(float(p1a)) > 1e-12 else None
        row["W3_over_W1"] = (float(W3a) / float(W1a)) if abs(float(W1a)) > 1e-12 else None
        row["p0_rel_1"] = p01
        row["p0_rel_3"] = p03
        row["W3_is"] = Wis

    # Method 6 — work identities (span-consistent; scale cancels)
    mdot_slab = st1.get("mdot_slab")
    w_ft = w_gas = None
    if u_mapped and Ft is not None and mdot_slab and abs(float(mdot_slab)) > 1e-16:
        # Ft_N already span-scaled; mdot_span = mdot_slab * (span/z)
        mdot_span = float(mdot_slab) * (span / max(z_thick, 1e-16))
        w_ft = float(Ft) * float(U_blade) / mdot_span
    if u_mapped and Wy1 is not None and Wy3 is not None:
        w_gas = float(U_blade) * (float(Wy1) - float(Wy3))
    row["w_Ft"] = w_ft
    row["w_gas"] = w_gas
    if w_ft is not None and w_gas is not None:
        scale = max(abs(w_ft), abs(w_gas), 1e-12)
        row["residual"] = abs(w_ft - w_gas) / scale
        row["work_match_ok"] = bool(row["residual"] <= work_match_tol)
    else:
        row["residual"] = None
        row["work_match_ok"] = False

    row["gate_ok"] = bool(row["plateau"] and row["sample_on_wall"] and row["work_match_ok"])
    row["publish_eta"] = bool(row["gate_ok"] and u_mapped)

    # Optional derived η — dark unless gate passes. Still PREDICTED; never eta_from_cfd.
    if u_mapped and w_ft is not None and p1a and T1a and p3a:
        cp, _ = _cp_r(gamma, rspec)
        # relative total enthalpy drop available (impulse scoop uses KE; ts uses exit p)
        h01 = cp * T0_rel(T1a, W1a or 0.0, gamma, rspec)
        # isentropic to exit static p3 from inlet relative total
        p01 = row.get("p0_rel_1")
        if p01 and p01 > 0 and p3a > 0:
            T3s = (h01 / cp) * (float(p3a) / float(p01)) ** ((gamma - 1.0) / gamma)
            dh_ts = h01 - cp * T3s
            # total-to-total isentropic to p03
            p03 = row.get("p0_rel_3")
            dh_tt = None
            if p03 and p03 > 0:
                T03s = (h01 / cp) * (float(p03) / float(p01)) ** ((gamma - 1.0) / gamma)
                dh_tt = h01 - cp * T03s
            if row["publish_eta"]:
                if dh_ts and abs(dh_ts) > 1e-9:
                    row["eta_ts_PREDICTED"] = float(w_ft) / dh_ts
                if dh_tt and abs(dh_tt) > 1e-9:
                    row["eta_tt_PREDICTED"] = float(w_ft) / dh_tt
            else:
                # dark η: keep keys, values None — residual still published
                row["eta_ts_PREDICTED"] = None
                row["eta_tt_PREDICTED"] = None
                row["eta_dark_reason"] = (
                    "gate failed: need Ft plateau + sample_on_wall + |w_Ft−w_gas| "
                    f"≤ {work_match_tol}"
                )

    return {
        "row_efficiency_PREDICTED": row,
        "eta_from_cfd": None,
        "predicted": True,
        "methods": {
            "8": "Y_rel=(p01rel−p03rel)/(p03rel−p3); ζ_rel=1−(W3/W3is)² from p,T,W→p0,rel",
            "7": "delta_s=Cp ln(T3/T1)−R ln(p3/p1) mass-avg",
            "6": "w_Ft=Ft*U/mdot ; w_gas=U*(Wy1−Wy3) ; residual honesty gate",
            "impulse_check": "p3/p2 and W3/W1 only — not η",
            "scoping_only": "1D meanline / Ainley–Mathieson — not CFD η",
        },
    }


CSV_COLUMNS = [
    "geom_hash",
    "p1_pa",
    "T1_K",
    "W1_m_s",
    "beta1_deg",
    "gas_gamma",
    "gas_R",
    "Y_rel",
    "zeta_rel",
    "delta_s",
    "w_Ft",
    "w_gas",
    "residual",
    "p3_over_p2",
    "W3_over_W1",
    "plateau",
    "sample_on_wall",
    "work_match_ok",
    "gate_ok",
    "predicted",
]


def append_efficiency_csv(
    out_path: Path | str,
    row: dict[str, Any],
) -> Path:
    """Append one efficiency CSV line (create with header if missing)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.is_file() or out_path.stat().st_size == 0
    with out_path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k) for k in CSV_COLUMNS})
    return out_path


def run_efficiency_on_case(
    case_dir: Path | str,
    job: dict[str, Any] | None = None,
    *,
    forces: dict[str, Any] | None = None,
    flags: dict[str, Any] | None = None,
    csv_path: Path | str | None = None,
) -> dict[str, Any]:
    """Compute + optionally append CSV. Safe on non-plateau (stamps gate flags)."""
    result = compute_efficiency_of(case_dir, job, forces=forces, flags=flags)
    row = result["row_efficiency_PREDICTED"]
    if csv_path is not None:
        result["csv_path"] = str(append_efficiency_csv(csv_path, row))
    return result
