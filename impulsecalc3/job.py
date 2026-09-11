"""One JSON job. One machine. No second setpoint.

Geometry knobs (v2 names aliased) write the metal. Packing knobs write pitch.
beta1/beta2 come from job geometry (meanline), not a second engine.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

FORMAT = "impulsecalc3_job_v1"

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "configs" / "marlin_v2_rotor.json"
LIVE_OUTPUT = (ROOT / "output").resolve()
LIVE_MARLIN_REPORT = LIVE_OUTPUT / "marlin_v2_rotor_report.json"

# v2 calcbody §2 names → ImpulseCalc3 canonical keys.
# Applied only when the canonical key is missing, so a copied v2-intent JSON works.
_GEOM_ALIASES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("upper_h", "outer_sagitta_c", "h_upper_c", "hu_c", "hu"), "upper_sagitta_c"),
    (("lower_h", "inner_sagitta_c", "h_lower_c", "hl_c", "hl"), "lower_sagitta_c"),
    (("family", "profile"), "profile_family"),
    (("Z", "n_blades", "nBlades"), "n_blades_machine"),
    (("sigma", "c_over_s"), "solidity"),
    (("chord",), "chord_m"),
    (("rm", "r_m", "r", "mean_radius"), "mean_radius_m"),
    (("rt", "tip_radius", "rotor_tip_radius"), "rotor_tip_radius_m"),
    (("rh", "hub_radius"), "hub_radius_m"),
    (("span", "h", "span_h"), "span_m"),
    (("beta1", "beta1_deg", "beta1_metal_deg"), "beta1_flow_deg"),
    (("beta2", "beta2_deg", "beta2_metal_deg"), "beta2_flow_deg"),
    (("t_c", "tc", "thickness_ratio", "t/c"), "thickness_c"),
    (("le", "le_fillet", "le_r_c"), "le_fillet_r_c"),
    (("te", "te_fillet", "te_r_c"), "te_fillet_r_c"),
    (("points",), "profile_points"),
    (("stagger",), "stagger_deg"),
)


def apply_geometry_aliases(g: dict[str, Any]) -> dict[str, Any]:
    """Map v2 knob names onto canonical keys. Does not overwrite a set canonical."""
    if "c" in g and ("chord_m" not in g or g.get("chord_m") in (None, "")):
        try:
            cv = float(g["c"])
            if 1e-4 <= cv <= 2.0:
                g["chord_m"] = cv
        except (TypeError, ValueError):
            pass
    for alts, canon in _GEOM_ALIASES:
        if canon in g and g[canon] not in (None, ""):
            continue
        for a in alts:
            if a in g and g[a] not in (None, ""):
                g[canon] = g[a]
                break
    # Absolute millimetre metal (does not zoom with chord).
    _mm = (
        (("hu_mm", "upper_sagitta_mm"), "upper_sagitta_m"),
        (("hl_mm", "lower_sagitta_mm"), "lower_sagitta_m"),
        (("le_mm", "le_fillet_mm", "r_le_mm"), "le_fillet_r_m"),
        (("te_mm", "te_fillet_mm", "r_te_mm"), "te_fillet_r_m"),
    )
    for srcs, dst in _mm:
        if g.get(dst) not in (None, ""):
            continue
        for s in srcs:
            if g.get(s) not in (None, ""):
                g[dst] = float(g[s]) * 1e-3
                break
    return g


def absorb_v2_shape(job: dict[str, Any]) -> dict[str, Any]:
    """If a copied v2 JSON has blade_shape, fold those knobs into geometry."""
    g = job.setdefault("geometry", {})
    shape = job.get("blade_shape")
    if isinstance(shape, dict):
        for k, v in shape.items():
            if k not in g or g[k] in (None, ""):
                g[k] = v
    inner = g.get("blade_shape")
    if isinstance(inner, dict):
        for k, v in inner.items():
            if k not in g or g[k] in (None, ""):
                g[k] = v
    return job


def apply_packing(g: dict[str, Any], *, driver: str | None = None) -> dict[str, Any]:
    """Keep chord, solidity σ, Z, pitch, rm/rt/rh, span consistent.

    Mesh pitch is chord/solidity (the 2D cascade strip).
    Machine pitch is 2π rm / Z.
    driver='z'     → σ and pitch from Z (knob Z moved).
    driver='sigma' → pitch from σ (knob σ moved).
    default        → if σ present, pitch = c/σ (Marlin-safe); else from Z.
    """
    rt = g.get("rotor_tip_radius_m")
    rh = g.get("hub_radius_m")
    rm = g.get("mean_radius_m")
    span = g.get("span_m")
    try:
        rt_f = float(rt) if rt not in (None, "") else None
    except (TypeError, ValueError):
        rt_f = None
    try:
        rh_f = float(rh) if rh not in (None, "") else None
    except (TypeError, ValueError):
        rh_f = None
    try:
        rm_f = float(rm) if rm not in (None, "") else None
    except (TypeError, ValueError):
        rm_f = None
    try:
        span_f = float(span) if span not in (None, "") else None
    except (TypeError, ValueError):
        span_f = None
    if rm_f is None and rt_f is not None and rh_f is not None:
        rm_f = 0.5 * (rt_f + rh_f)
        g["mean_radius_m"] = rm_f
    if span_f is None and rt_f is not None and rh_f is not None:
        span_f = abs(rt_f - rh_f)
        g["span_m"] = span_f
    if rt_f is None and rm_f is not None and span_f is not None:
        g["rotor_tip_radius_m"] = rm_f + 0.5 * span_f
    if rh_f is None and rm_f is not None and span_f is not None:
        g["hub_radius_m"] = rm_f - 0.5 * span_f
    rm_f = float(g["mean_radius_m"]) if g.get("mean_radius_m") not in (None, "") else None

    c = g.get("chord_m")
    sigma = g.get("solidity")
    z = g.get("n_blades_machine")
    try:
        c_f = float(c) if c not in (None, "") else None
    except (TypeError, ValueError):
        c_f = None
    try:
        sig_f = float(sigma) if sigma not in (None, "") else None
    except (TypeError, ValueError):
        sig_f = None
    try:
        z_i = int(z) if z not in (None, "") else None
    except (TypeError, ValueError):
        z_i = None

    drv = (driver or g.get("packing_driver") or "").lower().strip()
    if drv in ("z", "blades", "n_blades", "n_blades_machine"):
        if not (z_i and z_i >= 1 and rm_f and c_f):
            raise ValueError("packing from Z needs n_blades_machine, mean_radius_m, chord_m")
        pitch = 2.0 * math.pi * rm_f / float(z_i)
        g["pitch_m"] = pitch
        g["solidity"] = c_f / pitch
        g["n_blades_machine"] = z_i
        g["packing_driver"] = "z"
        return g
    if drv in ("pitch", "spacing", "s_mm", "pitch_m"):
        try:
            p_f = float(g["pitch_m"]) if g.get("pitch_m") not in (None, "") else None
        except (TypeError, ValueError):
            p_f = None
        if not (p_f and p_f > 0 and c_f):
            raise ValueError("packing from spacing needs pitch_m and chord_m")
        g["pitch_m"] = p_f
        g["solidity"] = c_f / p_f
        if rm_f and p_f > 0:
            g["n_blades_machine"] = max(3, int(round(2.0 * math.pi * rm_f / p_f)))
        g["packing_driver"] = "pitch"
        return g
    if drv in ("sigma", "solidity"):
        if not (sig_f and sig_f > 0 and c_f):
            raise ValueError("packing from σ needs solidity and chord_m")
        pitch = c_f / sig_f
        g["pitch_m"] = pitch
        g["solidity"] = sig_f
        if rm_f and pitch > 0:
            g["n_blades_machine"] = max(3, int(round(2.0 * math.pi * rm_f / pitch)))
        g["packing_driver"] = "sigma"
        return g

    # Default: explicit σ is the cascade pitch (do not smash Marlin's table σ from Z).
    if sig_f and sig_f > 0 and c_f:
        g["pitch_m"] = c_f / sig_f
        g["solidity"] = sig_f
    elif z_i and z_i >= 1 and rm_f and c_f:
        pitch = 2.0 * math.pi * rm_f / float(z_i)
        g["pitch_m"] = pitch
        g["solidity"] = c_f / pitch
        g["n_blades_machine"] = z_i
    if g.get("n_blades_machine") in (None, "") and rm_f and g.get("pitch_m"):
        g["n_blades_machine"] = max(3, int(round(2.0 * math.pi * rm_f / float(g["pitch_m"]))))
    return g


def set_n_blades_machine(g: dict[str, Any], z: int) -> dict[str, Any]:
    """Knob Z moved: recompute pitch = 2π rm / Z and σ = c/pitch. Remeshes the strip."""
    g["n_blades_machine"] = int(z)
    g["packing_driver"] = "z"
    return apply_packing(g, driver="z")


def set_solidity(g: dict[str, Any], sigma: float) -> dict[str, Any]:
    """Knob σ moved: recompute pitch = c/σ. Remeshes the strip."""
    g["solidity"] = float(sigma)
    g["packing_driver"] = "sigma"
    return apply_packing(g, driver="sigma")


def set_pitch(g: dict[str, Any], pitch_m: float) -> dict[str, Any]:
    """Knob s (blade spacing) moved: pitch is the 2D cyclic period. σ and Z derived."""
    g["pitch_m"] = float(pitch_m)
    g["packing_driver"] = "pitch"
    return apply_packing(g, driver="pitch")



def perfect_gas_rho_kg_m3(p_pa: float, r_specific_j_kg_k: float, t_k: float) -> float:
    """Ideal-gas density rho_default = p / (R T). Does not write job knobs."""
    denom = float(r_specific_j_kg_k) * float(t_k)
    if denom == 0.0:
        raise ValueError("R T is zero")
    return float(p_pa) / denom


def eos_residual_frac(p_pa: float, rho_kg_m3: float, r_specific_j_kg_k: float, t_k: float) -> float:
    """|p − ρ R T| / p. Independent knobs; does not rewrite p, ρ, T, or R."""
    p = float(p_pa)
    if p == 0.0:
        raise ValueError("p is zero")
    return abs(p - float(rho_kg_m3) * float(r_specific_j_kg_k) * float(t_k)) / abs(p)


def eos_residual_pct(p_pa: float, rho_kg_m3: float, r_specific_j_kg_k: float, t_k: float) -> float:
    return 100.0 * eos_residual_frac(p_pa, rho_kg_m3, r_specific_j_kg_k, t_k)


def cascade_view_xlim_m(job: dict[str, Any]) -> tuple[float, float]:
    """Cascade plot crop x in [-1.5c, 2c]. Dump to x_dn_c=6 is computational, not the view."""
    c = float(job["geometry"]["chord_m"])
    return (-1.5 * c, 2.0 * c)


def load_job(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_PATH
    data = json.loads(p.read_text(encoding="utf-8"))
    return validate_job(data, source=str(p))


def validate_job(job: dict[str, Any], source: str = "") -> dict[str, Any]:
    if job.get("format") != FORMAT:
        raise ValueError(f"job format must be {FORMAT}, got {job.get('format')!r} ({source})")
    for key in ("name", "geometry", "gas", "cfd", "engine"):
        if key not in job:
            raise ValueError(f"job missing {key}")
    absorb_v2_shape(job)
    g = job["geometry"]
    apply_geometry_aliases(g)
    apply_packing(g, driver=g.get("packing_driver"))
    gas = job["gas"]
    for k in (
        "rotor_tip_radius_m",
        "hub_radius_m",
        "mean_radius_m",
        "span_m",
        "n_blades_machine",
        "n_blades_cascade",
        "chord_m",
        "solidity",
        "beta1_flow_deg",
        "beta2_flow_deg",
    ):
        if k not in g:
            raise ValueError(f"geometry missing {k}")
    for k in (
        "p1_pa",
        "t1_k",
        "gamma",
        "r_specific_j_kg_k",
        "mu_pa_s",
        "w1_m_s",
        "blade_speed_u_m_s",
    ):
        if k not in gas:
            raise ValueError(f"gas missing {k}")
    # rho stored if typed; if missing, EOS default p/(R T). Never overwrite a typed rho.
    p1_gas = float(gas["p1_pa"])
    t1_gas = float(gas["t1_k"])
    r_gas = float(gas["r_specific_j_kg_k"])
    held = bool(gas.get("rho_held"))
    gas["rho_held"] = held
    if (not held) or gas.get("rho1_kg_m3") in (None, ""):
        gas["rho1_kg_m3"] = perfect_gas_rho_kg_m3(p1_gas, r_gas, t1_gas)
    if int(g["n_blades_cascade"]) != 3:
        raise ValueError("this build requires n_blades_cascade = 3")
    # Pc must not be used as cascade inlet
    pc = job["engine"].get("Pc_pa") or (float(job["engine"].get("Pc_bar", 0)) * 1e5)
    p1 = float(gas["p1_pa"])
    if pc and abs(p1 - pc) / max(pc, 1) < 0.05:
        raise ValueError("p1 looks like Pc — do not put chamber pressure into the cascade inlet")
    if not job.get("predicted", True):
        job["predicted"] = True
    gas.setdefault("predicted", True)
    g.setdefault("incidence_deg", 0.0)
    g.setdefault("deviation_deg", 0.0)
    g.setdefault("thickness_c", 0.12)
    # Fillet aliases already mapped; LE/TE radius fallbacks for the foil family.
    if "le_fillet_r_c" in g:
        g.setdefault("le_radius_c", g["le_fillet_r_c"])
    if "te_fillet_r_c" in g:
        g.setdefault("te_radius_c", g["te_fillet_r_c"])
    g.setdefault("le_radius_c", 0.03)
    g.setdefault("te_radius_c", 0.012)
    g.setdefault("le_fillet_r_c", g.get("le_radius_c", 0.03))
    g.setdefault("te_fillet_r_c", g.get("te_radius_c", 0.012))
    inc = float(g.get("incidence_deg", 0.0))
    dev = float(g.get("deviation_deg", 0.0))
    b1_m = float(g["beta1_flow_deg"]) - inc
    b2_m = float(g["beta2_flow_deg"]) + dev
    g.setdefault("stagger_deg", 0.5 * (b1_m + b2_m))
    cfd = job["cfd"]
    cfd.setdefault("solver", "rhoCentralFoam")
    cfd.setdefault("openfoam", "ESI-v2412")
    cfd.setdefault("wall", "noSlip")
    cfd.setdefault("turbulence", "laminar")
    cfd.setdefault("n_chords_min", 10.0)
    cfd.setdefault("inlet_bc", "static_rel")  # set total_rel or gas.pt_rel_pa for locked Pt,rel inlet
    cfd.setdefault("mesh", "body_fitted_OH")
    cfd.setdefault("z_thick_m", 0.001)
    cfd.setdefault("n_around", 56)
    cfd.setdefault("n_radial", 12)
    cfd.setdefault("n_inlet", 22)
    cfd.setdefault("n_outlet", 28)
    cfd.setdefault("n_cyclic", 16)
    cfd.setdefault("x_up_c", 1.5)
    cfd.setdefault("x_dn_c", 6.0)
    cfd.setdefault("outlet_p", "waveTransmissive")
    cfd.setdefault("max_co", 0.2)
    cfd.setdefault("stretch", 1.35)
    # Inlet H axial geometric pack toward LE (soft-capped in mesh; dump still uses stretch).
    cfd.setdefault("inlet_stretch", 1.12)
    cfd.setdefault("n_pitch_fill", 7)
    # Streamwise LE clustering on body_fitted wall ring (1=uniform). Optional n_le raises west share.
    cfd.setdefault("le_cluster", 2.5)
    cfd.setdefault("n_le", 14)
    # hybrid_OH_tri soft size field (m). None → scale from g_min in mesh writer.
    cfd.setdefault("h_le", None)
    cfd.setdefault("h_pass", None)
    cfd.setdefault("h_far", None)
    cfd.setdefault("growth", 1.25)
    # Axial dense-start station upstream of LE (fraction of chord); hybrid size ramp.
    cfd.setdefault("x_dense_c", 0.25)
    job.setdefault("output_dir", "output")
    job.setdefault("geometry_test", False)
    g = job["geometry"]
    if g.get("profile_points"):
        from .geometry import profile_from_points

        profile_from_points(g["profile_points"])
        g.setdefault("profile_family", "profile_points")
    else:
        fam = str(g.get("profile_family") or "").lower().strip()
        if fam in ("impulse_bucket", "dual_arc", "pelton", "bucket"):
            g["profile_family"] = "impulse_bucket"
            g.setdefault("upper_sagitta_c", 0.58)
            g.setdefault("lower_sagitta_c", 0.44)
            g.setdefault("le_fillet_r_c", 0.04)
            g.setdefault("te_fillet_r_c", 0.04)
        elif fam in ("goldman", "goldman_vortex", "vortex_impulse"):
            # Internal name only — live outline is the dual-arc / pointed writer.
            g["profile_family"] = "impulse_bucket"
            g.setdefault("upper_sagitta_c", 0.50)
            g.setdefault("lower_sagitta_c", 0.22)
            g.setdefault("le_fillet_r_c", 0.04)
            g.setdefault("te_fillet_r_c", 0.04)
        elif fam in ("circular_arc", "circular_arc_camber_metal_angles", "airfoil", "foil", "naca"):
            g["profile_family"] = "circular_arc_camber_metal_angles"
        else:
            # Marlin JSON has no family → circular-arc foil. App / geom tests set family explicitly.
            g.setdefault("profile_family", "circular_arc_camber_metal_angles")
    return job


def pitch_m(job: dict[str, Any]) -> float:
    g = job["geometry"]
    if g.get("pitch_m") not in (None, ""):
        try:
            p = float(g["pitch_m"])
            if p > 0:
                return p
        except (TypeError, ValueError):
            pass
    return float(g["chord_m"]) / float(g["solidity"])


def domain_x(job: dict[str, Any]) -> tuple[float, float]:
    c = float(job["geometry"]["chord_m"])
    cfd = job["cfd"]
    return (-float(cfd["x_up_c"]) * c, c + float(cfd["x_dn_c"]) * c)


def is_live_marlin_job(job: dict[str, Any], out_root: Path) -> bool:
    """True only for the on-disk 80 µs Marlin case/report tree."""
    return str(job.get("name")) == "marlin_v2_rotor" and Path(out_root).resolve() == LIVE_OUTPUT


def remesh_live_marlin_allowed() -> bool:
    return os.environ.get("IMPULSECALC3_REMESH_MARLIN", "").strip().lower() in ("1", "true", "yes")
