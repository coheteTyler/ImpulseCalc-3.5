"""Relative-frame triangles, Euler work, and a named Ainley–Mathieson placeholder.

CFD does not invent η. This module never writes eta_design_proxy into an
OpenFOAM folder. Loss-book numbers are PREDICTED until a signed GG station.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Meanline:
    beta1_flow_deg: float
    beta2_flow_deg: float
    beta1_metal_deg: float
    beta2_metal_deg: float
    incidence_deg: float
    deviation_deg: float
    stagger_deg: float
    w1_m_s: float
    w2_m_s: float  # |W2|=|W1| for the impulse triangle used here
    u_m_s: float
    wx1: float
    wy1: float
    wx2: float
    wy2: float
    c_theta1: float
    c_theta2: float
    euler_work_j_kg: float
    rho1_kg_m3: float
    a_m_s: float
    Mw1: float
    Re_c: float
    annulus_area_m2: float
    mass_flow_kg_s: float  # from 2π r_m h ρ |Wa| — PREDICTED
    power_w: float  # mdot * Euler — PREDICTED
    c1_m_s: float
    c2_m_s: float
    alpha1_abs_deg: float
    alpha2_abs_deg: float
    cx1: float
    cy1: float
    cx2: float
    cy2: float
    ainley_mathieson: dict[str, Any]
    loss_scoping: dict[str, Any]
    predicted: bool
    notes: list[str]
    Max1: float  # Mw1 * cos(β1), β from axial
    psi: float  # Euler Δh0 / U²
    phi: float  # Wx / U
    U_over_C1: float
    unique_incidence_active: bool
    mdot_open_kg_s: float
    mdot_gg_kg_s: float | None
    blockage_implied: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _yp_am_nozzle(s_c: float, beta2_abs_deg: float) -> float:
    """Placeholder fit to Ainley–Mathieson 1951 nozzle (α1=0) chart. PREDICTED."""
    t = min(max(abs(beta2_abs_deg) / 90.0, 0.0), 1.2)
    yp_min = 0.012 + 0.038 * t * t
    sc_opt = 0.80 - 0.20 * t
    return max(yp_min + 0.08 * (s_c - sc_opt) ** 2, 0.008)


def _yp_am_impulse(s_c: float, beta2_abs_deg: float) -> float:
    """Placeholder fit to Ainley–Mathieson 1951 impulse chart. PREDICTED."""
    t = min(max(abs(beta2_abs_deg) / 90.0, 0.0), 1.2)
    yp_min = 0.025 + 0.060 * t * t
    sc_opt = 0.65 - 0.15 * t
    return max(yp_min + 0.12 * (s_c - sc_opt) ** 2, 0.012)


def ainley_mathieson_profile_loss(
    *,
    beta1_deg: float,
    beta2_deg: float,
    solidity: float,
) -> dict[str, Any]:
    """Named loss book: Ainley–Mathieson (1951) profile loss Yp. PREDICTED.

    Interpolation between nozzle and impulse charts (Dixon / AM 1951):
        Yp = [Yp_n + (β1/β2)² Yp_i] / [1 + (β1/β2)²]
    This is a placeholder evaluation of that book, not a signed loss or an η.
    """
    s_c = 1.0 / max(solidity, 1e-9)
    b1 = math.radians(beta1_deg)
    b2 = math.radians(beta2_deg)
    ratio = b1 / b2 if abs(b2) > 1e-9 else 0.0
    yp_n = _yp_am_nozzle(s_c, abs(beta2_deg))
    yp_i = _yp_am_impulse(s_c, abs(beta2_deg))
    r2 = ratio * ratio
    yp = (yp_n + r2 * yp_i) / (1.0 + r2)
    return {
        "book": "Ainley-Mathieson-1951",
        "predicted": True,
        "s_over_c": s_c,
        "Yp_nozzle_chart": yp_n,
        "Yp_impulse_chart": yp_i,
        "Yp_profile": yp,
        "note": (
            "PREDICTED profile-loss coefficient from the named AM 1951 book. "
            "Not from CFD. Not a stage efficiency. Not Kacker-Okapuu unless "
            "a later ticket adds that book as a second named comparison."
        ),
    }



FT_TO_M = 0.3048

# ORBIT PURPL triangle sheet (2026-09-01). UNSIGNED fixture. Not Marlin live gas.
PURPL_SHEET_FT = {
    "c1_ft_s": 5467.78,
    "c2_ft_s": 3904.56,
    "w1_ft_s": 4677.92,
    "w2_ft_s": 4677.92,
    "u_ft_s": 829.72,
    "alpha1_abs_deg": 73.56,
    "alpha2_abs_deg": 23.35,
    "beta_rel_deg": 70.681,
}


def purpl_sheet_si() -> dict[str, float]:
    s = PURPL_SHEET_FT
    return {
        "c1_m_s": s["c1_ft_s"] * FT_TO_M,
        "c2_m_s": s["c2_ft_s"] * FT_TO_M,
        "w1_m_s": s["w1_ft_s"] * FT_TO_M,
        "w2_m_s": s["w2_ft_s"] * FT_TO_M,
        "u_m_s": s["u_ft_s"] * FT_TO_M,
        "alpha1_abs_deg": s["alpha1_abs_deg"],
        "alpha2_abs_deg": s["alpha2_abs_deg"],
        "beta_rel_deg": s["beta_rel_deg"],
    }


def scoping_loss_stack(am: dict[str, Any]) -> dict[str, Any]:
    """Safety-case loss column. PREDICTED. Not CFD. Not STAND-true."""
    yp = float(am.get("Yp_profile") or 0.0)
    return {
        "authority": "SCOPING",
        "predicted": True,
        "ko_available": False,
        "profile": {"Yp": yp, "book": "Ainley-Mathieson-1951 placeholder", "predicted": True},
        "shock": {"status": "not_implemented", "note": "Kacker-Okapuu shock term not wired"},
        "tip_secondary": {"status": "not_implemented", "note": "unshrouded tip / secondary — no KO yet"},
        "friction": {"status": "not_implemented", "note": "skin friction not a second η"},
        "leading_edge": {"status": "not_implemented", "note": "LE loss not a NACA chart ingest"},
        "note": "Column A (safety). OpenFOAM is column B. Do not paint this as FIELD_CFD.",
    }


def compute_meanline(job: dict[str, Any]) -> Meanline:
    g = job["geometry"]
    gas = job["gas"]
    b1_flow = float(g["beta1_flow_deg"])
    b2_flow = float(g["beta2_flow_deg"])
    inc = float(g.get("incidence_deg", 0.0))
    dev = float(g.get("deviation_deg", 0.0))
    b1_m = b1_flow - inc
    b2_m = b2_flow + dev
    w1 = float(gas["w1_m_s"])
    u = float(gas["blade_speed_u_m_s"])
    p1 = float(gas["p1_pa"])
    t1 = float(gas["t1_k"])
    gamma = float(gas["gamma"])
    r = float(gas["r_specific_j_kg_k"])
    mu = float(gas["mu_pa_s"])
    chord = float(g["chord_m"])
    solidity = float(g["solidity"])
    span = float(g["span_m"])
    rm = float(g["mean_radius_m"])

    rad1, rad2 = math.radians(b1_flow), math.radians(b2_flow)
    wx1, wy1 = w1 * math.cos(rad1), w1 * math.sin(rad1)
    # Pure-impulse triangle: |W2|=|W1|, flow β2 as given.
    w2 = w1
    wx2, wy2 = w2 * math.cos(rad2), w2 * math.sin(rad2)
    # Cθ = Wθ + U  (U along +y)
    ct1 = wy1 + u
    ct2 = wy2 + u
    euler = u * (ct1 - ct2)
    cx1, cy1 = wx1, ct1
    cx2, cy2 = wx2, ct2
    c1 = math.hypot(cx1, cy1)
    c2 = math.hypot(cx2, cy2)
    a1 = math.degrees(math.atan2(cy1, cx1))
    a2 = math.degrees(math.atan2(cy2, cx2))

    if bool(gas.get("rho_held")) and gas.get("rho1_kg_m3") not in (None, ""):
        rho = float(gas["rho1_kg_m3"])
    else:
        rho = p1 / (r * t1)
    a = math.sqrt(gamma * r * t1)
    re = rho * w1 * chord / max(mu, 1e-16)
    annulus = 2.0 * math.pi * rm * span
    wa = abs(wx1)
    mdot = rho * wa * annulus
    power = mdot * euler
    mw1 = w1 / a if a else 0.0
    max1 = mw1 * math.cos(rad1)
    psi = euler / (u * u) if abs(u) > 1e-12 else 0.0
    phi = wx1 / u if abs(u) > 1e-12 else 0.0
    u_c1 = u / c1 if c1 else 0.0
    ui_active = abs(max1) < 1.0 and mw1 >= 1.0
    mdot_gg = None
    for src in (gas.get("mdot_gg_kg_s"), (job.get("engine") or {}).get("mdot_gg_kg_s")):
        if src not in (None, ""):
            mdot_gg = float(src)
            break
    blockage = None
    if mdot_gg is not None and mdot > 1e-12:
        blockage = 1.0 - (mdot_gg / mdot)

    am = ainley_mathieson_profile_loss(
        beta1_deg=b1_flow, beta2_deg=b2_flow, solidity=solidity
    )
    notes = [
        "Relative cascade: inlet W1 at flow β1. U is for triangles/Euler only.",
        "Metal β* = flow β − i / + δ. Geometry uses metal. i=δ=0 ⇒ β*=β.",
        "Euler work U(Cθ1−Cθ2) from the PREDICTED triangle. Not from CFD.",
        "Ainley–Mathieson Yp is a named loss-book placeholder. PREDICTED.",
        "2D CFD must not invent stage efficiency.",
        "W1/U/p1/T1/gas are PREDICTED until ORBIT signs a GG-to-rotor station.",
        "Pc of the TCA is not the cascade inlet.",
        "ṁ_gg is a cycle-card number, not a cascade inlet BC.",
        "Goldman designer (if used) takes γ and Mw1 only — not U, ṁ, or OF.",
    ]
    return Meanline(
        beta1_flow_deg=b1_flow,
        beta2_flow_deg=b2_flow,
        beta1_metal_deg=b1_m,
        beta2_metal_deg=b2_m,
        incidence_deg=inc,
        deviation_deg=dev,
        stagger_deg=0.5 * (b1_m + b2_m),
        w1_m_s=w1,
        w2_m_s=w2,
        u_m_s=u,
        wx1=wx1,
        wy1=wy1,
        wx2=wx2,
        wy2=wy2,
        c_theta1=ct1,
        c_theta2=ct2,
        euler_work_j_kg=euler,
        rho1_kg_m3=rho,
        a_m_s=a,
        Mw1=mw1,
        Re_c=re,
        annulus_area_m2=annulus,
        mass_flow_kg_s=mdot,
        power_w=power,
        c1_m_s=c1,
        c2_m_s=c2,
        alpha1_abs_deg=a1,
        alpha2_abs_deg=a2,
        cx1=cx1,
        cy1=cy1,
        cx2=cx2,
        cy2=cy2,
        ainley_mathieson=am,
        loss_scoping=scoping_loss_stack(am),
        predicted=True,
        notes=notes,
        Max1=max1,
        psi=psi,
        phi=phi,
        U_over_C1=u_c1,
        unique_incidence_active=ui_active,
        mdot_open_kg_s=mdot,
        mdot_gg_kg_s=mdot_gg,
        blockage_implied=blockage,
    )
