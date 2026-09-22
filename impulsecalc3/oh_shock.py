"""Goldman shock body-fitted O–H (all quads). Used by mesh.write_polymesh for body_fitted_OH.

Hybrid_OH_tri stays a polyline/periodicity donor only — not the SOLVE default.
"""
from __future__ import annotations

import math

import numpy as np


def y1_wall_m(
    *,
    mu_pa_s: float,
    rho1_kg_m3: float,
    w1_m_s: float,
    yplus_target: float = 1.0,
    u_tau_frac_w1: float = 0.05,
) -> tuple[float, float, str]:
    """First wall-normal spacing from y+ with u_τ ≈ frac*|W1| (skin / Blasius-class)."""
    mu = float(mu_pa_s)
    rho = max(float(rho1_kg_m3), 1e-12)
    w1 = abs(float(w1_m_s))
    ut = max(u_tau_frac_w1 * w1, 1e-6)
    y1 = float(yplus_target) * mu / (rho * ut)
    note = (
        f"y1_m={y1:.6g} from y+={yplus_target:g}, u_τ={u_tau_frac_w1:g}*|W1|="
        f"{ut:.6g} m/s (HOT μ={mu:.3g}, ρ={rho:.6g}, W1={w1:.6g})"
    )
    return y1, ut, note


def d_o_from_y1(y1: float, n_rad: int, stretch: float) -> float:
    """O collar thickness: geometric wall-normal layers from y1."""
    r = max(float(stretch), 1.0)
    n = max(int(n_rad), 1)
    if abs(r - 1.0) < 1e-12:
        return float(y1) * n
    return float(y1) * (r**n - 1.0) / (r - 1.0)


def dump_xs_rx(
    x0: float,
    *,
    rx: float = 1.18,
    dx_last: float = 1.4e-3,
    dx0: float | None = None,
    n_min: int = 8,
    n_max: int = 64,
) -> tuple[np.ndarray, float, int]:
    """Geometric dump x-nodes from x0; last cell ≈ dx_last. Returns xs, L_dump, n_cells."""
    r = max(float(rx), 1.0 + 1e-9)
    dx_l = max(float(dx_last), 1e-6)
    if dx0 is None or dx0 <= 0:
        best = None
        for n_try in range(n_min, n_max + 1):
            d0 = dx_l / (r ** (n_try - 1))
            L = d0 * (r**n_try - 1.0) / (r - 1.0)
            score = abs(L - 0.02)
            if best is None or score < best[0]:
                best = (score, n_try, d0, L)
        assert best is not None
        _, n, d0, L = best
    else:
        d0 = float(dx0)
        n = int(round(1.0 + math.log(max(dx_l / d0, 1.0)) / math.log(r)))
        n = max(n_min, min(n_max, n))
        L = d0 * (r**n - 1.0) / (r - 1.0)
    xs = [float(x0)]
    x = float(x0)
    dx = d0
    for _ in range(n):
        x += dx
        xs.append(x)
        dx *= r
    return np.asarray(xs, dtype=float), float(xs[-1] - x0), int(n)


def shear_dir(beta_deg: float) -> np.ndarray:
    """β from +x (axial), positive toward +y (pitch). ê = (cos β, sin β)."""
    rad = math.radians(float(beta_deg))
    return np.array([math.cos(rad), math.sin(rad)], dtype=float)


def _lin_dir(p0: np.ndarray, p1: np.ndarray, n_seg: int) -> np.ndarray:
    p0 = np.asarray(p0, dtype=float).reshape(2)
    p1 = np.asarray(p1, dtype=float).reshape(2)
    t = np.linspace(0.0, 1.0, n_seg + 1)
    return (1.0 - t)[:, None] * p0 + t[:, None] * p1


def sheared_inlet_edge(
    p_east: np.ndarray,
    x_in: float,
    beta1_deg: float,
) -> np.ndarray:
    """Map O-west point back to x=x_in along −W1 (sheared inlet)."""
    e = shear_dir(beta1_deg)
    ex = float(e[0])
    if abs(ex) < 1e-12:
        return np.array([x_in, float(p_east[1])], dtype=float)
    t = (float(p_east[0]) - float(x_in)) / ex
    return np.array([x_in, float(p_east[1]) - t * float(e[1])], dtype=float)


def sheared_outlet_point(
    p_west: np.ndarray,
    x_target: float,
    beta2_deg: float,
) -> np.ndarray:
    """Advance from p_west along +W2 to x=x_target (sheared wake/dump)."""
    e = shear_dir(beta2_deg)
    ex = float(e[0])
    if abs(ex) < 1e-12:
        return np.array([x_target, float(p_west[1])], dtype=float)
    t = (float(x_target) - float(p_west[0])) / ex
    return np.array(
        [float(p_west[0]) + t * float(e[0]), float(p_west[1]) + t * float(e[1])],
        dtype=float,
    )


def count_te_angular_cells(
    wall: np.ndarray,
    *,
    te_a: np.ndarray,
    te_b: np.ndarray,
    max_deg: float = 20.0,
) -> int:
    """Count wall segments in the first max_deg polar degrees off each TE cut tip."""
    wall = np.asarray(wall, dtype=float)
    n = wall.shape[0]
    if n < 3:
        return 0

    def _count_from(tip: np.ndarray, indices: range) -> int:
        pts = [wall[i] for i in indices]
        if len(pts) < 2:
            return 0
        ref = pts[0] - tip
        if float(np.hypot(ref[0], ref[1])) < 1e-16:
            if len(pts) < 2:
                return 0
            ref = pts[1] - tip
        count = 0
        for p in pts:
            v = p - tip
            ln = float(np.hypot(v[0], v[1]))
            if ln < 1e-16:
                continue
            cosang = float(np.dot(ref, v) / (np.hypot(ref[0], ref[1]) * ln + 1e-30))
            cosang = max(-1.0, min(1.0, cosang))
            ang = math.degrees(math.acos(cosang))
            if ang <= max_deg + 1e-9:
                count += 1
            else:
                break
        return count

    c0 = _count_from(np.asarray(te_a, dtype=float), range(0, min(n, 40)))
    c1 = _count_from(np.asarray(te_b, dtype=float), range(n - 1, max(-1, n - 41), -1))
    return int(c0 + c1)


def target_side_counts(
    *,
    chord_m: float,
    le_radius_m: float,
    te_radius_m: float,
    g_min: float,
    n_inlet: int,
    ds_le: float | None = None,
    ds_inner: float = 30e-6,
    n_pitchwise_throat: int = 40,
    te_angular_min: int = 10,
) -> dict[str, int]:
    """Streamwise / pitchwise side counts for Goldman shock O–H."""
    c = max(float(chord_m), 1e-6)
    ds_le = float(ds_le if ds_le is not None else 0.001 * c)
    r_le = max(float(le_radius_m), 1e-6)
    r_te = max(float(te_radius_m), 1e-6)
    n_le = max(16, int(math.ceil(math.pi * r_le / ds_le)))
    n_te = max(te_angular_min + 4, int(math.ceil(math.pi * r_te / max(ds_inner, ds_le))))
    n_inner = max(24, int(math.ceil(0.55 * c / max(ds_inner, 1e-6))))
    n_back = max(24, int(math.ceil(0.45 * c / max(ds_inner, 1e-6))))
    n_pw = max(int(n_pitchwise_throat), 8)
    n_in = max(int(n_inlet), 8)
    return {
        "n_in": n_in,
        "n_out": max(n_te, 14),
        "n_cyc": max(n_back, 20),
        "n_stem": max(n_le, n_inner // 2, 16),
        "n_fill": n_pw,
        "n_cav_x": n_pw,
        "n_le_arc": n_le,
        "n_te_arc": n_te,
        "n_inner": n_inner,
    }
