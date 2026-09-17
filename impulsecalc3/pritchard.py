"""Pritchard 11-parameter axial turbine airfoil (ASME 85-GT-219).

Adapted from David Poves' MIT Python port of L.J. Pritchard's eleven-parameter
axial turbine airfoil geometry model:
  https://github.com/DavidPoves/11-Parameters-Turbine-Blade-Generator

Interactive matplotlib/pandas I/O removed. Returns a closed 2-D polyline for
ImpulseCalc3 outline + cassette mesh. Default TE is rounded (straight_te=False).
LE/TE fillets are driven only by absolute le_r / te_r (mm or m).

MIT License — Copyright (c) 2020 David Poves

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

# Canonical FIELD-path family id is pritchard_11 (aliases kept for import JSON).
PRITCHARD_FAMILIES = frozenset(
    {
        "pritchard_11",
        "pritchard11",
        "pritchard",
        "eleven_parameter",
        "11param",
    }
)
CANONICAL_FAMILY = "pritchard_11"


@dataclass
class PritchardParams:
    """Eleven geometric parameters. Angles in radians; lengths in metres."""

    radius: float
    axial_chord: float
    tangential_chord: float
    unguided_turning: float
    inlet_blade: float
    inlet_half_wedge: float
    le_r: float
    outlet_blade: float
    te_r: float
    n_blades: int
    throat: float
    n_points: int = 100

    @property
    def pitch(self) -> float:
        return (2.0 * math.pi * self.radius) / max(int(self.n_blades), 1)

    @property
    def chord(self) -> float:
        return math.hypot(self.axial_chord, self.tangential_chord)

    @property
    def stagger_rad(self) -> float:
        return math.atan2(self.tangential_chord, self.axial_chord)


def _third_poly_coeffs(
    x1: float, y1: float, beta_1: float, x2: float, y2: float, beta_2: float
) -> tuple[float, float, float, float]:
    d = (math.tan(beta_1) + math.tan(beta_2)) / (x1 - x2) ** 2 - 2.0 * (y1 - y2) / (x1 - x2) ** 3
    c = (y1 - y2) / (x1 - x2) ** 2 - math.tan(beta_2) / (x1 - x2) - d * (x1 + 2.0 * x2)
    b = math.tan(beta_2) - 2.0 * c * x2 - 3.0 * d * x2 ** 2
    a = y2 - b * x2 - c * x2 ** 2 - d * x2 ** 3
    return a, b, c, d


def generate_pritchard_sides(
    p: PritchardParams,
    *,
    straight_te: bool = False,
    max_iter: int = 250,
    tol: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Suction and pressure polylines from LE tip toward TE (both start at LE)."""
    R = float(p.radius)
    cx = float(p.axial_chord)
    ct = float(p.tangential_chord)
    unguided = float(p.unguided_turning)
    beta_i = float(p.inlet_blade)
    epsilon_i = float(p.inlet_half_wedge)
    le_r = float(p.le_r)
    beta_o = float(p.outlet_blade)
    te_r = float(p.te_r)
    n_blades = max(int(p.n_blades), 1)
    throat = float(p.throat)
    n_points = max(5 * round(max(int(p.n_points), 25) / 5), 25)

    if cx <= 0 or le_r <= 0 or te_r <= 0 or throat <= 0:
        raise ValueError(
            f"Pritchard requires cx>0, le_r>0, te_r>0, throat>0 "
            f"(got cx={cx}, le_r={le_r}, te_r={te_r}, throat={throat})"
        )

    pitch = (2.0 * math.pi * R) / n_blades
    epsilon_o = 0.5 * unguided

    x1 = y1 = x2 = y2 = x3 = y3 = x4 = y4 = x5 = y5 = 0.0
    beta_1 = beta_2 = beta_3 = beta_4 = beta_5 = 0.0
    x0 = y0 = R0 = 0.0
    converged = False

    for _ in range(max_iter):
        beta_1 = beta_o - epsilon_o
        x1 = cx - te_r * (1.0 + math.sin(beta_1))
        y1 = te_r * math.cos(beta_1)

        beta_2 = beta_o - epsilon_o + unguided
        x2 = cx - te_r + (throat + te_r) * math.sin(beta_2)
        y2 = pitch - (throat + te_r) * math.cos(beta_2)

        beta_3 = beta_i + epsilon_i
        x3 = le_r * (1.0 - math.sin(beta_3))
        y3 = ct + le_r * math.cos(beta_3)

        beta_4 = beta_i - epsilon_i
        x4 = le_r * (1.0 + math.sin(beta_4))
        y4 = ct - le_r * math.cos(beta_4)

        beta_5 = beta_o + epsilon_o
        x5 = cx - te_r * (1.0 - math.sin(beta_5))
        y5 = -te_r * math.cos(beta_5)

        denom = math.tan(beta_2) - math.tan(beta_1)
        if abs(denom) < 1e-14:
            raise ValueError("Pritchard throat iteration: tan(beta_2)~=tan(beta_1)")
        x0 = (
            (y1 - y2) * math.tan(beta_1) * math.tan(beta_2)
            + x1 * math.tan(beta_2)
            - x2 * math.tan(beta_1)
        ) / denom
        if abs(math.tan(beta_1)) < 1e-14:
            raise ValueError("Pritchard throat iteration: tan(beta_1)~=0")
        y0 = -(x0 - x1) / math.tan(beta_1) + y1
        R0 = math.hypot(x1 - x0, y1 - y0)
        disc = R0 ** 2 - (x2 - x0) ** 2
        if disc < 0.0:
            epsilon_o *= 1.05
            if epsilon_o <= 0 or epsilon_o > math.pi / 2:
                raise TimeoutError(
                    "Pritchard: throat disc < 0. Try larger throat or smaller |beta_o| / unguided."
                )
            continue
        yy2 = y0 + math.sqrt(disc)
        if abs(y2 - yy2) < tol:
            converged = True
            break
        factor = (y2 / yy2) ** 4 if abs(yy2) > 1e-16 else 1.1
        epsilon_o *= factor
        if epsilon_o < 0:
            raise TimeoutError(
                "Pritchard: wedge-out went negative. Try reducing |exit blade angle| or throat."
            )

    if not converged:
        raise TimeoutError(f"Pritchard: throat discontinuity not closed in {max_iter} iters")

    x7 = cx - te_r
    y7 = 0.0
    x8, y8 = 0.0, ct
    x9, y9 = le_r, ct

    a_s, b_s, c_s, d_s = _third_poly_coeffs(x2, y2, beta_2, x3, y3, beta_3)
    a_p, b_p, c_p, d_p = _third_poly_coeffs(x4, y4, beta_4, x5, y5, beta_5)

    n5 = max(n_points // 5, 5)
    xs = [x8]
    ys = [y8]
    xp = [x8]
    yp = [y8]

    dxp = (x4 - x8) / max(n5 - 1, 1)
    dxs = (x3 - x8) / max(n5 - 1, 1)
    for _ in range(n5):
        xp.append(xp[-1] + dxp)
        disc_p = le_r ** 2 - (xp[-1] - x9) ** 2
        yp.append(y9 - math.sqrt(max(disc_p, 0.0)))
        xs.append(xs[-1] + dxs)
        disc_s = le_r ** 2 - (xs[-1] - x9) ** 2
        ys.append(y9 + math.sqrt(max(disc_s, 0.0)))

    dxp = (x5 - x4) / max(3 * n5, 1)
    dxs = (x2 - x3) / max(2 * n5, 1)
    for _ in range(2 * n5):
        xp.append(xp[-1] + dxp)
        yp.append(a_p + b_p * xp[-1] + c_p * xp[-1] ** 2 + d_p * xp[-1] ** 3)
        xs.append(xs[-1] + dxs)
        ys.append(a_s + b_s * xs[-1] + c_s * xs[-1] ** 2 + d_s * xs[-1] ** 3)

    dxs = (x1 - x2) / max(n5, 1)
    for _ in range(n5):
        xp.append(xp[-1] + dxp)
        yp.append(a_p + b_p * xp[-1] + c_p * xp[-1] ** 2 + d_p * xp[-1] ** 3)
        xs.append(xs[-1] + dxs)
        disc_arc = R0 ** 2 - (xs[-1] - x0) ** 2
        ys.append(y0 + math.sqrt(max(disc_arc, 0.0)))

    # Rounded TE (default). Fixed original dxs operator-precedence bug.
    if not straight_te:
        dxp = (cx - x5) / max(n5 + 1, 1)
        dxs = (cx - x1) / max(n5 + 1, 1)
        for _ in range(n5):
            xp.append(min(xp[-1] + dxp, cx))
            disc_tp = te_r ** 2 - (xp[-1] - x7) ** 2
            yp.append(y7 - math.sqrt(max(disc_tp, 0.0)))
            xs.append(min(xs[-1] + dxs, cx))
            disc_ts = te_r ** 2 - (xs[-1] - x7) ** 2
            ys.append(y7 + math.sqrt(max(disc_ts, 0.0)))

    suction = np.column_stack([np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)])
    pressure = np.column_stack([np.asarray(xp, dtype=float), np.asarray(yp, dtype=float)])
    meta = {
        "pitch": float(pitch),
        "stagger_deg": math.degrees(math.atan2(ct, cx)),
        "chord": math.hypot(cx, ct),
        "half_wedge_out_deg": math.degrees(epsilon_o),
        "R0": float(R0),
        "x0": float(x0),
        "y0": float(y0),
        "epsilon_o_rad": float(epsilon_o),
        "straight_te": 1.0 if straight_te else 0.0,
    }
    return suction, pressure, meta


def closed_pritchard_polyline(
    p: PritchardParams,
    *,
    straight_te: bool = False,
) -> list[tuple[float, float]]:
    """Closed CCW outline: suction LE->TE, then pressure TE->LE.

    If the throat-closure iteration fails (common when te_r grows at fixed throat),
    retry with a reduced throat down to ~0.12·pitch before raising.
    """
    from dataclasses import replace

    pitch = p.pitch
    attempts = [p]
    # Back off throat if needed; keep le_r/te_r as the fillet drivers.
    for frac in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.12):
        thr = frac * pitch
        if thr < p.throat:
            attempts.append(replace(p, throat=thr))
    last_err: Exception | None = None
    suction = pressure = None
    for attempt in attempts:
        try:
            suction, pressure, _meta = generate_pritchard_sides(attempt, straight_te=straight_te)
            p = attempt  # noqa: keep successful params for any caller introspection
            break
        except (TimeoutError, ValueError) as exc:
            last_err = exc
            continue
    else:
        raise TimeoutError(
            f"Pritchard geometry failed after throat backoff (te_r={p.te_r}, throat0={attempts[0].throat}): {last_err}"
        )

    pts: list[tuple[float, float]] = [(float(x), float(y)) for x, y in suction]
    for x, y in pressure[-1:0:-1]:
        pts.append((float(x), float(y)))

    cleaned: list[tuple[float, float]] = []
    for pt in pts:
        if not cleaned or math.hypot(pt[0] - cleaned[-1][0], pt[1] - cleaned[-1][1]) > 1e-12:
            cleaned.append(pt)
    if len(cleaned) < 3:
        raise ValueError("Pritchard polyline too short")
    if math.hypot(cleaned[0][0] - cleaned[-1][0], cleaned[0][1] - cleaned[-1][1]) > 1e-12:
        cleaned.append(cleaned[0])
    else:
        cleaned[-1] = cleaned[0]

    body = cleaned[:-1]
    area = 0.0
    n = len(body)
    for i in range(n):
        x1, y1 = body[i]
        x2, y2 = body[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    if area < 0:
        body = list(reversed(body))
        cleaned = body + [body[0]]
    return cleaned


def _first_num(g: dict[str, Any], *keys: str, default: float | None = None) -> float | None:
    for k in keys:
        if k in g and g[k] not in (None, ""):
            return float(g[k])
    return default


def _first_len_m(g: dict[str, Any], *keys: str, default: float | None = None) -> float | None:
    for k in keys:
        if k in g and g[k] not in (None, ""):
            v = float(g[k])
            if k.endswith("_mm"):
                return v * 1e-3
            return v
    return default


def params_from_geometry(
    g: dict[str, Any],
    *,
    pitch_m: float | None = None,
) -> PritchardParams:
    """Map ImpulseCalc3 geometry / filter knobs -> PritchardParams.

    Pritchard 11 independents (airfoil-first):
      R, cx, ct, unguided_turning, inlet_blade, inlet_wedge (εi),
      le_r, exit_blade, te_r, N, throat.
    Dependents: pitch = 2πR/N, stagger = atan(ct/cx), chord = hypot(cx, ct).

    Fillet drivers: absolute le_r / te_r (le_mm, te_mm, *_fillet_r_m).
    RTS r_le_ratio / r_te_ratio map to absolute via /chord (Pritchard chord),
    only when absolute is absent or ≤0. Stems lin/lout ignored — Goldman only.
    UI angles are degrees → radians here.
    """
    R = _first_num(g, "mean_radius_m", "radius_m", "R", "rm", default=0.0375) or 0.0375
    Z = int(_first_num(g, "n_blades_machine", "N", "Z", "n_blades", default=25) or 25)
    pitch = float(pitch_m) if pitch_m not in (None, 0) else (2.0 * math.pi * R) / max(Z, 1)

    # Independents cx, ct. Never free-rotate stagger while freezing β*.
    cx = _first_len_m(g, "cx_m", "axial_chord_m", "cx_mm", "axial_chord_mm")
    ct = _first_len_m(g, "ct_m", "tangential_chord_m", "ct_mm", "tangential_chord_mm")
    chord_in = _first_len_m(g, "chord_m", "chord", "c", "chord_mm")
    if cx is None and ct is None:
        # Fall back: chord + metal-law stagger ½(β1*+β2*) → cx, ct (β±65 ⇒ stagger 0).
        c = chord_in if chord_in is not None else 0.01
        b1 = _first_num(
            g, "beta1_metal_deg", "beta1_flow_deg", "beta1", "beta_i_deg", default=65.0
        ) or 65.0
        b2 = _first_num(
            g, "beta2_metal_deg", "beta2_flow_deg", "beta2", "beta_o_deg", default=-65.0
        ) or -65.0
        stag = 0.5 * (b1 + b2)  # metal law; NOT a free stagger knob
        sr = math.radians(float(stag))
        cx = c * math.cos(sr)
        ct = c * math.sin(sr)
    elif cx is None:
        cx = chord_in if chord_in is not None else 0.01
        ct = float(ct or 0.0)
    elif ct is None:
        ct = 0.0
    cx = max(float(cx), 1e-6)
    ct = float(ct)
    # Pritchard chord for ratio reference — always hypot(cx,ct), not a competing RTS chord.
    c_ref = math.hypot(cx, ct)

    beta_i_deg = _first_num(
        g, "beta_i_deg", "beta1_metal_deg", "beta1_flow_deg", "beta1", "beta1_deg", default=65.0
    )
    beta_o_deg = _first_num(
        g, "beta_o_deg", "beta2_metal_deg", "beta2_flow_deg", "beta2", "beta2_deg", default=-65.0
    )
    eps_i_deg = _first_num(
        g, "epsilon_i_deg", "inlet_half_wedge_deg", "epsilon_i", default=10.0
    ) or 10.0
    # Unguided turning = Pritchard uncovered turning — NOT L_out.
    ug_deg = _first_num(
        g, "unguided_turning_deg", "unguided_turning", "theta_u_deg", default=8.0
    ) or 8.0

    le_r = _first_len_m(
        g, "le_r_m", "le_fillet_r_m", "le_r_mm", "le_mm", "le_fillet_mm", "le_r",
    )
    te_r = _first_len_m(
        g, "te_r_m", "te_fillet_r_m", "te_r_mm", "te_mm", "te_fillet_mm", "te_r",
    )
    # ≤0 absolute is unset (te_mm=0 must not kill TE / block demoted ratio).
    if le_r is not None and le_r <= 0:
        le_r = None
    if te_r is not None and te_r <= 0:
        te_r = None
    if le_r is None:
        rle_c = _first_num(g, "r_le_ratio", "le_fillet_r_c", "le_radius_c", default=0.04) or 0.04
        le_r = float(rle_c) * c_ref
    if te_r is None:
        rte_c = _first_num(g, "r_te_ratio", "te_fillet_r_c", "te_radius_c", default=0.04) or 0.04
        te_r = float(rte_c) * c_ref
    floor_r = max(0.04 * c_ref, 1e-6)
    le_r = max(float(le_r), floor_r * 0.25)  # keep positive; default path already 0.04c
    te_r = max(float(te_r), floor_r * 0.25)
    # Prefer matching defaults at 0.04c when still at floor-ish from empty import
    if le_r < floor_r * 0.5:
        le_r = floor_r
    if te_r < floor_r * 0.5:
        te_r = floor_r

    throat = _first_len_m(g, "throat_m", "throat", "throat_mm")
    if throat is None:
        thr_ratio = _first_num(g, "throat_pitch_ratio", "throat_ratio", default=0.25) or 0.25
        throat = float(thr_ratio) * pitch
    throat = max(float(throat), 1e-6)

    n_pts = int(_first_num(g, "n_profile_points", "n_points", default=100) or 100)

    return PritchardParams(
        radius=R,
        axial_chord=cx,
        tangential_chord=ct,
        unguided_turning=math.radians(float(ug_deg)),
        inlet_blade=math.radians(float(beta_i_deg or 65.0)),
        inlet_half_wedge=math.radians(float(eps_i_deg)),
        le_r=float(le_r),
        outlet_blade=math.radians(float(beta_o_deg if beta_o_deg is not None else -65.0)),
        te_r=float(te_r),
        n_blades=Z,
        throat=throat,
        n_points=n_pts,
    )


def pritchard_profile_from_job(job: dict[str, Any]) -> list[tuple[float, float]]:
    """Closed Pritchard outline from a job dict. Stems (lin/lout) forced off."""
    g = dict(job.get("geometry") or {})
    g["lin_m"] = 0.0
    g["lout_m"] = 0.0
    g.pop("lin_mm", None)
    g.pop("lout_mm", None)
    pitch = None
    try:
        from .job import pitch_m as _pitch_m

        pitch = float(_pitch_m(job))
    except Exception:
        pitch = None
    params = params_from_geometry(g, pitch_m=pitch)
    straight = bool(g.get("straight_te") in (True, 1, "1", "true", "True"))
    return closed_pritchard_polyline(params, straight_te=straight)
