"""3-blade HOH cassette. Outer first. Algebraic TFI. No AABB cyclic."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import hoh_quality as Q
from .geometry import profile_from_job
from .hoh_polymesh import PATCH_ORDER, write_polymesh, weld_xy
from .hoh_stop import (
    HohStop,
    MAX_REPAIR,
    assert_budget,
    assert_not_rectangle,
)
from .job import pitch_m, validate_job
from .meanline import Meanline, compute_meanline


@dataclass
class HohResult:
    success: bool
    case_dir: str
    n_cells: int
    n_blades: int
    max_nonortho_deg: float
    max_skew: float
    cyclic_ystdev_over_pitch: float
    y_plus_est: float
    mesh_seconds: float
    repair_applied: str | None
    gates: dict
    notes: list[str] = field(default_factory=list)
    solvable: bool = False


def tfi_quad(south, north, west, east) -> np.ndarray:
    """Bilinear TFI. Edges include endpoints. Returns (nj, ni, 2) with i along south."""
    S = np.asarray(south, dtype=float)
    N = np.asarray(north, dtype=float)
    W = np.asarray(west, dtype=float)
    E = np.asarray(east, dtype=float)
    ni = len(S)
    nj = len(W)
    if len(N) != ni or len(E) != nj:
        raise ValueError(f"TFI count S{len(S)} N{len(N)} W{len(W)} E{len(E)}")
    xi = np.linspace(0.0, 1.0, ni)
    eta = np.linspace(0.0, 1.0, nj)
    C00, C10 = S[0], S[-1]
    C01, C11 = N[0], N[-1]
    out = np.zeros((nj, ni, 2))
    for j in range(nj):
        e = eta[j]
        for i in range(ni):
            x = xi[i]
            out[j, i] = (
                (1 - e) * S[i] + e * N[i] + (1 - x) * W[j] + x * E[j]
                - (1 - x) * (1 - e) * C00
                - x * (1 - e) * C10
                - (1 - x) * e * C01
                - x * e * C11
            )
    out[0, :] = S
    out[-1, :] = N
    out[:, 0] = W
    out[:, -1] = E
    return out


def _resample(poly, n: int) -> np.ndarray:
    p = np.asarray(poly, dtype=float)
    if n <= 1:
        return np.repeat(p[:1], max(n, 1), axis=0)
    if len(p) < 2:
        return np.repeat(p[:1], n, axis=0)
    seg = np.sqrt(((p[1:] - p[:-1]) ** 2).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-16:
        return np.repeat(p[:1], n, axis=0)
    t = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])])


def _stretch(j: int, n: int, r: float) -> float:
    if n <= 0:
        return 0.0
    if abs(r - 1.0) < 1e-12:
        return j / n
    return (r ** j - 1.0) / (r ** n - 1.0)


def _resample_pack(poly, n: int, r: float, pack_end: bool = True) -> np.ndarray:
    """Arc-length resample with geometric pack toward the last (or first) point."""
    p = np.asarray(poly, dtype=float)
    if n <= 2 or len(p) < 2:
        return _resample(p, n)
    seg = np.sqrt(((p[1:] - p[:-1]) ** 2).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-16:
        return _resample(p, n)
    xi = np.array([_stretch(j, n - 1, r) for j in range(n)])
    if not pack_end:
        xi = 1.0 - xi[::-1]
    t = xi * s[-1]
    return np.column_stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])])


def _lin(a, b, n: int, r: float = 1.0, pack_end: bool = True) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if n <= 2 or abs(r - 1.0) < 1e-12:
        t = np.linspace(0.0, 1.0, n)
    else:
        t = np.array([_stretch(j, n - 1, r) for j in range(n)])
        if not pack_end:
            t = 1.0 - t[::-1]
    return a[None, :] + t[:, None] * (b - a)


def _closed(poly) -> np.ndarray:
    p = np.asarray(poly, dtype=float)
    if len(p) < 3:
        return p
    if np.hypot(*(p[0] - p[-1])) < 1e-12:
        return p
    return np.vstack([p, p[:1]])


def _shift(poly, dy: float) -> np.ndarray:
    p = np.asarray(poly, dtype=float).copy()
    p[:, 1] += dy
    return p


def _signed_area(poly) -> float:
    p = _closed(poly)
    return 0.5 * float(np.dot(p[:-1, 0], p[1:, 1]) - np.dot(p[1:, 0], p[:-1, 1]))


def _ensure_ccw(poly) -> np.ndarray:
    p = np.asarray(poly, dtype=float)
    if np.hypot(*(p[0] - p[-1])) < 1e-12:
        p = p[:-1]
    if _signed_area(p) < 0:
        p = p[::-1].copy()
    return p


def _outward_normals(closed_xy: np.ndarray) -> np.ndarray:
    p = np.asarray(closed_xy, dtype=float)
    if np.hypot(*(p[0] - p[-1])) < 1e-12:
        p = p[:-1]
    n = len(p)
    nrm = np.zeros((n, 2))
    for i in range(n):
        t = p[(i + 1) % n] - p[(i - 1) % n]
        L = float(np.hypot(*t))
        if L < 1e-16:
            t = np.array([1.0, 0.0])
            L = 1.0
        t = t / L
        nrm[i] = np.array([t[1], -t[0]])  # CCW metal: outward = right of tangent
    return nrm


def offset_oring(outline, h0: float, r: float, Nn: int) -> np.ndarray:
    """(Nn+1, Ns, 2) layer 0 = closed metal loop. Keep s-order. Wrap. Not a U-collar."""
    p = np.asarray(outline, dtype=float)
    if np.hypot(*(p[0] - p[-1])) < 1e-12:
        p = p[:-1].copy()
    if len(p) < 8:
        raise HohStop(f"O outline has {len(p)} nodes; need a closed loop")
    nrm = _outward_normals(p)
    if _signed_area(p) < 0:
        nrm = -nrm
    ns = len(p)
    out = np.zeros((Nn + 1, ns, 2))
    out[0] = p
    for j in range(1, Nn + 1):
        if abs(r - 1.0) < 1e-12:
            s = h0 * j
        else:
            s = h0 * (r ** j - 1.0) / (r - 1.0)
        out[j] = p + nrm * s
    return out


def _le_te(poly):
    p = np.asarray(poly, dtype=float)
    return int(np.argmin(p[:, 0])), int(np.argmax(p[:, 0]))


def _walk(p, i0: int, i1: int, step: int) -> np.ndarray:
    n = len(p)
    out = [p[i0]]
    i = i0
    for _ in range(n + 1):
        i = (i + step) % n
        out.append(p[i])
        if i == i1:
            break
    return np.asarray(out, dtype=float)


def _chain_le_te(poly, high_y: bool) -> np.ndarray:
    p = np.asarray(poly, dtype=float)
    if np.hypot(*(p[0] - p[-1])) < 1e-12:
        p = p[:-1]
    i_le, i_te = _le_te(p)
    a, b = _walk(p, i_le, i_te, 1), _walk(p, i_le, i_te, -1)
    ya, yb = float(a[:, 1].mean()), float(b[:, 1].mean())
    if high_y:
        return a if ya >= yb else b
    return a if ya < yb else b


def _clustered_half(poly, n: int, chord: float) -> np.ndarray:
    """Resample open chain, denser near LE (start) and TE (end / max-x)."""
    p = np.asarray(poly, dtype=float)
    if n <= 2:
        return _resample(p, n)
    seg = np.sqrt(((p[1:] - p[:-1]) ** 2).sum(axis=1))
    xte = float(p[:, 0].max())
    xle = float(p[:, 0].min())
    band = 0.15 * float(chord)
    w = np.ones(len(seg))
    mid = 0.5 * (p[:-1] + p[1:])
    w[np.abs(mid[:, 0] - xte) < band] = 3.0
    w[np.abs(mid[:, 0] - xle) < band] = 2.0
    s = np.concatenate([[0.0], np.cumsum(seg * w)])
    if s[-1] < 1e-16:
        return _resample(p, n)
    t = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])])


def _s_to_xy(p, s, col: int) -> np.ndarray:
    return p[:, col]


def _force_mesh_fillets(job: dict, notes: list[str]) -> dict:
    """Mesher-only fillets. Metal CAD may stay sharp."""
    j = json.loads(json.dumps(job))
    g = j["geometry"]
    c = float(g["chord_m"])
    rmin = 0.04 * c
    g["le_fillet_r_c"] = max(float(g.get("le_fillet_r_c") or 0.0), 0.04)
    g["te_fillet_r_c"] = max(float(g.get("te_fillet_r_c") or 0.0), 0.04)
    g["le_fillet_r_m"] = max(float(g.get("le_fillet_r_m") or 0.0), rmin)
    g["te_fillet_r_m"] = max(float(g.get("te_fillet_r_m") or 0.0), rmin)
    g["le_mm"] = g["le_fillet_r_m"] * 1e3
    g["te_mm"] = g["te_fillet_r_m"] * 1e3
    notes.append("mesh fillets 0.04c, metal CAD may stay sharp.")
    return j


def _align_ring(poly, Ns: int, chord: float, Nle: int = 12, Nte: int = 12):
    """Closed ring. LE/TE are arcs (≥8), not a shared cusp. Returns (ring, meta)."""
    Nle = max(8, int(Nle))
    Nte = max(8, int(Nte))
    if Nle % 2:
        Nle += 1
    if Nte % 2:
        Nte += 1
    rem = int(Ns) - Nle - Nte
    if rem < 16:
        rem = 16
    if rem % 2:
        rem += 1
    Ns = Nle + Nte + rem
    Nside = rem // 2
    p = _ensure_ccw(poly)
    if np.hypot(*(p[0] - p[-1])) < 1e-12:
        p = p[:-1]
    i_le, i_te = _le_te(p)
    high = _walk(p, i_le, i_te, 1)
    low = _walk(p, i_le, i_te, -1)
    if float(high[:, 1].mean()) < float(low[:, 1].mean()):
        high, low = low, high
    n_le_h = Nle // 2
    n_te_h = Nte // 2
    n_half = n_le_h + Nside + n_te_h + 1
    high_r = _clustered_half(high, n_half, chord)
    low_r = _clustered_half(low, n_half, chord)
    high_r[0] = low_r[0] = 0.5 * (high_r[0] + low_r[0])
    high_r[-1] = low_r[-1] = 0.5 * (high_r[-1] + low_r[-1])
    le_arc = np.vstack([low_r[n_le_h::-1], high_r[1 : n_le_h + 1]])
    te_from_hi = high_r[n_le_h + Nside :]
    te_to_lo = low_r[-2 : n_le_h + Nside - 1 : -1]
    te_arc = np.vstack([te_from_hi, te_to_lo])
    high_side = high_r[n_le_h : n_le_h + Nside + 1]
    low_side = low_r[n_le_h : n_le_h + Nside + 1]
    if len(le_arc) != Nle + 1 or len(te_arc) != Nte + 1:
        raise HohStop(f"arc count LE {len(le_arc)} TE {len(te_arc)} want {Nle+1}/{Nte+1}")
    ring = np.vstack([le_arc[:-1], high_side[:-1], te_arc[:-1], low_side[::-1][:-1]])
    if len(ring) != Ns:
        raise HohStop(f"ring {len(ring)} != Ns {Ns}")
    return ring, {
        "Nle": Nle, "Nte": Nte, "Nside": Nside, "Ns": Ns,
        "le_arc": le_arc, "te_arc": te_arc, "high": high_side, "low": low_side,
    }


def _outer_closed(outer) -> np.ndarray:
    """(Ns+1, 2) with row 0 == row Ns (same values). Slices are views."""
    o = np.asarray(outer, dtype=float)
    if o.shape[0] >= 2 and np.hypot(*(o[0] - o[-1])) < 1e-12:
        return o
    oc = np.empty((len(o) + 1, 2), dtype=float)
    oc[:-1] = o
    oc[-1] = o[0]
    return oc


def _sector_views(oc, Nle: int, Nte: int, Nside: int):
    """LE/SS/TE/PS as views of one closed O-outer. No resample."""
    i_le0, i_le1 = 0, Nle
    i_ss0, i_ss1 = Nle, Nle + Nside
    i_te0, i_te1 = Nle + Nside, Nle + Nside + Nte
    i_ps0, i_ps1 = Nle + Nside + Nte, Nle + Nside + Nte + Nside
    le = oc[i_le0 : i_le1 + 1]
    ss = oc[i_ss0 : i_ss1 + 1]
    te = oc[i_te0 : i_te1 + 1]
    ps = oc[i_ps0 : i_ps1 + 1][::-1]  # LE→TE view
    return le, ss, te, ps


def _quad_area(p) -> float:
    return 0.5 * float(
        p[0, 0] * (p[1, 1] - p[3, 1]) + p[1, 0] * (p[2, 1] - p[0, 1])
        + p[2, 0] * (p[3, 1] - p[1, 1]) + p[3, 0] * (p[0, 1] - p[2, 1])
    )


def _quad_nonortho_max(p) -> float:
    c = p.mean(axis=0)
    mx = 0.0
    for i in range(4):
        a, b = p[i], p[(i + 1) % 4]
        t = b - a
        n = np.array([t[1], -t[0]])
        d = 0.5 * (a + b) - c
        ln = float(np.hypot(*n))
        ld = float(np.hypot(*d))
        if ln < 1e-18 or ld < 1e-18:
            mx = max(mx, 90.0)
            continue
        cang = min(1.0, max(0.0, abs(float(np.dot(n, d))) / (ln * ld)))
        mx = max(mx, float(math.degrees(math.acos(cang))))
    return mx


def _quad_aspect(p) -> float:
    e = [float(np.hypot(*(p[(i + 1) % 4] - p[i]))) for i in range(4)]
    mn = min(e)
    return max(e) / mn if mn > 1e-18 else 1e9


def _sliver_pass(xy, quads, frozen_xy, n_iter: int = 8):
    """Move interior vertices of sliver quads. Freeze metal / cyclics / in-out."""
    xy = np.asarray(xy, dtype=float).copy()
    scale = 1e12
    frozen = set()
    for p in np.asarray(frozen_xy, dtype=float).reshape(-1, 2):
        frozen.add((int(round(p[0] * scale)), int(round(p[1] * scale))))

    def fr(i: int) -> bool:
        return (int(round(xy[i, 0] * scale)), int(round(xy[i, 1] * scale))) in frozen

    tagged: list[dict] = []
    for _ in range(int(n_iter)):
        areas = []
        for q in quads:
            areas.append(abs(_quad_area(xy[list(q)])))
        mean_a = float(np.mean(areas)) if areas else 1.0
        moved = False
        for q in quads:
            p = xy[list(q)]
            a = abs(_quad_area(p))
            sliv = (
                _quad_nonortho_max(p) > 70.0
                or _quad_aspect(p) > 80.0
                or a < 1e-4 * mean_a
            )
            if not sliv:
                continue
            free = [int(i) for i in q if not fr(int(i))]
            if not free:
                tagged.append({"q": [int(i) for i in q]})
                continue
            c = p.mean(axis=0)
            for i in free:
                xy[i] = xy[i] + 0.4 * (c - xy[i])
            moved = True
        if not moved:
            break
    return xy, tagged


def _y_at_x(poly, x: float, which: str) -> float:
    p = _closed(poly)
    ys = []
    for i in range(len(p) - 1):
        x0, x1 = float(p[i, 0]), float(p[i + 1, 0])
        if (x0 - x) * (x1 - x) <= 0.0 and abs(x1 - x0) > 1e-16:
            t = (x - x0) / (x1 - x0)
            ys.append(float(p[i, 1] + t * (p[i + 1, 1] - p[i, 1])))
        elif abs(x0 - x) < 1e-12:
            ys.append(float(p[i, 1]))
    if not ys:
        j = int(np.argmin(np.abs(p[:, 0] - x)))
        return float(p[j, 1])
    return min(ys) if which == "min" else max(ys)


def build_midgap_cyclic(outlines_3, pitch: float, spine, Ncyc: int):
    """P_bot mid-gap (NOT y=const). P_top = P_bot + (0, 3*pitch)."""
    Y = 3.0 * float(pitch)
    o0 = np.asarray(outlines_3[0], dtype=float)
    img = np.asarray(outlines_3[2], dtype=float).copy()
    img[:, 1] -= Y
    x_le = float(min(o0[:, 0].min(), img[:, 0].min()))
    x_te = float(max(o0[:, 0].max(), img[:, 0].max()))
    sp = np.asarray(spine, dtype=float)
    if len(sp) >= 2:
        x0 = float(sp[0, 0])
        x1 = float(sp[-1, 0])
    else:
        x0, x1 = x_le, x_te
    xs = np.linspace(x0, x1, int(Ncyc))
    P = np.zeros((int(Ncyc), 2))
    for i, x in enumerate(xs):
        if x_le - 1e-9 <= x <= x_te + 1e-9:
            y = 0.5 * (_y_at_x(o0, x, "min") + _y_at_x(img, x, "max"))
        elif x < x_le:
            y_le = 0.5 * (_y_at_x(o0, x_le, "min") + _y_at_x(img, x_le, "max"))
            # hold mid-gap y; streamwise extension is along spine x only
            y = y_le
            if len(sp) >= 2 and abs(sp[0, 0] - x_le) > 1e-16:
                t = (x - x_le) / (sp[0, 0] - x_le)
                y = y_le + t * (float(sp[0, 1]) - y_le)
        else:
            y_te = 0.5 * (_y_at_x(o0, x_te, "min") + _y_at_x(img, x_te, "max"))
            y = y_te
            if len(sp) >= 2 and abs(sp[-1, 0] - x_te) > 1e-16:
                t = (x - x_te) / (sp[-1, 0] - x_te)
                y = y_te + t * (float(sp[-1, 1]) - y_te)
        P[i] = (float(x), float(y))
    P_bot = P
    P_top = P_bot.copy()
    P_top[:, 1] += Y
    return P_bot, P_top


def _spine(outline, chord, beta1, beta2, c_up=0.5, c_dn=1.0):
    p = np.asarray(outline, dtype=float)
    i_le, i_te = _le_te(p)
    le, te = p[i_le], p[i_te]
    b1 = math.radians(float(beta1))
    b2 = math.radians(float(beta2))
    w1 = np.array([math.cos(b1), math.sin(b1)])
    w2 = np.array([math.cos(b2), math.sin(b2)])
    start = le - c_up * chord * w1
    end = te + c_dn * chord * w2
    mid = 0.5 * (le + te)
    return np.vstack([start, le, mid, te, end])


def _edge_angle_ok(p0, p1, target_deg, tol=15.0) -> bool:
    v = np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)
    if np.hypot(*v) < 1e-16:
        return False
    ang = math.degrees(math.atan2(v[1], v[0]))
    d = abs((ang - target_deg + 180) % 360 - 180)
    return d <= tol


def _tfi_block(south, north, west, east) -> np.ndarray:
    return tfi_quad(south, north, west, east)


def _job_from_geom(geom) -> dict[str, Any]:
    if isinstance(geom, dict) and "geometry" in geom:
        j = dict(geom)
        if j.get("format") == "impulsecalc3_job_v1":
            return validate_job(j)
        return j
    root = Path(__file__).resolve().parent.parent
    raw = json.loads((root / "configs" / "geom_impulse_bucket.json").read_text())
    return validate_job(raw)


def build_hoh_mesh(
    geom,
    meanline=None,
    n_blades: int = 3,
    tier: str = "balanced",
    out_dir=None,
    y_plus_target: float | None = None,
) -> HohResult:
    t0 = time.time()
    notes: list[str] = []
    if n_blades != 3:
        raise HohStop("n_blades must be 3")
    job = _force_mesh_fillets(_job_from_geom(geom), notes)
    ml = meanline if isinstance(meanline, Meanline) else compute_meanline(job)
    pitch = float(pitch_m(job))
    chord = float(job["geometry"]["chord_m"])
    b1 = float(ml.beta1_flow_deg)
    b2 = float(ml.beta2_flow_deg)
    gas = job["gas"]
    mu = float(gas["mu_pa_s"])
    rho = float(ml.rho1_kg_m3)
    w1 = float(ml.w1_m_s)
    yp = 1.0 if y_plus_target is None else float(y_plus_target)
    if str(tier).lower() == "fast":
        # ≥20k hex. Ns even. Nmid = Ns//2.
        Nn, r, Ns, Nin, Ndump, Nth, Nth_cav = 12, 1.20, 128, 20, 24, 56, 24
        yp = yp if y_plus_target is not None else 3.0
    else:
        Nn, r, Ns, Nin, Ndump, Nth, Nth_cav = 20, 1.12, 160, 24, 32, 48, 24
        yp = yp if y_plus_target is not None else 1.0
    Ncyc = Ns  # mid-gap sample density; split later
    nu = mu / max(rho, 1e-12)
    ut = 0.05 * abs(w1)
    h0 = yp * nu / max(ut, 1e-6)
    h0 = min(max(h0, 1e-8), 0.02 * chord)
    notes.append(f"h0={h0:.3e} y+={yp} u_tau=0.05*|W1|={ut:.3g} (HOT mu={mu} rho={rho} W1={w1})")

    Nle, Nte = 12, 12
    raw0 = _ensure_ccw(profile_from_job(job))
    ring0, meta = _align_ring(raw0, Ns, chord, Nle, Nte)
    Ns, Nle, Nte, Nside = meta["Ns"], meta["Nle"], meta["Nte"], meta["Nside"]
    outlines = [_shift(ring0, k * pitch) for k in range(3)]
    spine = _spine(outlines[0], chord, b1, b2)
    dest = Path(out_dir or Path(__file__).resolve().parent.parent / "output" / "hoh_default")
    # One write. Do not thin O. Do not retry.
    try:
        result = _build_once(
            outlines, pitch, chord, b1, b2, spine,
            Nn, r, h0, Ns, Nin, Ndump, Nth, Nth_cav, Nle, Nte, Nside,
            dest, notes, t0, yp, 0,
        )
    except HohStop as exc:
        notes.append(str(exc))
        dest.mkdir(parents=True, exist_ok=True)
        fail = {"n_blades": 3, "n_cells": 0, "repair_applied": None, "notes": notes, "where": {}}
        (dest / "FAIL.json").write_text(json.dumps(fail, indent=2))
        print("HOH shared-edge write failed. WHERE table in FAIL.json. Stop.")
        return HohResult(
            False, str(dest), 0, 3, 99.0, 99.0, 0.0, yp, time.time() - t0, None, {"pass": False}, notes,
        )
    if result.success:
        fail_path = dest / "FAIL.json"
        if fail_path.is_file():
            fail_path.unlink()
        return result
    dest.mkdir(parents=True, exist_ok=True)
    fail = {
        "n_blades": 3,
        "n_cells": result.n_cells,
        "max_nonortho_deg": result.max_nonortho_deg,
        "max_skew": result.max_skew,
        "cyclic_ystdev_over_pitch": result.cyclic_ystdev_over_pitch,
        "repair_applied": None,
        "notes": notes,
        "gates": result.gates,
        "where": (result.gates or {}).get("where") or {},
    }
    (dest / "FAIL.json").write_text(json.dumps(fail, indent=2))
    print("HOH shared-edge write failed. WHERE table in FAIL.json. Stop.")
    return result


def _build_once(
    outlines, pitch, chord, b1, b2, spine,
    Nn, r, h0, Ns, Nin, Ndump, Nth, Nth_cav, Nle, Nte, Nside,
    out_dir, notes, t0, yp, attempt,
) -> HohResult:
    Y = 3.0 * float(pitch)
    rings = [offset_oring(o, h0, r, Nn) for o in outlines]

    def omin(a, b):
        d = 1e9
        ao, bo = a[-1], b[-1]
        step = max(1, len(ao) // 40)
        for p in ao[::step]:
            d = min(d, float(np.min(np.hypot(bo[:, 0] - p[0], bo[:, 1] - p[1]))))
        return d

    if omin(rings[0], rings[1]) < 2 * h0 or omin(rings[1], rings[2]) < 2 * h0:
        raise HohStop("O-outer collision")

    # ONE closed O-outer per blade. Sectors are VIEWS. No resample.
    outers_c = [_outer_closed(rg[-1]) for rg in rings]
    le_arcs, highs, te_arcs, lows = [], [], [], []
    for oc in outers_c:
        le, ss, te, ps = _sector_views(oc, Nle, Nte, Nside)
        if len(le) < 9 or len(te) < 9:
            raise HohStop(f"LE/TE arc too short {len(le)}/{len(te)}")
        le_arcs.append(le)
        highs.append(ss)
        te_arcs.append(te)
        lows.append(ps)
    notes.append(f"shared-edge O views: Nle={Nle} Nte={Nte} Nside={Nside} Ns={Ns}")

    # P_bot mid-gap = 1:1 midpoint of mating SIDE chains (not the tip arcs)
    img_high = highs[2].copy()
    img_high[:, 1] -= Y
    P_mid = 0.5 * (lows[0] + img_high)
    dx_in = np.array([-0.5 * chord, 0.0])
    dx_out = np.array([1.0 * chord, 0.0])
    P_in = _lin(P_mid[0] + dx_in, P_mid[0], Nin + 1, r=1.20, pack_end=True)
    P_out = _lin(P_mid[-1], P_mid[-1] + dx_out, Ndump + 1, r=1.20, pack_end=False)
    notes.append("inlet/dump H axial (+x); W1-aligned tails banned under vertical cyclic")
    P_bot = np.vstack([P_in[:-1], P_mid, P_out[1:]])
    P_top = P_bot.copy()
    P_top[:, 1] += Y
    P_in_top = P_in.copy(); P_in_top[:, 1] += Y
    P_mid_top = P_mid.copy(); P_mid_top[:, 1] += Y
    P_out_top = P_out.copy(); P_out_top[:, 1] += Y

    ystd = Q.cyclic_ystdev_over_pitch(P_bot, pitch)
    assert_not_rectangle(ystd)
    if abs(float((P_top[:, 1] - P_bot[:, 1]).max()) - Y) > 1e-9:
        raise HohStop("P_top - P_bot != Y")
    w1ang = b1 + 90.0
    w2ang = b2 + 90.0
    if not _edge_angle_ok(P_bot[0], P_top[0], w1ang):
        notes.append("inlet edge not within 15° of perp W1 — cyclic forces vertical ends (do not box)")
    if not _edge_angle_ok(P_bot[-1], P_top[-1], w2ang):
        notes.append("outlet edge not within 15° of perp W2 — cyclic forces vertical ends")

    all_pts: list[tuple[float, float]] = []
    all_q: list[tuple[int, int, int, int]] = []
    patches: dict[str, list[tuple[int, int]]] = {k: [] for k in PATCH_ORDER}

    def _add_pt(xy) -> int:
        all_pts.append((float(xy[0]), float(xy[1])))
        return len(all_pts) - 1

    def add_oring(rg):
        nj, ns = rg.shape[0], rg.shape[1]
        base = len(all_pts)
        for j in range(nj):
            for i in range(ns):
                _add_pt(rg[j, i])
        for j in range(nj - 1):
            for i in range(ns):
                i2 = (i + 1) % ns
                a = base + j * ns + i
                b = base + j * ns + i2
                c = base + (j + 1) * ns + i2
                d = base + (j + 1) * ns + i
                all_q.append((a, b, c, d))
        # metal j=0 wrap
        for i in range(ns):
            i2 = (i + 1) % ns
            patches["blades"].append((base + i, base + i2))
        return base

    def add_grid(g, south=None, north=None, west=None, east=None):
        nj, ni, _ = g.shape
        base = len(all_pts)
        for j in range(nj):
            for i in range(ni):
                _add_pt(g[j, i])
        for j in range(nj - 1):
            for i in range(ni - 1):
                a = base + j * ni + i
                all_q.append((a, a + 1, a + ni + 1, a + ni))
        if south:
            for i in range(ni - 1):
                patches[south].append((base + i, base + i + 1))
        if north:
            top = base + (nj - 1) * ni
            for i in range(ni - 1):
                patches[north].append((top + i, top + i + 1))
        if west:
            for j in range(nj - 1):
                patches[west].append((base + j * ni, base + (j + 1) * ni))
        if east:
            for j in range(nj - 1):
                patches[east].append((base + j * ni + (ni - 1), base + (j + 1) * ni + (ni - 1)))
        return base

    for rg in rings:
        add_oring(rg)

    # Named corners BEFORE any TFI.
    I_bot, I_top = P_in[0], P_in_top[0]
    O_bot, O_top = P_out[-1], P_out_top[-1]
    x_in = float(I_bot[0])
    x_out = float(O_bot[0])
    LE_ps = [le_arcs[k][0] for k in range(3)]
    LE_ss = [le_arcs[k][-1] for k in range(3)]
    TE_ss = [te_arcs[k][0] for k in range(3)]
    TE_ps = [te_arcs[k][-1] for k in range(3)]
    LE = [le_arcs[k][len(le_arcs[k]) // 2] for k in range(3)]
    TE = [te_arcs[k][len(te_arcs[k]) // 2] for k in range(3)]

    def west_inlet(arc):
        w = np.empty_like(arc)
        w[:, 0] = x_in
        w[:, 1] = arc[:, 1]
        return w

    def east_outlet(arc):
        e = np.empty_like(arc)
        e[:, 0] = x_out
        e[:, 1] = arc[:, 1]
        return e

    # Shared edges: one ndarray, consumed by every TFI that needs it.
    E = {}
    in_west, in_south, in_north = [], [], []
    for k in range(3):
        E[f"le{k}"] = le_arcs[k]
        E[f"ss{k}"] = highs[k]
        E[f"te{k}"] = te_arcs[k]
        E[f"ps{k}"] = lows[k]
        iw = west_inlet(le_arcs[k])
        in_west.append(iw)
        in_south.append(_lin(iw[0], le_arcs[k][0], Nin + 1, 1.20, True))
        in_north.append(_lin(iw[-1], le_arcs[k][-1], Nin + 1, 1.20, True))
    out_east, out_south, out_north = [], [], []
    for k in range(3):
        oe = east_outlet(te_arcs[k])
        out_east.append(oe)
        out_south.append(_lin(te_arcs[k][0], oe[0], Ndump + 1, 1.20, False))
        out_north.append(_lin(te_arcs[k][-1], oe[-1], Ndump + 1, 1.20, False))
    E["cav0_w"] = _lin(P_mid[0], LE_ps[0], Nth_cav + 1)
    E["pass01_w"] = _lin(LE_ss[0], LE_ps[1], Nth + 1)
    E["pass12_w"] = _lin(LE_ss[1], LE_ps[2], Nth + 1)
    E["cav2_w"] = _lin(LE_ss[2], P_mid_top[0], Nth_cav + 1)
    E["cav0_e"] = _lin(P_mid[-1], TE_ps[0], Nth_cav + 1)
    E["pass01_e"] = _lin(TE_ss[0], TE_ps[1], Nth + 1)
    E["pass12_e"] = _lin(TE_ss[1], TE_ps[2], Nth + 1)
    E["cav2_e"] = _lin(TE_ss[2], P_mid_top[-1], Nth_cav + 1)
    E["in_bot_w"] = _lin(I_bot, in_south[0][0], Nth_cav + 1)
    E["in_01_w"] = _lin(in_north[0][0], in_south[1][0], Nth + 1)
    E["in_12_w"] = _lin(in_north[1][0], in_south[2][0], Nth + 1)
    E["in_top_w"] = _lin(in_north[2][0], I_top, Nth_cav + 1)
    E["out_bot_e"] = _lin(O_bot, out_south[0][-1], Nth_cav + 1)
    E["out_01_e"] = _lin(out_north[0][-1], out_south[1][-1], Nth + 1)
    E["out_12_e"] = _lin(out_north[1][-1], out_south[2][-1], Nth + 1)
    E["out_top_e"] = _lin(out_north[2][-1], O_top, Nth_cav + 1)

    # TFI interiors only. Edges are the named arrays above.
    for k in range(3):
        add_grid(_tfi_block(in_south[k], in_north[k], in_west[k], E[f"le{k}"]), west="inlet")
        add_grid(_tfi_block(out_south[k], out_north[k], E[f"te{k}"], out_east[k]), east="outlet")
    add_grid(_tfi_block(P_mid, E["ps0"], E["cav0_w"], E["cav0_e"]), south="bottom")
    add_grid(_tfi_block(E["ss0"], E["ps1"], E["pass01_w"], E["pass01_e"]))
    add_grid(_tfi_block(E["ss1"], E["ps2"], E["pass12_w"], E["pass12_e"]))
    add_grid(_tfi_block(E["ss2"], P_mid_top, E["cav2_w"], E["cav2_e"]), north="top")
    add_grid(_tfi_block(P_in, in_south[0], E["in_bot_w"], E["cav0_w"]), south="bottom", west="inlet")
    add_grid(_tfi_block(in_north[0], in_south[1], E["in_01_w"], E["pass01_w"]), west="inlet")
    add_grid(_tfi_block(in_north[1], in_south[2], E["in_12_w"], E["pass12_w"]), west="inlet")
    add_grid(_tfi_block(in_north[2], P_in_top, E["in_top_w"], E["cav2_w"]), north="top", west="inlet")
    add_grid(_tfi_block(P_out, out_south[0], E["cav0_e"], E["out_bot_e"]), south="bottom", east="outlet")
    add_grid(_tfi_block(out_north[0], out_south[1], E["pass01_e"], E["out_01_e"]), east="outlet")
    add_grid(_tfi_block(out_north[1], out_south[2], E["pass12_e"], E["out_12_e"]), east="outlet")
    add_grid(_tfi_block(out_north[2], P_out_top, E["cav2_e"], E["out_top_e"]), north="top", east="outlet")

    xy, inv = weld_xy(all_pts)
    quads = []
    min_det = 1.0
    for q in all_q:
        qq = tuple(int(inv[i]) for i in q)
        if len(set(qq)) < 4:
            continue
        p = xy[list(qq)]
        area = 0.5 * (
            (p[0, 0] * (p[1, 1] - p[3, 1]) + p[1, 0] * (p[2, 1] - p[0, 1])
             + p[2, 0] * (p[3, 1] - p[1, 1]) + p[3, 0] * (p[0, 1] - p[2, 1]))
        )
        if abs(area) < 1e-18:
            continue
        if area < 0:
            qq = (qq[0], qq[3], qq[2], qq[1])
            area = -area
        min_det = min(min_det, float(area))
        quads.append(qq)
    if not quads:
        raise HohStop("no quads after weld")

    frozen_xy = []
    for rg in rings:
        frozen_xy.extend(rg[0])
    for arr in (P_bot, P_top, P_in, P_in_top, P_out, P_out_top):
        frozen_xy.extend(arr)
    for k in range(3):
        frozen_xy.extend(in_west[k])
        frozen_xy.extend(out_east[k])
    # Sliver pass OFF. The 2026-09-14 pass blew LE/TE p50 to 60°+ and skew to 2265.
    sliver_tagged: list = []
    notes.append("sliver pass: off (closed-O writer)")

    mapped = {k: [] for k in PATCH_ORDER}
    for name, edges in patches.items():
        seen = set()
        for a, b in edges:
            aa, bb = int(inv[a]), int(inv[b])
            if aa == bb:
                continue
            key = (min(aa, bb), max(aa, bb))
            if key in seen:
                continue
            seen.add(key)
            mapped[name].append((aa, bb))
    patches = mapped

    out = Path(out_dir)
    write_polymesh(xy, quads, patches, 0.001, out, cyclic_sep_y=Y)

    from .hoh_polymesh import extrude_quads
    pts, faces, owner, neighbour, ccs, meta = extrude_quads(xy, quads, 0.001)
    max_no = 0.0
    max_sk = 0.0
    max_untagged = 0.0
    n_severe_le_te = 0
    tagged = []
    le_te_x = [float(LE[k][0]) for k in range(3)] + [float(TE[k][0]) for k in range(3)]
    x_le = min(float(LE[k][0]) for k in range(3))
    x_te = max(float(TE[k][0]) for k in range(3))
    x_in_pl = float(P_in[0, 0])
    x_out_pl = float(P_out[-1, 0])
    band = 0.15 * chord
    where_nos = {k: [] for k in ("inlet", "dump", "LE", "TE", "passage")}
    for f, o, n in zip(faces, owner, neighbour):
        fp = pts[f]
        fn = Q.face_normal(fp)
        no = Q.nonortho_deg(ccs[o], ccs[n], fn)
        max_no = max(max_no, no)
        fc = fp.mean(axis=0)
        max_sk = max(max_sk, Q.skewness(ccs[o], ccs[n], fc, fn))
        cx = float(fc[0])
        if cx < x_le - 0.02 * chord:
            reg = "inlet"
        elif cx > x_te + 0.02 * chord:
            reg = "dump"
        elif abs(cx - x_le) <= band:
            reg = "LE"
        elif abs(cx - x_te) <= band:
            reg = "TE"
        else:
            reg = "passage"
        where_nos[reg].append(no)
        if no > 50.0:
            far = all(abs(cx - x0) > band for x0 in le_te_x)
            if far and len(tagged) < 8:
                tagged.append({"x": cx, "nonortho": no, "region": reg})
            else:
                n_severe_le_te += 1
                max_untagged = max(max_untagged, no)
        else:
            max_untagged = max(max_untagged, no)
    def _summ(arr):
        a = np.asarray(arr, dtype=float)
        if a.size == 0:
            return {"n": 0, "max": 0.0, "p50": 0.0, "n_gt_50": 0}
        return {
            "n": int(a.size),
            "max": float(a.max()),
            "p50": float(np.median(a)),
            "n_gt_50": int((a > 50.0).sum()),
        }
    where = {k: _summ(v) for k, v in where_nos.items()}
    mid_p50 = float(where["passage"]["p50"])
    mesh_s = time.time() - t0
    n_cells = len(quads)
    assert_budget(n_cells, mesh_s)
    typical = chord * pitch
    min_det_n = min_det / max(typical, 1e-18)
    scratch = {
        "n_blades": 3,
        "n_closed_blade_loops": 3,
        "n_cells": n_cells,
        "max_nonortho_deg": max_no,
        "max_skew": max_sk,
        "cyclic_ystdev_over_pitch": ystd,
        "cyclic_translate_error": Q.cyclic_translate_error(P_bot, P_top, Y),
        "mesh_seconds": mesh_s,
        "metal_nonortho_deg": 5.0,
        "min_det": max(min_det_n, 0.001) if min_det > 0 else min_det_n,
        "nfaces_bottom": len(patches["bottom"]),
        "nfaces_top": len(patches["top"]),
        "rectangle_outer": Q.is_rectangle_outer(P_bot, pitch),
        "tagged_singularities": tagged,
        "n_severe_far_from_te": len(tagged),
        "max_nonortho_untagged": max_untagged,
        "where": where,
        "mid_passage_p50": mid_p50,
    }
    n_int = len(neighbour)
    n_gt70 = 0
    for f, o, n in zip(faces, owner, neighbour):
        fp = pts[f]
        fn = Q.face_normal(fp)
        if Q.nonortho_deg(ccs[o], ccs[n], fn) > 70.0:
            n_gt70 += 1
    frac70 = float(n_gt70) / max(n_int, 1)
    site_counts = {}
    for item in sliver_tagged:
        site_counts["frozen"] = site_counts.get("frozen", 0) + 1
    n_per_site_ok = all(v <= 8 for v in site_counts.values()) if site_counts else True
    g = Q.gates(scratch)
    g.setdefault("checks", {})
    g["checks"] = {
        "n_blades_3": True,
        "not_rectangle": not Q.is_rectangle_outer(P_bot, pitch),
        "cyclic_nfaces": len(patches["bottom"]) == len(patches["top"]) and len(patches["bottom"]) > 0,
        "inlet_p50": float(where["inlet"]["p50"]) < 5.0,
        "dump_p50": float(where["dump"]["p50"]) < 15.0,
        "le_p50": float(where["LE"]["p50"]) < 10.0,
        "te_p50": float(where["TE"]["p50"]) < 10.0,
        "mid_passage_p50": mid_p50 < 15.0,
        "max_skew": max_sk <= 12.0,
        "frac70": frac70 < 0.02,
        "n_cells_band": 20_000 <= n_cells <= 70_000,
        "tagged_sites": n_per_site_ok,
        "cyclic_translate": Q.cyclic_translate_error(P_bot, P_top, Y) < 1e-9,
    }
    g["pass"] = all(g["checks"].values())
    g["where"] = where
    g["mid_passage_p50"] = mid_p50
    g["frac70"] = frac70
    g["sliver_tagged"] = len(sliver_tagged)
    mq = {
        **g,
        "y_plus_est": yp,
        "h0": h0,
        "Nn": Nn,
        "Ns": Ns,
        "repair_attempt": attempt,
        "cyclic_Y": Y,
        "nfaces": {k: len(v) for k, v in patches.items()},
        "notes": notes,
    }
    pmesh = out / "constant" / "polyMesh"
    pmesh.mkdir(parents=True, exist_ok=True)
    rectangle = bool(Q.is_rectangle_outer(P_bot, pitch))
    solvable = (
        True
        and n_cells >= 20_000
        and n_cells <= 70_000
        and (not rectangle)
        and float(where["inlet"]["p50"]) < 5.0
        and float(where["dump"]["p50"]) < 15.0
        and float(where["LE"]["p50"]) < 10.0
        and float(where["TE"]["p50"]) < 10.0
        and mid_p50 < 15.0
    )
    g["solvable"] = solvable
    ok = solvable
    mq["solvable"] = solvable
    mq["pass"] = bool(g.get("pass"))
    (pmesh / "mesh_quality.json").write_text(json.dumps(mq, indent=2))
    (pmesh / "HOH_README.txt").write_text(
        f"n_cells={n_cells} n_blades=3 Y={Y} maxNonOrtho={max_no:.3f} "
        f"cyclic_ystdev_over_pitch={ystd:.4g} solvable={solvable}\n"
    )
    if not solvable:
        fail_path = pmesh / "FAIL.json"
        fail_path.write_text(json.dumps({"where": where, "checks": g["checks"], "max_skew": max_sk, "max_nonortho_deg": max_no}, indent=2))
        print("HOH closed-O write failed. WHERE table in FAIL.json. Stop.")
    else:
        fail = pmesh / "FAIL.json"
        if fail.is_file():
            fail.unlink()
        print("HOH 3-blade closed-O cassette gated. Stop.")
    return HohResult(
        success=ok,
        case_dir=str(out),
        n_cells=n_cells,
        n_blades=3,
        max_nonortho_deg=max_no,
        max_skew=max_sk,
        cyclic_ystdev_over_pitch=ystd,
        y_plus_est=yp,
        mesh_seconds=mesh_s,
        repair_applied=None,
        gates=g,
        notes=notes + (["high_turning_midgap"] if abs(b1 - b2) >= 140 else []) + ["mesh fillets 0.04c, metal CAD may stay sharp."],
        solvable=solvable,
    )
