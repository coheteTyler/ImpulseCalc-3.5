"""Body-fitted 2D O-grid cascade mesh → OpenFOAM polyMesh.

O-grid: uniform-offset from the metal (wall-normal), n_around × n_radial.
H-blocks: TFI from the offset ring to the pitch rectangle (inlet / cyclic /
outlet). Three pitches are tiled and stitched. Stair-step is not the load path.
Optional Cartesian dump is debug-only and is never written as the forces mesh.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import (
    BladeSpec,
    center_in_pitch,
    passage_gap,
    polygon_centroid,
    polygon_signed_area,
    profile_from_job,
    resample_open_arclength,
    split_ps_ss,
)
from .job import pitch_m as cascade_pitch_m


def _foam_header(cls: str, obj: str, note: str = "") -> str:
    extra = f'\n    note        "{note}";' if note else ""
    return (
        "/*--------------------------------*- C++ -*----------------------------------*\\\n"
        "| ImpulseCalc3 body-fitted cascade mesh                                       |\n"
        "\\*---------------------------------------------------------------------------*/\n"
        "FoamFile\n"
        "{\n"
        "    version     2.0;\n"
        "    format      ascii;\n"
        f"    class       {cls};\n"
        f"    object      {obj};{extra}\n"
        "}\n"
        "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"
    )


def _stretch(j: int, n: int, r: float) -> float:
    if n <= 0:
        return 0.0
    if abs(r - 1.0) < 1e-9:
        return j / n
    return (r**j - 1.0) / (r**n - 1.0)


def _inlet_pack_r(n: int, r: float, max_cell_ratio: float = 20.0) -> float:
    """Geometric ratio for inlet H; soft-cap Δx_in/Δx_LE so west TFI stays positive."""
    r = max(float(r), 1.0)
    if n <= 1 or r <= 1.0 + 1e-12:
        return 1.0
    # With pack-toward-hi, inlet-plane Δx / LE Δx ≈ r**(n-1).
    if r ** (n - 1) <= max_cell_ratio:
        return r
    return float(max_cell_ratio ** (1.0 / (n - 1)))


def _xs_pack_hi(x_lo: float, x_hi: float, n: int, r: float) -> np.ndarray:
    """n+1 x-nodes from x_lo→x_hi with geometric pack toward x_hi (LE / blade).

    Mirrors dump ``xs_east`` which packs toward the low-x (blade) end via ``_stretch``.
    Inlet H runs x_in→LE join, so pack the high-x end.
    """
    r = _inlet_pack_r(n, r)
    return np.array(
        [x_hi - (x_hi - x_lo) * _stretch(n - j, n, r) for j in range(n + 1)],
        dtype=float,
    )



def _pack_r_for_d0(L: float, nseg: int, d0: float, *, r_max: float = 1.25) -> float:
    """Geometric pack ratio so first cell ≈ d0 over length L (nseg cells). Cap ≤ r_max."""
    d0 = max(float(d0), 1e-12)
    L = max(float(L), 1e-12)
    nseg = max(int(nseg), 1)
    if nseg <= 1:
        return 1.0
    target = d0 / L
    # Uniform first cell = L/nseg; if already ≤ d0*1.05, no pack needed.
    if L / nseg <= d0 * 1.05:
        return 1.0
    # d0/L = (r-1)/(r^n-1). Binary search r in [1.02, r_max].
    lo, hi = 1.02, max(float(r_max), 1.02)
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        val = (mid - 1.0) / (mid ** nseg - 1.0)
        if val > target:
            lo = mid
        else:
            hi = mid
    return float(min(0.5 * (lo + hi), float(r_max)))


def _nseg_for_d0(L: float, d0: float, *, r_max: float = 1.25, n_max: int = 80) -> int:
    """Min cell count so first cell ≈ d0 with growth ≤ r_max over length L."""
    import math as _math
    d0 = max(float(d0), 1e-12)
    L = max(float(L), 1e-12)
    r = max(float(r_max), 1.0 + 1e-9)
    if L <= d0 * 1.01:
        return 1
    if abs(r - 1.0) < 1e-12:
        n = int(_math.ceil(L / d0))
    else:
        n = int(_math.ceil(_math.log(max(1.0 + L * (r - 1.0) / d0, 1.0001)) / _math.log(r)))
    return int(max(2, min(n, int(n_max))))


def dump_xs_1c(
    x_te: float,
    chord: float,
    dx_near: float,
    *,
    n_near: int = 10,
    stretch_max: float = 1.25,
    L_dump_c: float = 1.0,
    L_near_c: float = 0.4,
) -> np.ndarray:
    """Dump x-nodes: TE collar → ≥1.0c. First Δx ≈ dx_near, growth ≤ stretch_max (~1.25).

    Soft TE collar↔dump join: no chalk-line size cliff (neighbor ratio gate ~≤4).
    ``n_near`` / ``L_near_c`` kept for API compat; sizing is geometric from dx_near.
    """
    import math as _math
    _ = n_near, L_near_c  # API compat; geometric law owns the near band now.
    c = max(float(chord), 1e-9)
    x_te = float(x_te)
    L = max(float(L_dump_c), 1.0) * c
    x_out = x_te + L
    dx0 = max(float(dx_near), 1e-9)
    r = min(max(float(stretch_max), 1.0), 1.25)
    if abs(r - 1.0) < 1e-12:
        n = max(2, int(_math.ceil(L / dx0)))
    else:
        n = max(2, int(_math.ceil(_math.log(max(1.0 + L * (r - 1.0) / dx0, 1.0001)) / _math.log(r))))
    n = min(n, 160)
    # Place exactly on [x_te, x_out] so first Δx ≈ dx0 with growth r.
    xs = np.array([x_te + L * _stretch(j, n, r) for j in range(n + 1)], dtype=float)
    xs[0] = x_te
    xs[-1] = x_out
    return xs



def _join_polylines(*arrs: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Concatenate open polylines (N,2), dropping shared endpoints."""
    chunks: list[np.ndarray] = []
    for a in arrs:
        a = np.asarray(a, dtype=float)
        if a.size == 0:
            continue
        if a.ndim != 2 or a.shape[1] != 2:
            a = a.reshape(-1, 2)
        if not chunks:
            chunks.append(a)
            continue
        if float(np.linalg.norm(a[0] - chunks[-1][-1])) <= eps:
            chunks.append(a[1:])
        else:
            chunks.append(a)
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 2), dtype=float)


def _dump_block_from_west(
    west: np.ndarray,
    xs_dump: np.ndarray,
    x_out: float | None = None,
) -> np.ndarray:
    """Dump H: curved west edge → x_out with dump_xs fractional packing (no TE cliff).

    pts shape (n_x+1, n_y, 2). Each ray keeps constant y; axial nodes follow
    dump_xs fractions mapped onto [x_west, x_out].
    """
    west = np.asarray(west, dtype=float).reshape(-1, 2)
    xs_dump = np.asarray(xs_dump, dtype=float).reshape(-1)
    if west.shape[0] < 2 or xs_dump.shape[0] < 2:
        raise RuntimeError("dump west / xs_dump too short")
    x_te = float(xs_dump[0])
    x_hi = float(xs_dump[-1] if x_out is None else x_out)
    span0 = max(x_hi - x_te, 1e-15)
    frac = (xs_dump - x_te) / span0
    ni = int(xs_dump.shape[0] - 1)
    nj = int(west.shape[0])
    pts = np.zeros((ni + 1, nj, 2), dtype=float)
    for j in range(nj):
        x0 = float(west[j, 0])
        y0 = float(west[j, 1])
        xs_j = x0 + frac * (x_hi - x0)
        xs_j[0] = x0
        xs_j[-1] = x_hi
        pts[:, j, 0] = xs_j
        pts[:, j, 1] = y0
    return pts


def te_plane_fence_hit(
    h_blocks: list[np.ndarray],
    *,
    x_te: float,
    chord: float,
    y_bot: float,
    y_top: float,
    tol_c: float = 0.02,
    span_frac: float = 0.90,
) -> bool:
    """True if a near-vertical full-pitch column sits within tol_c*c of x_TE.

    Detects the old O→Cartesian dump join (AABB east wall) that can fake Δp.
    """
    c = max(float(chord), 1e-9)
    tol = float(tol_c) * c
    pitch = float(y_top) - float(y_bot)
    if pitch <= 1e-12:
        return False
    # Collect nearly-vertical interior edges near x_TE (axis-aligned fence).
    y_hits: list[tuple[float, float]] = []
    for hb in h_blocks:
        p = np.asarray(hb, dtype=float)
        if p.ndim != 3 or p.shape[0] < 2 or p.shape[1] < 2:
            continue
        # i-const columns: edges between j and j+1
        for i in range(p.shape[0]):
            col = p[i]
            xs = col[:, 0]
            ys = col[:, 1]
            if float(np.max(xs) - np.min(xs)) > tol:
                continue
            xmid = float(np.mean(xs))
            if abs(xmid - float(x_te)) > tol:
                continue
            y_hits.append((float(np.min(ys)), float(np.max(ys))))
        # Also check single i-faces that are vertical between neighboring i at fixed j
        # (cartesian dump west): look at first i-column span
    if not y_hits:
        return False
    # Merge y intervals at this TE plane
    y_hits.sort()
    merged: list[list[float]] = []
    for lo, hi in y_hits:
        if not merged or lo > merged[-1][1] + 1e-9:
            merged.append([lo, hi])
        else:
            merged[-1][1] = max(merged[-1][1], hi)
    span = sum(hi - lo for lo, hi in merged)
    return span >= float(span_frac) * pitch


def _chain_arclength(chain: list[tuple[float, float]]) -> tuple[list[float], float]:
    s = [0.0]
    for i in range(1, len(chain)):
        s.append(s[-1] + math.hypot(chain[i][0] - chain[i - 1][0], chain[i][1] - chain[i - 1][1]))
    total = s[-1] if s[-1] > 0 else 1.0
    return s, total


def _interp_at_arclength(
    chain: list[tuple[float, float]], s: list[float], total: float, target: float
) -> tuple[float, float]:
    if target <= 0:
        return chain[0]
    if target >= total:
        return chain[-1]
    j = 0
    while j < len(s) - 1 and s[j + 1] < target:
        j += 1
    span = s[j + 1] - s[j] or 1e-16
    t = (target - s[j]) / span
    x = chain[j][0] + t * (chain[j + 1][0] - chain[j][0])
    y = chain[j][1] + t * (chain[j + 1][1] - chain[j][1])
    return (x, y)


def _geometric_fractions(n_seg: int, ratio: float, *, dense_at: str = "start") -> list[float]:
    """n_seg+1 fractions in [0, 1]. ratio>=1 densifies toward dense_at (start|end|both)."""
    if n_seg < 1:
        return [0.0]
    r = max(float(ratio), 1.0)
    if abs(r - 1.0) < 1e-12:
        return [k / n_seg for k in range(n_seg + 1)]
    if dense_at == "both":
        # Half geometric from each end; meet at mid index.
        n0 = n_seg // 2
        n1 = n_seg - n0
        if n0 < 1:
            return _geometric_fractions(n_seg, r, dense_at="start")
        left = _geometric_fractions(n0, r, dense_at="start")
        right = _geometric_fractions(n1, r, dense_at="end")
        # map left to [0,0.5], right to [0.5,1]
        out = [0.5 * f for f in left[:-1]]
        out.extend(0.5 + 0.5 * f for f in right)
        return out
    # Growing segment lengths → small cells at start when r>1.
    lengths = [r ** i for i in range(n_seg)]
    tot = sum(lengths) or 1.0
    fracs = [0.0]
    acc = 0.0
    for L in lengths:
        acc += L
        fracs.append(acc / tot)
    if dense_at == "end":
        fracs = [1.0 - f for f in reversed(fracs)]
    return fracs


def _le_x_weight(x: float, xmin: float, xmax: float, le_cluster: float) -> float:
    """Weight >=1 peaking at LE (min x). le_cluster=1 → uniform."""
    r = max(float(le_cluster), 1.0)
    if abs(r - 1.0) < 1e-12:
        return 1.0
    span = max(xmax - xmin, 1e-16)
    t = max(0.0, min(1.0, (xmax - float(x)) / span))  # 1 at LE, 0 at TE
    return 1.0 + (r - 1.0) * (t * t)


def _resample_at_fractions(
    chain: list[tuple[float, float]], fracs: list[float]
) -> list[tuple[float, float]]:
    if len(chain) < 2:
        return list(chain)
    s, total = _chain_arclength(chain)
    return [_interp_at_arclength(chain, s, total, total * float(f)) for f in fracs]


def _resample(chain: list[tuple[float, float]], n_seg: int) -> list[tuple[float, float]]:
    """n_seg segments → n_seg+1 points including both ends, even arc-length."""
    if n_seg < 1 or len(chain) < 2:
        return list(chain)
    return _resample_at_fractions(chain, [k / n_seg for k in range(n_seg + 1)])


def _resample_le_cluster(
    chain: list[tuple[float, float]],
    n_seg: int,
    le_cluster: float = 1.0,
    *,
    xmin: float | None = None,
    xmax: float | None = None,
) -> list[tuple[float, float]]:
    """Arc-length sample with streamwise density peaking at LE (min x).

    le_cluster=1 → uniform. le_cluster>1 biases Δs toward the low-x (LE) region
    via quadratic x-weights on each segment (both tips of a bucket get denser).
    """
    if n_seg < 1 or len(chain) < 2:
        return list(chain)
    r = max(float(le_cluster), 1.0)
    if abs(r - 1.0) < 1e-12:
        return _resample(chain, n_seg)
    xs = [p[0] for p in chain]
    x0 = float(xmin) if xmin is not None else float(min(xs))
    x1 = float(xmax) if xmax is not None else float(max(xs))
    s, total = _chain_arclength(chain)
    # Weighted segment lengths.
    wlen = [0.0]
    acc = 0.0
    for i in range(1, len(chain)):
        ds = s[i] - s[i - 1]
        xm = 0.5 * (chain[i][0] + chain[i - 1][0])
        acc += ds * _le_x_weight(xm, x0, x1, r)
        wlen.append(acc)
    wtot = wlen[-1] if wlen[-1] > 0 else 1.0
    out: list[tuple[float, float]] = []
    for k in range(n_seg + 1):
        target_w = wtot * k / n_seg
        # Map weighted progress back to geometric arc length.
        if target_w <= 0:
            out.append(chain[0])
            continue
        if target_w >= wtot:
            out.append(chain[-1])
            continue
        j = 0
        while j < len(wlen) - 1 and wlen[j + 1] < target_w:
            j += 1
        span_w = wlen[j + 1] - wlen[j] or 1e-16
        t = (target_w - wlen[j]) / span_w
        target_s = s[j] + t * (s[j + 1] - s[j])
        out.append(_interp_at_arclength(chain, s, total, target_s))
    return out


def _chain_ccw(pts: list[tuple[float, float]], i0: int, i1: int) -> list[tuple[float, float]]:
    n = len(pts)
    out = []
    i = i0
    guard = 0
    while True:
        out.append(pts[i])
        if i == i1:
            break
        i = (i + 1) % n
        guard += 1
        if guard > n + 2:
            break
    return out


def outer_rectangle(
    x_in: float,
    x_out: float,
    y_bot: float,
    y_top: float,
    n_bot: int,
    n_out: int,
    n_top: int,
    n_in: int,
) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    """CCW points around the pitch rectangle, starting at SW. n_* are cell counts."""
    xs_bot = np.linspace(x_in, x_out, n_bot + 1)
    ys_out = np.linspace(y_bot, y_top, n_out + 1)
    xs_top = np.linspace(x_out, x_in, n_top + 1)
    ys_in = np.linspace(y_top, y_bot, n_in + 1)
    pts: list[tuple[float, float]] = []
    for i in range(n_bot):
        pts.append((float(xs_bot[i]), y_bot))
    for i in range(n_out):
        pts.append((x_out, float(ys_out[i])))
    for i in range(n_top):
        pts.append((float(xs_top[i]), y_top))
    for i in range(n_in):
        pts.append((x_in, float(ys_in[i])))
    ranges = {
        "bottom": (0, n_bot),
        "outlet": (n_bot, n_bot + n_out),
        "top": (n_bot + n_out, n_bot + n_out + n_top),
        "inlet": (n_bot + n_out + n_top, n_bot + n_out + n_top + n_in),
    }
    return np.array(pts, dtype=float), ranges


def inner_from_profile(
    poly: list[tuple[float, float]],
    outer: np.ndarray,
    ranges: dict[str, tuple[int, int]],
) -> np.ndarray:
    """Match 4 blade sectors to the 4 rectangle sides (CCW, same index)."""
    pts = poly[:-1] if poly and poly[0] == poly[-1] else list(poly)
    splits: list[int] = []
    used: set[int] = set()
    corner_idx = [ranges[name][0] for name in ("bottom", "outlet", "top", "inlet")]
    for ci in corner_idx:
        ox, oy = outer[ci]
        cx, cy = polygon_centroid(pts)
        dx, dy = ox - cx, oy - cy
        best_i, best_dot = 0, -1e99
        for i, (x, y) in enumerate(pts):
            if i in used:
                continue
            dot = (x - cx) * dx + (y - cy) * dy
            if dot > best_dot:
                best_dot, best_i = dot, i
        splits.append(best_i)
        used.add(best_i)
    side_names = ("bottom", "outlet", "top", "inlet")
    # Shorter-arc / geometric-side TFI: pick the arc that actually sits on that AABB side.
    pts_arr = np.array(pts, dtype=float)
    side_axis = {"bottom": (1, False), "outlet": (0, True), "top": (1, True), "inlet": (0, False)}
    inner: list[tuple[float, float]] = []
    for k, name in enumerate(side_names):
        i0 = splits[k]
        i1 = splits[(k + 1) % 4]
        axis, want_max = side_axis[name]
        idx = _arc_choose(pts_arr, i0, i1, axis, want_max)
        chain = [pts[i] for i in idx]
        n_seg = ranges[name][1] - ranges[name][0]
        samp = _resample(chain, n_seg)
        inner.extend(samp[:-1])
    if len(inner) != len(outer):
        raise RuntimeError(f"inner/outer count mismatch {len(inner)} vs {len(outer)}")
    return np.array(inner, dtype=float)


def inner_match_by_angle(poly: list[tuple[float, float]], outer: np.ndarray) -> np.ndarray:
    """One inner point per outer AABB node, matched by polar angle from centroid.

    4-side AABB sector mapping put the whole east side on the TE (and west on
    the LE). Those fans are the checkMesh quality hole. Angle matching shares
    the rectangle among LE/SS/TE/PS.
    """
    pts = poly[:-1] if poly and poly[0] == poly[-1] else list(poly)
    if polygon_signed_area(pts) < 0:
        pts = list(reversed(pts))
    cx, cy = polygon_centroid(pts)
    n_out = int(outer.shape[0])
    n_in = len(pts)
    ang_in = np.arctan2(np.array([p[1] - cy for p in pts]), np.array([p[0] - cx for p in pts]))
    # unwrap along the closed loop so search is monotonic
    unw = ang_in.copy()
    for i in range(1, n_in):
        d = unw[i] - unw[i - 1]
        if d > np.pi:
            unw[i:] -= 2 * np.pi
        elif d < -np.pi:
            unw[i:] += 2 * np.pi
    inner = np.zeros((n_out, 2), dtype=float)
    # start at outer[0] nearest inner
    a0 = math.atan2(outer[0, 1] - cy, outer[0, 0] - cx)
    i0 = int(np.argmin(np.abs(((ang_in - a0 + np.pi) % (2 * np.pi)) - np.pi)))
    # walk outer; pick inner by closest unwrapped angle, locally
    j = i0
    for k in range(n_out):
        ao = math.atan2(outer[k, 1] - cy, outer[k, 0] - cx)
        best_j, best_d = j, 1e99
        for dj in range(-n_in // 4, n_in // 4 + 1):
            jj = (j + dj) % n_in
            d = abs(((ang_in[jj] - ao + np.pi) % (2 * np.pi)) - np.pi)
            if d < best_d:
                best_d, best_j = d, jj
        inner[k] = pts[best_j]
        j = best_j
    return inner


def inner_match_by_arclength(
    poly: list[tuple[float, float]],
    outer: np.ndarray,
    le_cluster: float = 1.0,
) -> np.ndarray:
    """Metal arc-length (optional LE cluster), start at the node nearest AABB SW.

    Polar angle matching on a cup puts a chord-long inner edge on the SW AABB
    wrap (checkMesh skew 4.68). Arc-length deletes that wrap jump.
    le_cluster>1 biases wall Δs toward the min-x (LE) region.
    """
    pts = poly[:-1] if poly and poly[0] == poly[-1] else list(poly)
    if polygon_signed_area(pts) < 0:
        pts = list(reversed(pts))
    n_out = int(outer.shape[0])
    ox, oy = float(outer[0, 0]), float(outer[0, 1])
    i0 = min(range(len(pts)), key=lambda i: (pts[i][0] - ox) ** 2 + (pts[i][1] - oy) ** 2)
    rot = pts[i0:] + pts[:i0]
    xs = [p[0] for p in rot]
    samp = _resample_le_cluster(
        rot + [rot[0]], n_out, le_cluster, xmin=min(xs), xmax=max(xs)
    )
    return np.array(samp[:-1], dtype=float)


def build_pitch_ogrid(
    inner: np.ndarray,
    outer: np.ndarray,
    n_radial: int,
    stretch: float,
) -> np.ndarray:
    """pts[i, j, 2] with j=0 wall, j=n_radial outer. i periodic."""
    n_i = inner.shape[0]
    pts = np.zeros((n_i, n_radial + 1, 2), dtype=float)
    for j in range(n_radial + 1):
        t = _stretch(j, n_radial, stretch)
        pts[:, j, :] = (1.0 - t) * inner + t * outer
    return pts


def build_hybrid_aabb_ogrid(
    inner: np.ndarray,
    outer: np.ndarray,
    n_radial: int,
    stretch: float,
    d_off: float,
) -> np.ndarray:
    """Near-wall layers follow a wall-normal offset; outer layers morph to the AABB.

    Distortion from the AABB corners is pushed off the metal (wall Cp). Outer ring
    stays AABB so Cartesian H-blocks still conformal-stitch (including cyclics).
    """
    n_i = inner.shape[0]
    pts = np.zeros((n_i, n_radial + 1, 2), dtype=float)
    pts[:, 0, :] = inner
    pts[:, -1, :] = outer
    off = offset_closed(inner, max(d_off, 1e-9), n_smooth=12)
    # Morph only the last 1–2 layers onto the AABB. The old 45% morph was the
    # LE/TE non-ortho / pyramid / skew hole (checkMesh failed_checks=3).
    j_off = max(n_radial - 2, 2)
    j_off = min(j_off, n_radial - 1)
    for j in range(1, n_radial):
        if j <= j_off:
            t = _stretch(j, j_off, stretch)
            pts[:, j, :] = (1.0 - t) * inner + t * off
        else:
            t = (j - j_off) / max(n_radial - j_off, 1)
            pts[:, j, :] = (1.0 - t) * off + t * outer
    return pts


def _relax_last_morph(pts: np.ndarray) -> np.ndarray:
    """If the AABB morph layer inverted a column, pull that column back toward the offset.

    Wall (j=0) and AABB outer stay put. Used after arc-length inner, which can
    leave a few last-layer folds at SE/SW while deleting the fat SW wrap edge.
    """
    n_i, n_j, _ = pts.shape
    out = pts.copy()
    if n_j < 4:
        return out
    j0 = n_j - 3
    jm = n_j - 2
    tcol = np.full(n_i, 0.5)
    for _ in range(24):
        if min_cell_area_2d(out) > 0:
            return out
        for i in range(n_i):
            i2 = (i + 1) % n_i
            a_m = _quad_area(out[i, j0], out[i2, j0], out[i2, jm], out[i, jm])
            a_o = _quad_area(out[i, jm], out[i2, jm], out[i2, -1], out[i, -1])
            if a_m > 0 and a_o > 0:
                continue
            tcol[i] *= 0.5
            tcol[i2] *= 0.5
            out[i, jm] = (1.0 - tcol[i]) * out[i, j0] + tcol[i] * out[i, -1]
            out[i2, jm] = (1.0 - tcol[i2]) * out[i2, j0] + tcol[i2] * out[i2, -1]
    return out



def _square_aabb_wraps(pts: np.ndarray, n_cyc: int = 0, n_out: int = 0) -> np.ndarray:
    """Last morph layer: wrap columns at AABB corners become a rectangle.

    Outer AABB nodes stay put so Cartesian H-blocks still stitch.
    """
    n_i, n_j, _ = pts.shape
    if n_j < 3:
        return pts
    out = pts.copy()
    ring = out[:, -1, :]
    xmin, xmax = float(ring[:, 0].min()), float(ring[:, 0].max())
    ymin, ymax = float(ring[:, 1].min()), float(ring[:, 1].max())
    targets = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    used: set[int] = set()
    corners: list[int] = []
    for tx, ty in targets:
        best_i, best_d = 0, 1e99
        for i in range(n_i):
            if i in used:
                continue
            d = (ring[i, 0] - tx) ** 2 + (ring[i, 1] - ty) ** 2
            if d < best_d:
                best_d, best_i = d, i
        corners.append(best_i)
        used.add(best_i)
    jm = n_j - 2
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    for ic in corners:
        ip = (ic - 1) % n_i
        C = ring[ic]
        W = ring[ip]
        sx = 1.0 if cx > C[0] else -1.0
        sy = 1.0 if cy > C[1] else -1.0
        side = max(float(np.hypot(W[0] - C[0], W[1] - C[1])), 1e-9)
        # Interior of wrap columns sit on the inward square; AABB outer unchanged.
        if abs(W[0] - C[0]) < abs(W[1] - C[1]):
            out[ip, jm, 0] = C[0] + sx * side
            out[ip, jm, 1] = W[1]
            out[ic, jm, 0] = C[0] + sx * side
            out[ic, jm, 1] = C[1]
        else:
            out[ip, jm, 0] = W[0]
            out[ip, jm, 1] = C[1] + sy * side
            out[ic, jm, 0] = C[0]
            out[ic, jm, 1] = C[1] + sy * side
    return out


def _quad_area(a, b, c, d) -> float:
    return 0.5 * (
        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        + (c[0] - a[0]) * (d[1] - a[1]) - (c[1] - a[1]) * (d[0] - a[0])
    )


def min_cell_area_2d(pts: np.ndarray) -> float:
    n_i, n_j, _ = pts.shape
    amin = 1e99
    for i in range(n_i):
        i2 = (i + 1) % n_i
        for j in range(n_j - 1):
            a = _quad_area(pts[i, j], pts[i2, j], pts[i2, j + 1], pts[i, j + 1])
            if a < amin:
                amin = a
    return float(amin)


def min_cell_area_2d_rect(pts: np.ndarray) -> float:
    ni, nj, _ = pts.shape
    amin = 1e99
    for i in range(ni - 1):
        for j in range(nj - 1):
            a = _quad_area(pts[i, j], pts[i + 1, j], pts[i + 1, j + 1], pts[i, j + 1])
            if a < amin:
                amin = a
    return float(amin)


def cartesian_block(x0: float, x1: float, y0: float, y1: float, nx: int, ny: int) -> np.ndarray:
    xs = np.linspace(x0, x1, nx + 1)
    ys = np.linspace(y0, y1, ny + 1)
    return cartesian_block_xy(xs, ys)


def cartesian_block_xy(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Axis-aligned H-block with prescribed edge spacings (still Cartesian)."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    pts = np.zeros((xs.shape[0], ys.shape[0], 2), dtype=float)
    pts[:, :, 0] = xs[:, None]
    pts[:, :, 1] = ys[None, :]
    return pts


def _aabb_sides(outer: np.ndarray, n_cyc: int, n_out: int, n_in: int):
    """Inclusive SW→SE, SE→NE, NE→NW, NW→SW (corners duplicated on adjacent sides)."""
    i_se = n_cyc
    i_ne = n_cyc + n_out
    i_nw = n_cyc + n_out + n_cyc
    south = outer[0 : i_se + 1].copy()
    east = outer[i_se : i_ne + 1].copy()
    north = outer[i_ne : i_nw + 1].copy()
    west = np.concatenate([outer[i_nw:], outer[0:1]], axis=0).copy()
    if west.shape[0] != n_in + 1:
        raise RuntimeError(f"west side {west.shape[0]} != n_in+1={n_in+1}")
    return south, east, north, west


def _ring_from_aabb_sides(south, east, north, west) -> np.ndarray:
    return np.concatenate([south[:-1], east[:-1], north[:-1], west[:-1]], axis=0)


def _metal_ccw_mid(
    poly: list[tuple[float, float]], p0: np.ndarray, p1: np.ndarray
) -> np.ndarray:
    """Arc-length midpoint on the CCW metal walk from p0 toward p1."""
    pts = poly[:-1] if poly and poly[0] == poly[-1] else list(poly)
    if polygon_signed_area(pts) < 0:
        pts = list(reversed(pts))
    def nearest(p):
        return min(range(len(pts)), key=lambda i: (pts[i][0] - p[0]) ** 2 + (pts[i][1] - p[1]) ** 2)
    i0, i1 = nearest(p0), nearest(p1)
    chain = _chain_ccw(pts, i0, i1)
    if len(chain) < 2:
        return 0.5 * (np.asarray(p0, dtype=float) + np.asarray(p1, dtype=float))
    samp = _resample([(float(p[0]), float(p[1])) for p in chain], 2)
    return np.array(samp[1], dtype=float)


def split_aabb_wrap_edges(
    inner: np.ndarray,
    outer: np.ndarray,
    poly: list[tuple[float, float]],
    n_cyc: int,
    n_out: int,
    n_in: int,
    *,
    max_ratio: float = 1.6,
    max_add: int = 16,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], int, int]:
    """Delete fat SW/SE AABB wrap cells by splitting them on the AABB legs.

    Angle matching on a cup puts a ~chord-long inner edge on the SW (and SE)
    AABB corner cell — that face is checkMesh skew > 4. Insert CCW metal
    midpoints; outer nodes stay on the AABB (H-blocks remain Cartesian).
    Does not restore offset-O + TFI H.
    """
    south, east, north, west = _aabb_sides(outer, n_cyc, n_out, n_in)
    inner = inner.copy()
    added = 0
    n_cyc_u, n_out_u, n_in_u = n_cyc, n_out, n_in
    while added < max_add:
        ring_o = _ring_from_aabb_sides(south, east, north, west)
        if inner.shape[0] != ring_o.shape[0]:
            raise RuntimeError("inner/outer mismatch while splitting AABB wrap")
        n = inner.shape[0]
        ds = np.array(
            [
                float(np.hypot(inner[(i + 1) % n, 0] - inner[i, 0], inner[(i + 1) % n, 1] - inner[i, 1]))
                for i in range(n)
            ]
        )
        med = float(np.median(ds)) or 1e-16
        i = int(np.argmax(ds))
        n_s = south.shape[0] - 1
        n_e = east.shape[0] - 1
        n_n = north.shape[0] - 1
        # Only split west/east legs (SW/SE AABB wrap). South/north must stay
        # equal-count so cyclic x-nodes pair.
        order = np.argsort(-ds)
        picked = None
        for cand in order:
            if ds[cand] <= max_ratio * med:
                break
            if n_s <= cand < n_s + n_e or cand >= n_s + n_e + n_n:
                picked = int(cand)
                break
        if picked is None:
            break
        i = picked
        i2 = (i + 1) % n
        q0, q1 = ring_o[i], ring_o[i2]
        mid_out = 0.5 * (q0 + q1)
        if abs(q0[0] - q1[0]) < 1e-14:
            mid_out[0] = q0[0]
        elif abs(q0[1] - q1[1]) < 1e-14:
            mid_out[1] = q0[1]
        if n_s <= i < n_s + n_e:
            east = np.insert(east, i - n_s + 1, mid_out, axis=0)
            n_out_u += 1
        else:
            west = np.insert(west, i - n_s - n_e - n_n + 1, mid_out, axis=0)
            n_in_u += 1
        added += 1
        # Re-pair metal by polar angle on the refined AABB. Inserting metal
        # midpoints independently crossed radials (folded last-layer quads).
        outer = _ring_from_aabb_sides(south, east, north, west)
        inner = inner_match_by_angle(poly, outer)
    outer = _ring_from_aabb_sides(south, east, north, west)
    if inner.shape[0] != outer.shape[0]:
        inner = inner_match_by_angle(poly, outer)
    return inner, outer, (south, east, north, west), added, n_in_u


def tfi_block(
    south: np.ndarray,
    north: np.ndarray,
    west: np.ndarray,
    east: np.ndarray,
) -> np.ndarray:
    """Bilinear TFI. south/north: (nx+1, 2); west/east: (ny+1, 2)."""
    nx = south.shape[0] - 1
    ny = west.shape[0] - 1
    if north.shape[0] != nx + 1 or east.shape[0] != ny + 1:
        raise RuntimeError(
            f"TFI edge mismatch south={south.shape} north={north.shape} "
            f"west={west.shape} east={east.shape}"
        )
    pts = np.zeros((nx + 1, ny + 1, 2), dtype=float)
    for i in range(nx + 1):
        xi = 0.0 if nx == 0 else i / nx
        for j in range(ny + 1):
            eta = 0.0 if ny == 0 else j / ny
            pts[i, j] = (
                (1 - eta) * south[i]
                + eta * north[i]
                + (1 - xi) * west[j]
                + xi * east[j]
                - (1 - xi) * (1 - eta) * south[0]
                - xi * (1 - eta) * south[-1]
                - (1 - xi) * eta * north[0]
                - xi * eta * north[-1]
            )
    return pts


def tfi_block_eta_pack(
    south: np.ndarray,
    north: np.ndarray,
    west: np.ndarray,
    east: np.ndarray,
    r: float = 1.0,
    dense_at: str = "north",
) -> np.ndarray:
    """TFI with geometric η packing toward north (O) or south."""
    nx = south.shape[0] - 1
    ny = west.shape[0] - 1
    if north.shape[0] != nx + 1 or east.shape[0] != ny + 1:
        raise RuntimeError(
            f"TFI edge mismatch south={south.shape} north={north.shape} "
            f"west={west.shape} east={east.shape}"
        )
    r = max(float(r), 1.0)
    pts = np.zeros((nx + 1, ny + 1, 2), dtype=float)
    for i in range(nx + 1):
        xi = 0.0 if nx == 0 else i / nx
        for j in range(ny + 1):
            if ny == 0:
                eta = 0.0
            elif r <= 1.0 + 1e-12:
                eta = j / ny
            elif dense_at == "north":
                # pack toward eta=1
                eta = float(_stretch(j, ny, r))
            else:
                eta = float(1.0 - _stretch(ny - j, ny, r))
            # Boundary west/east already include packing if caller packed them;
            # blend with parametric edges at the same eta via linear edge eval.
            wj = (1.0 - eta) * west[0] + eta * west[-1]
            ej = (1.0 - eta) * east[0] + eta * east[-1]
            # Prefer actual west/east node spacing when j maps 1:1
            if r <= 1.0 + 1e-12 or True:
                # remap j onto packed index for west/east samples
                if r > 1.0 + 1e-12 and dense_at == "north":
                    jj = float(_stretch(j, ny, r)) * ny
                elif r > 1.0 + 1e-12:
                    jj = float(1.0 - _stretch(ny - j, ny, r)) * ny
                else:
                    jj = float(j)
                j0 = int(np.floor(jj)); j1 = min(j0 + 1, ny); tj = jj - j0
                wj = (1.0 - tj) * west[j0] + tj * west[j1]
                ej = (1.0 - tj) * east[j0] + tj * east[j1]
            pts[i, j] = (
                (1 - eta) * south[i]
                + eta * north[i]
                + (1 - xi) * wj
                + xi * ej
                - (1 - xi) * (1 - eta) * south[0]
                - xi * (1 - eta) * south[-1]
                - (1 - xi) * eta * north[0]
                - xi * eta * north[-1]
            )
    return pts


def resample_closed(poly: list[tuple[float, float]], n: int) -> np.ndarray:
    """n points around a closed polygon, even arc-length, no duplicate wrap."""
    pts = poly[:-1] if poly and poly[0] == poly[-1] else list(poly)
    chain = pts + [pts[0]]
    samp = _resample(chain, n)
    return np.array(samp[:-1], dtype=float)


def offset_closed(inner: np.ndarray, dist: float, n_smooth: int = 8) -> np.ndarray:
    """Outward offset of a CCW closed loop (fluid is outside / right of walk)."""
    n = inner.shape[0]
    normals = np.zeros_like(inner)
    for i in range(n):
        p0 = inner[(i - 1) % n]
        p1 = inner[i]
        p2 = inner[(i + 1) % n]
        e1 = p1 - p0
        e2 = p2 - p1
        l1 = float(np.hypot(e1[0], e1[1])) or 1e-16
        l2 = float(np.hypot(e2[0], e2[1])) or 1e-16
        # Right-normal of CCW walk = outward.
        n1 = np.array([e1[1] / l1, -e1[0] / l1])
        n2 = np.array([e2[1] / l2, -e2[0] / l2])
        nn = n1 + n2
        ln = float(np.hypot(nn[0], nn[1])) or 1e-16
        normals[i] = nn / ln
    for _ in range(n_smooth):
        normals = 0.5 * normals + 0.25 * np.roll(normals, 1, axis=0) + 0.25 * np.roll(normals, -1, axis=0)
        ln = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.clip(ln, 1e-16, None)
    return inner + dist * normals


def smooth_ogrid(pts: np.ndarray, n_iter: int = 50, omega: float = 0.55) -> np.ndarray:
    """Laplacian smooth of interior radial layers. Wall and outer ring stay put."""
    n_i, n_j, _ = pts.shape
    cur = pts.copy()
    for _ in range(n_iter):
        new = cur.copy()
        for j in range(1, n_j - 1):
            avg = 0.25 * (
                np.roll(cur[:, j, :], 1, axis=0)
                + np.roll(cur[:, j, :], -1, axis=0)
                + cur[:, j - 1, :]
                + cur[:, j + 1, :]
            )
            new[:, j, :] = (1.0 - omega) * cur[:, j, :] + omega * avg
        new[:, 0, :] = pts[:, 0, :]
        new[:, -1, :] = pts[:, -1, :]
        cur = new
    return cur


def smooth_block(pts: np.ndarray, n_iter: int = 20, omega: float = 0.5) -> np.ndarray:
    """Laplacian smooth of a structured block, boundary nodes fixed."""
    ni, nj, _ = pts.shape
    cur = pts.copy()
    for _ in range(n_iter):
        new = cur.copy()
        for i in range(1, ni - 1):
            for j in range(1, nj - 1):
                avg = 0.25 * (cur[i - 1, j] + cur[i + 1, j] + cur[i, j - 1] + cur[i, j + 1])
                new[i, j] = (1.0 - omega) * cur[i, j] + omega * avg
        cur = new
    return cur


def _corner_indices(ring: np.ndarray) -> tuple[int, int, int, int]:
    """SW, SE, NE, NW indices on a closed ring (geometric AABB corners)."""
    x, y = ring[:, 0], ring[:, 1]
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    targets = [
        (xmin, ymin),
        (xmax, ymin),
        (xmax, ymax),
        (xmin, ymax),
    ]
    used: set[int] = set()
    out: list[int] = []
    for tx, ty in targets:
        best_i, best_d = 0, 1e99
        for i in range(ring.shape[0]):
            if i in used:
                continue
            d = (ring[i, 0] - tx) ** 2 + (ring[i, 1] - ty) ** 2
            if d < best_d:
                best_d, best_i = d, i
        out.append(best_i)
        used.add(best_i)
    return out[0], out[1], out[2], out[3]


def _arc_indices(n: int, i0: int, i1: int) -> list[int]:
    """Inclusive i0 → i1 walking +i modulo n."""
    out = [i0]
    i = i0
    guard = 0
    while i != i1:
        i = (i + 1) % n
        out.append(i)
        guard += 1
        if guard > n + 2:
            break
    return out


def _arc_choose(ring: np.ndarray, i0: int, i1: int, axis: int, want_max: bool) -> list[int]:
    """Pick the i0→i1 arc whose mean coordinate is the geometric side."""
    n = int(ring.shape[0])
    fwd = _arc_indices(n, i0, i1)
    bak = [i0]
    i = i0
    guard = 0
    while i != i1:
        i = (i - 1) % n
        bak.append(i)
        guard += 1
        if guard > n + 2:
            break

    def mean(idx: list[int]) -> float:
        return float(ring[idx, axis].mean())

    if want_max:
        return fwd if mean(fwd) >= mean(bak) else bak
    return fwd if mean(fwd) <= mean(bak) else bak


def _lin(p0: np.ndarray, p1: np.ndarray, n_seg: int) -> np.ndarray:
    p0 = np.asarray(p0, dtype=float).reshape(2)
    p1 = np.asarray(p1, dtype=float).reshape(2)
    t = np.linspace(0.0, 1.0, n_seg + 1)[:, None]
    return (1.0 - t) * p0[None, :] + t * p1[None, :]


def _lin_pack_end(p0, p1, n_seg: int, r: float) -> np.ndarray:
    """Polyline p0→p1 with geometric pack toward p1 (same law as ``_xs_pack_hi``)."""
    p0 = np.asarray(p0, dtype=float).reshape(2)
    p1 = np.asarray(p1, dtype=float).reshape(2)
    r = _inlet_pack_r(n_seg, r)
    t = np.array([1.0 - _stretch(n_seg - j, n_seg, r) for j in range(n_seg + 1)], dtype=float)[:, None]
    return (1.0 - t) * p0[None, :] + t * p1[None, :]


def _lin_pack_start(p0, p1, n_seg: int, r: float) -> np.ndarray:
    """Polyline p0→p1 with geometric pack toward p0 (dense at start / O side)."""
    return _lin_pack_end(p1, p0, n_seg, r)[::-1].copy()


def _y_on_polyline_at_x(poly: np.ndarray, x: float) -> float:
    """Linear y(x) on a polyline assumed mostly monotone in x."""
    p = np.asarray(poly, dtype=float)
    x = float(x)
    if p.shape[0] < 2:
        return float(p[0, 1])
    # exact hits
    for i in range(p.shape[0]):
        if abs(float(p[i, 0]) - x) <= 1e-14:
            return float(p[i, 1])
    best = None
    for i in range(p.shape[0] - 1):
        x0, y0 = float(p[i, 0]), float(p[i, 1])
        x1, y1 = float(p[i + 1, 0]), float(p[i + 1, 1])
        if (x0 - x) * (x1 - x) <= 0.0 and abs(x1 - x0) > 1e-16:
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
        mid = 0.5 * (x0 + x1)
        d = abs(mid - x)
        if best is None or d < best[0]:
            best = (d, y0 + ((x - x0) / (x1 - x0 + 1e-30)) * (y1 - y0))
    return float(best[1]) if best else float(p[0, 1])


def _ray_block_vertical(
    y_lo_of_x,
    y_hi_of_x,
    xs: np.ndarray,
    n_fill: int,
    *,
    pack_r: float = 1.0,
    dense_at: str = "lo",
) -> np.ndarray:
    """H-block with vertical rays at xs. j=0 at y_lo, j=-1 at y_hi.

    dense_at 'lo' packs toward the lower edge (O/mouth); 'hi' toward upper.
    """
    xs = np.asarray(xs, dtype=float).reshape(-1)
    n = int(xs.shape[0])
    pts = np.zeros((n, n_fill + 1, 2), dtype=float)
    r = max(float(pack_r), 1.0)
    def _y_at(spec, i, x):
        if callable(spec):
            return float(spec(float(x)))
        if isinstance(spec, (int, float, np.floating)):
            return float(spec)
        return float(np.asarray(spec, dtype=float).reshape(-1)[i])

    for i, x in enumerate(xs):
        y0 = _y_at(y_lo_of_x, i, x)
        y1 = _y_at(y_hi_of_x, i, x)
        p0 = np.array([x, y0], dtype=float)
        p1 = np.array([x, y1], dtype=float)
        if r > 1.0 + 1e-12 and dense_at == "lo":
            line = _lin_pack_start(p0, p1, n_fill, r)
        elif r > 1.0 + 1e-12 and dense_at == "hi":
            line = _lin_pack_end(p0, p1, n_fill, r)
        else:
            line = _lin(p0, p1, n_fill)
        pts[i] = line
    return pts



def _ray_block_horizontal(
    edge_lo_x,
    x_hi_of_y,
    ys: np.ndarray,
    n_seg: int,
    *,
    pack_r: float = 1.0,
    dense_at: str = "hi",
) -> np.ndarray:
    """H-block with horizontal rays at ys. i=0 at x_lo, i=-1 at x_hi.

    edge_lo_x: callable/array/float for x at each y on the low-x side.
    x_hi_of_y: callable/array/float for high-x side.
    dense_at 'hi' packs toward high-x (O); 'lo' toward low-x.
    Returns pts[n_seg+1, len(ys), 2].
    """
    ys = np.asarray(ys, dtype=float).reshape(-1)
    n = int(ys.shape[0])
    pts = np.zeros((n_seg + 1, n, 2), dtype=float)
    r = max(float(pack_r), 1.0)

    def _x_at(spec, j, y):
        if callable(spec):
            return float(spec(float(y)))
        if isinstance(spec, (int, float, np.floating)):
            return float(spec)
        return float(np.asarray(spec, dtype=float).reshape(-1)[j])

    for j, y in enumerate(ys):
        x0 = _x_at(edge_lo_x, j, y)
        x1 = _x_at(x_hi_of_y, j, y)
        p0 = np.array([x0, y], dtype=float)
        p1 = np.array([x1, y], dtype=float)
        if r > 1.0 + 1e-12 and dense_at == "hi":
            line = _lin_pack_end(p0, p1, n_seg, r)
        elif r > 1.0 + 1e-12 and dense_at == "lo":
            line = _lin_pack_start(p0, p1, n_seg, r)
        else:
            line = _lin(p0, p1, n_seg)
        pts[:, j, :] = line
    return pts



def _ray_block_sheared(
    edge_o: np.ndarray,
    x_target: float,
    n_seg: int,
    beta_deg: float,
    *,
    pack_r: float = 1.0,
    dense_at_o: bool = True,
    toward_inlet: bool = True,
) -> np.ndarray:
    """Sheared H-block: each O node maps to x_target along ±W(β).

    Returns pts[n_seg+1, n_o, 2] with i=0 at far (x_target side) when toward_inlet
    else i=0 at O; O column is always the dense side when dense_at_o.
    """
    from .oh_shock import sheared_inlet_edge, sheared_outlet_point

    edge_o = np.asarray(edge_o, dtype=float)
    n_o = int(edge_o.shape[0])
    pts = np.zeros((n_seg + 1, n_o, 2), dtype=float)
    r = max(float(pack_r), 1.0)
    for j in range(n_o):
        p_o = edge_o[j]
        if toward_inlet:
            p_far = sheared_inlet_edge(p_o, x_target, beta_deg)
        else:
            p_far = sheared_outlet_point(p_o, x_target, beta_deg)
        if dense_at_o and r > 1.0 + 1e-12:
            # i=0 at far, i=-1 at O (pack toward O)
            line = _lin_pack_end(p_far, p_o, n_seg, r)
        elif dense_at_o:
            line = _lin(p_far, p_o, n_seg)
        else:
            line = _lin(p_o, p_far, n_seg)
        pts[:, j, :] = line
    return pts

def smooth_rect_block(pts: np.ndarray, n_iter: int = 40, omega: float = 0.45) -> np.ndarray:
    """Laplacian smooth of interior nodes; boundary edges stay put."""
    cur = np.asarray(pts, dtype=float).copy()
    if cur.ndim != 3 or cur.shape[0] < 3 or cur.shape[1] < 3:
        return cur
    nx, ny = cur.shape[0], cur.shape[1]
    for _ in range(int(n_iter)):
        new = cur.copy()
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                new[i, j] = (1.0 - omega) * cur[i, j] + omega * 0.25 * (
                    cur[i - 1, j] + cur[i + 1, j] + cur[i, j - 1] + cur[i, j + 1]
                )
        if min_cell_area_2d_rect(new) <= 0:
            return cur
        cur = new
    return cur




def _force_mono_x(pts: np.ndarray) -> np.ndarray:
    xs = np.asarray(pts[:, 0], dtype=float).copy()
    for i in range(1, xs.shape[0]):
        if xs[i] <= xs[i - 1]:
            xs[i] = xs[i - 1] + 1e-10
    out = pts.copy()
    out[:, 0] = xs
    return out


def _arclen_frac_index(idx: list[int], ring: np.ndarray, frac: float) -> int:
    s = [0.0]
    for k in range(1, len(idx)):
        p, q = ring[idx[k - 1]], ring[idx[k]]
        s.append(s[-1] + float(np.hypot(q[0] - p[0], q[1] - p[1])))
    tot = s[-1] if s[-1] > 0 else 1.0
    target = float(frac) * tot
    return int(min(range(len(s)), key=lambda i: abs(s[i] - target)))


def _down_u_splits(ring: np.ndarray):
    """Tips, cavity (Lt→Rt CCW), outer (Rt→Lt CCW), inner 22/78% corners, outer high-band.

    Returns None if the profile is not a down-opening U with a real cavity.
    """
    n = int(ring.shape[0])
    if n < 16:
        return None
    cx = 0.5 * (float(ring[:, 0].min()) + float(ring[:, 0].max()))
    left = np.where(ring[:, 0] < cx)[0]
    right = np.where(ring[:, 0] >= cx)[0]
    if left.size < 4 or right.size < 4:
        return None
    iLt = int(left[np.argmin(ring[left, 1])])
    iRt = int(right[np.argmin(ring[right, 1])])
    inner = _arc_indices(n, iLt, iRt)
    outer = _arc_indices(n, iRt, iLt)
    if float(ring[inner, 1].max()) > float(ring[outer, 1].max()) + 1e-12:
        # CCW Lt→Rt was the back, not the cavity
        return None
    yspan = float(ring[:, 1].max() - ring[:, 1].min())
    xspan = float(ring[:, 0].max() - ring[:, 0].min())
    ymin = float(ring[:, 1].min())
    depth = float(ring[inner, 1].max() - min(ring[iLt, 1], ring[iRt, 1]))
    # Two bottom tips (pointed U). A camber foil has one PS min-y, not twin stems.
    if yspan <= 1e-16 or depth < 0.15 * yspan:
        return None
    if abs(ring[iLt, 0] - ring[iRt, 0]) < 0.25 * max(xspan, 1e-16):
        return None
    if float(ring[iLt, 1]) > ymin + 0.08 * yspan or float(ring[iRt, 1]) > ymin + 0.08 * yspan:
        return None
    kNW = _arclen_frac_index(inner, ring, 0.22)
    kNE = _arclen_frac_index(inner, ring, 0.78)
    if not (1 <= kNW < kNE <= len(inner) - 2):
        return None
    xs = ring[outer, 0]
    ys = ring[outer, 1]
    kE = int(np.argmax(xs + ys))
    kW = int(np.argmax(-xs + ys))
    ymax = float(ys.max())
    # Wider high-band: shoulders → north (vertical rays) not west_up (horizontal ~70° NO)
    ks = np.where(ys >= 0.72 * ymax)[0]
    if ks.size >= 3:
        kE_top, kW_top = int(ks[0]), int(ks[-1])
    else:
        kTop = int(np.argmax(ys))
        kE_top = max(kE + 1, kTop - max(2, len(outer) // 20))
        kW_top = min(kW - 1, kTop + max(2, len(outer) // 20))
    if not (1 <= kE < kE_top < kW_top < kW <= len(outer) - 2):
        return None
    return {
        "iLt": iLt,
        "iRt": iRt,
        "inner": inner,
        "outer": outer,
        "kNW": kNW,
        "kNE": kNE,
        "kE": kE,
        "kW": kW,
        "kE_top": kE_top,
        "kW_top": kW_top,
        "depth": depth,
        "yspan": yspan,
    }


def profile_has_cavity(poly: list[tuple[float, float]]) -> bool:
    pts = poly[:-1] if poly and poly[0] == poly[-1] else list(poly)
    if polygon_signed_area(pts) < 0:
        pts = list(reversed(pts))
    ring = np.array(pts, dtype=float)
    return _down_u_splits(ring) is not None



def _le_te_side_arcs(ring: np.ndarray, west_frac: float = 0.14, east_frac: float = 0.14):
    """Four CCW metal arcs: west=LE, north=SS, east=TE, south=PS.

    AABB-nearest domain corners pin NE and SE both at TE (3–7 point fans) —
    that is the checkMesh non-ortho / pyramid / skew hole. Corners sit a
    fixed fraction off LE (min x) and TE (max x). Walk is metal-CCW.
    """
    n = int(ring.shape[0])
    iW = int(np.argmin(ring[:, 0]))
    iE = int(np.argmax(ring[:, 0]))
    nw = max(10, int(west_frac * n))
    ne = max(10, int(east_frac * n))
    iSW = (iW - nw // 2) % n
    iNW = (iW + (nw - nw // 2)) % n
    iNE = (iE - ne // 2) % n
    iSE = (iE + (ne - ne // 2)) % n
    west = _arc_indices(n, iSW, iNW)
    north = _arc_indices(n, iNW, iNE)
    east = _arc_indices(n, iNE, iSE)
    south = _arc_indices(n, iSE, iSW)
    return south, east, north, west


def _untangle_block(hb: np.ndarray, n_pass: int = 10) -> np.ndarray:
    """Laplacian interior only. Boundary (O-ring / pitch rectangle) stays put."""
    cur = hb.copy()
    for _ in range(n_pass):
        cur = smooth_block(cur, n_iter=8, omega=0.5)
        if min_cell_area_2d_rect(cur) > 0:
            return cur
    return cur



def _trim_edge_steep(edge: np.ndarray, max_dx_dy: float = 1.1, from_start: bool = True) -> np.ndarray:
    """Drop tip segments whose |dx/dy| exceeds max_dx_dy (keeps stem for horizontal H)."""
    e = np.asarray(edge, dtype=float)
    if e.shape[0] < 4:
        return e
    if not from_start:
        e = e[::-1].copy()
    keep = 0
    for i in range(e.shape[0] - 1):
        dx = abs(float(e[i + 1, 0] - e[i, 0]))
        dy = abs(float(e[i + 1, 1] - e[i, 1]))
        if dy < 1e-16 or (dx / dy) > max_dx_dy:
            keep = i + 1
            continue
        break
    out = e[keep:]
    if out.shape[0] < 3:
        out = e[max(0, e.shape[0] // 5):]
    if not from_start:
        out = out[::-1].copy()
    return out


def _join_xs(*arrs: np.ndarray) -> np.ndarray:
    """Concatenate increasing x-arrays, dropping shared endpoints."""
    chunks: list[np.ndarray] = []
    for a in arrs:
        a = np.asarray(a, dtype=float).reshape(-1)
        if a.size == 0:
            continue
        if not chunks:
            chunks.append(a)
            continue
        if abs(float(a[0]) - float(chunks[-1][-1])) <= 1e-12:
            chunks.append(a[1:])
        else:
            chunks.append(a)
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=float)


def _interp_arc_xs(pts: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Interpolate an open polyline onto xs (vertical projection). Monotone-x."""
    p = np.asarray(pts, dtype=float)
    xs = np.asarray(xs, dtype=float).reshape(-1)
    if p.shape[0] < 2:
        return np.column_stack([xs, np.full(xs.shape[0], float(p[0, 1]))])
    if float(p[0, 0]) > float(p[-1, 0]):
        p = p[::-1].copy()
    x = p[:, 0].copy()
    y = p[:, 1].copy()
    for i in range(1, x.shape[0]):
        if x[i] <= x[i - 1]:
            x[i] = x[i - 1] + 1e-12
    ys = np.interp(xs, x, y)
    return np.column_stack([xs, ys])


def _outer_dxdy_corners(outer_idx: list[int], ring: np.ndarray, max_dx_dy: float = 1.0):
    """Walk outer Rt→Lt. Return k_SE, k_E, k_W, k_SW into outer_idx (45° |dx|=|dy|).

    Pattern: tip-wrap (shallow) → right stem (steep) → back (shallow) → left stem
    (steep) → tip-wrap. Corners are the four steepness transitions.
    """
    n = len(outer_idx)
    if n < 8:
        return 1, n // 4, (3 * n) // 4, n - 2
    dxdy = []
    for k in range(n - 1):
        d = ring[outer_idx[k + 1]] - ring[outer_idx[k]]
        dxdy.append(abs(float(d[0])) / max(abs(float(d[1])), 1e-16))
    steep = np.array([v <= max_dx_dy for v in dxdy], dtype=bool)
    trans: list[int] = []
    for k in range(1, steep.shape[0]):
        if bool(steep[k]) != bool(steep[k - 1]):
            trans.append(k)
    if len(trans) >= 4:
        # Prefer the first four: SE, E (back start), W (back end), SW
        return int(trans[0]), int(trans[1]), int(trans[2]), int(trans[3])
    # Fallback: 45° via argmax(x+y) / argmax(-x+y) on the back, plus 5% tip trim
    xs = ring[outer_idx, 0]
    ys = ring[outer_idx, 1]
    kE = int(np.argmax(xs + ys))
    kW = int(np.argmax(-xs + ys))
    kSE = max(2, n // 20)
    kSW = min(n - 3, n - n // 20)
    if not (kSE < kE < kW < kSW):
        kSE, kE, kW, kSW = 2, n // 4, (3 * n) // 4, n - 3
    return kSE, kE, kW, kSW


def _pos_block(hb: np.ndarray, name: str, keep_edges: bool = False) -> np.ndarray:
    """Orientation with strictly positive min 2D area. No pinched-quad waiver.

    Do not Laplacian-untangle a block that is already positive: thin north
    layers (offset back almost on the cyclic) invert under that smoother.
    keep_edges: do not reverse i/j (shared O/H edges must stay put).
    """
    amin0 = min_cell_area_2d_rect(hb)
    if amin0 > 0:
        return hb
    hb = _untangle_block(hb)
    if min_cell_area_2d_rect(hb) > 0:
        return hb
    if keep_edges:
        # i-flip only: keep j=0 / j=-1 as the O and cyclic edges.
        cands = (hb, hb[::-1, :, :].copy())
        best = max(cands, key=min_cell_area_2d_rect)
        amin = min_cell_area_2d_rect(best)
        if amin <= 0:
            raise RuntimeError(f"folded H-block {name} min_area={amin:.3e}")
        return best
    cands = (hb, hb[:, ::-1, :].copy(), hb[::-1, :, :].copy(), hb[::-1, ::-1, :].copy())
    best = max(cands, key=min_cell_area_2d_rect)
    amin = min_cell_area_2d_rect(best)
    if amin <= 0:
        raise RuntimeError(f"folded H-block {name} min_area={amin:.3e}")
    return best


def _resample_xy(
    chain_idx: list[int],
    ring: np.ndarray,
    n_seg: int,
    le_cluster: float = 1.0,
    *,
    xmin: float | None = None,
    xmax: float | None = None,
) -> np.ndarray:
    pts = [tuple(ring[i]) for i in chain_idx]
    if max(float(le_cluster), 1.0) <= 1.0 + 1e-12:
        return np.array(_resample(pts, n_seg), dtype=float)
    return np.array(
        _resample_le_cluster(pts, n_seg, le_cluster, xmin=xmin, xmax=xmax),
        dtype=float,
    )


def _ring_corners_to_domain(ring: np.ndarray, x_in: float, x_out: float, y_bot: float, y_top: float):
    """SW, SE, NE, NW indices on a closed ring, nearest to the pitch-rectangle corners."""
    targets = [(x_in, y_bot), (x_out, y_bot), (x_out, y_top), (x_in, y_top)]
    used: set[int] = set()
    out: list[int] = []
    for tx, ty in targets:
        best_i, best_d = 0, 1e99
        for i in range(ring.shape[0]):
            if i in used:
                continue
            d = (ring[i, 0] - tx) ** 2 + (ring[i, 1] - ty) ** 2
            if d < best_d:
                best_d, best_i = d, i
        out.append(best_i)
        used.add(best_i)
    return out[0], out[1], out[2], out[3]





def _project_to_seg(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    v = b - a
    den = float(np.dot(v, v))
    if den < 1e-30:
        return np.asarray(a, dtype=float).copy()
    t = float(np.clip(np.dot(p - a, v) / den, 0.0, 1.0))
    return a + t * v


def te_wake_block(
    ogrid: np.ndarray,
    te_wall: np.ndarray,
    te_outer: np.ndarray,
) -> np.ndarray:
    """TE wake H; TE nodes projected onto cut chords (unkinked, n_span=2)."""
    e0 = np.asarray(ogrid[0], dtype=float).copy()
    e1 = np.asarray(ogrid[-1], dtype=float).copy()
    tw = np.asarray(te_wall, dtype=float).reshape(2)
    to = np.asarray(te_outer, dtype=float).reshape(2)
    south = np.stack([e0[0], tw, e1[0]], axis=0)
    north = np.stack([e0[-1], to, e1[-1]], axis=0)
    hb = tfi_block(south, north, e0, e1)
    if min_cell_area_2d_rect(hb) <= 0:
        hb = tfi_block(south, north, e1, e0)
    if min_cell_area_2d_rect(hb) <= 0:
        hb = tfi_block(north, south, e0, e1)
    if min_cell_area_2d_rect(hb) <= 0:
        hb = tfi_block(north, south, e1, e0)
    return _pos_block(hb, "te_wake")



def _snap_points(hb: np.ndarray, mapping: list[tuple[np.ndarray, np.ndarray]], eps: float = 1e-7) -> np.ndarray:
    """Move any block node near src onto dst (TE → chord projection)."""
    out = np.asarray(hb, dtype=float).copy()
    flat = out.reshape(-1, 2)
    for src_p, dst_p in mapping:
        d = np.linalg.norm(flat - src_p, axis=1)
        flat[d < eps] = dst_p
    return out


def build_offset_oh(
    poly0: list[tuple[float, float]],
    *,
    x_in: float,
    x_out: float,
    y_bot: float,
    y_top: float,
    n_in: int,
    n_out: int,
    n_cyc: int,
    n_rad: int,
    n_fill: int,
    stretch: float,
    d_o: float,
    n_out_x: int | None = None,
    le_cluster: float = 1.0,
    inlet_stretch: float | None = None,
    dump_xs: np.ndarray | None = None,
    dump_rx: float | None = None,
    beta1_deg: float = 65.0,
    beta2_deg: float = -65.0,
) -> tuple[np.ndarray, list[np.ndarray], float, list[str]]:
    """Offset O-collar + axis-aligned H-blocks to the pitch rectangle.

    Shallow-U cavity TFI (22/78 inner split) put collinear west/north edges on
    the inner arc → ~89° non-ortho / pyramid / skew at the stem. Horizontal TFI
    of the tip-wrap onto a rectangular inlet did the same at the LE tip. H-blocks
    here are vertical rays (cavity+south, back→y_top) and horizontal rays (steep
    stems→inlet/outlet). Cyclic top/bottom share x. 1:1 wall-normal O–H nodes.
    """
    notes: list[str] = []
    n_out_x = int(n_out_x or n_out)
    r_inlet = max(float(inlet_stretch if inlet_stretch is not None else stretch), 1.0)
    n_hi = max(4 * (2 * n_cyc + n_in + n_out), 400)
    inner_hi = resample_closed(poly0, n_hi)
    if polygon_signed_area([(float(x), float(y)) for x, y in inner_hi]) < 0:
        inner_hi = inner_hi[::-1].copy()
    d_use = float(d_o)
    outer_hi = None
    for _try in range(8):
        cand = offset_closed(inner_hi, d_use, n_smooth=16)
        if float(cand[:, 1].min()) > y_bot + 1e-6 and float(cand[:, 1].max()) < y_top - 1e-6:
            outer_hi = cand
            break
        d_use *= 0.6
    if outer_hi is None:
        raise RuntimeError("offset-O ring collides with cyclic pitch boundary")
    if d_use < d_o - 1e-16:
        notes.append(f"d_o shrunk {d_o:.3g} → {d_use:.3g} m so offset clears cyclics")

    spl = _down_u_splits(outer_hi)
    if spl is None:
        raise RuntimeError("offset-O U-split failed (not a down-opening cavity)")
    inner_idx = spl["inner"]
    outer_idx = spl["outer"]
    notes.append(
        f"U-cavity depth {spl['depth']*1e3:.2f} mm / y-span {spl['yspan']*1e3:.2f} mm; "
        f"axis-aligned H (vertical cavity/back, horizontal stems) + offset collar"
    )

    le_r = max(float(le_cluster), 1.0)
    if le_r > 1.0 + 1e-12:
        notes.append(
            f"streamwise LE cluster le_cluster={le_r:.3g} on wall/offset arcs "
            "(Δs biased to min-x per arc; O wall-normal stretch unchanged)"
        )

    def _side(ring, idx, nseg):
        return _resample_xy(idx, ring, nseg, le_r)

    # 45° |dx|=|dy| corners on the outer back — not 22/78 inner or 0.72 ymax.
    # Inner is a shallow arc (tangents ≲30°): do NOT split it into west/north/east
    # TFI (those corners are collinear → 89° non-ortho / pyramid / skew).
    kSE, kE, kW, kSW = _outer_dxdy_corners(outer_idx, outer_hi, max_dx_dy=1.0)
    n_inner = int(n_cyc)
    n_fill_h = max(int(n_fill), 4)
    n_tip = max(6, int(n_out) // 2)

    iLt = int(spl["iLt"])
    iRt = int(spl["iRt"])
    x_Lt = float(inner_hi[iLt, 0])
    x_Rt = float(inner_hi[iRt, 0])
    xs_blade = np.linspace(x_Lt, x_Rt, n_inner + 1)
    x_kE = float(outer_hi[outer_idx[kE], 0])
    x_kW = float(outer_hi[outer_idx[kW], 0])
    i_nw_b = int(np.argmin(np.abs(xs_blade - x_kW)))
    i_ne_b = int(np.argmin(np.abs(xs_blade - x_kE)))
    if i_ne_b < i_nw_b:
        i_nw_b, i_ne_b = i_ne_b, i_nw_b
    i_nw_b = max(1, min(i_nw_b, n_inner - 2))
    i_ne_b = max(i_nw_b + 4, min(i_ne_b, n_inner - 1))
    # Same node count as south_o (tip wraps + inner) so N/S cyclic faces pair.
    n_south_o = n_inner + 2 * n_tip
    xs_north = np.linspace(min(x_kW, x_kE), max(x_kW, x_kE), n_south_o + 1)

    met_cav = _interp_arc_xs(inner_hi[inner_idx], xs_blade)
    met_north = _interp_arc_xs(inner_hi[outer_idx[kE : kW + 1]], xs_north)
    # CCW ring: Lt → inner → Rt → east tip/stem → north (Rt-side → Lt-side) → west stem/tip
    met_east_tip = _side(inner_hi, outer_idx[: kSE + 1], n_tip)
    met_east_stem = _side(inner_hi, outer_idx[kSE : kE + 1], n_out)
    met_west_stem = _side(inner_hi, outer_idx[kW : kSW + 1], n_in)
    met_west_tip = _side(inner_hi, outer_idx[kSW :], n_tip)
    # north metal is kW→kE in +x; CCW walk is kE→kW
    met_north_ccw = met_north[::-1].copy()

    inner = np.concatenate(
        [
            met_cav[:-1],
            met_east_tip[:-1],
            met_east_stem[:-1],
            met_north_ccw[:-1],
            met_west_stem[:-1],
            met_west_tip[:-1],
        ],
        axis=0,
    )
    # 1:1 wall-normal outer (independent outer resample was the O non-ortho hole).
    outer = offset_closed(inner, d_use, n_smooth=12)
    if float(outer[:, 1].min()) <= y_bot + 1e-6 or float(outer[:, 1].max()) >= y_top - 1e-6:
        for _try in range(8):
            d_use *= 0.6
            outer = offset_closed(inner, d_use, n_smooth=12)
            if float(outer[:, 1].min()) > y_bot + 1e-6 and float(outer[:, 1].max()) < y_top - 1e-6:
                break
        else:
            raise RuntimeError("wall-normal offset-O collides with cyclic pitch boundary")
        notes.append(f"d_o shrunk for wall-normal outer → {d_use:.3g} m")
    # Rebuild H-facing outer arcs from the same-index outer ring (conformal O–H).
    n_cav_n = int(met_cav.shape[0] - 1)
    n_east_tip = int(met_east_tip.shape[0] - 1)
    n_east_stem = int(met_east_stem.shape[0] - 1)
    n_north_n = int(met_north_ccw.shape[0] - 1)
    n_west_stem = int(met_west_stem.shape[0] - 1)
    n_west_tip = int(met_west_tip.shape[0] - 1)
    lens = [n_cav_n, n_east_tip, n_east_stem, n_north_n, n_west_stem, n_west_tip]
    if int(sum(lens)) != int(inner.shape[0]):
        raise RuntimeError(f"O ring lens {sum(lens)} != inner {inner.shape[0]}")
    offs = [0]
    for L in lens:
        offs.append(offs[-1] + L)

    def _arc(i0: int, i1: int) -> np.ndarray:
        pts = [outer[k % outer.shape[0]] for k in range(i0, i1)]
        pts.append(outer[i1 % outer.shape[0]])
        return np.asarray(pts, dtype=float)

    cav_o = _arc(offs[0], offs[1])                 # Lt → Rt inner offset
    east_tip_o = _arc(offs[1], offs[2])            # Rt → SE 45°
    east_stem_o = _arc(offs[2], offs[3])           # SE 45° → kE
    north_o_ccw = _arc(offs[3], offs[4])           # kE → kW
    west_stem_o = _arc(offs[4], offs[5])           # kW → SW 45°
    west_tip_o = _arc(offs[5], offs[6] if offs[6] < outer.shape[0] else outer.shape[0])
    if west_tip_o.shape[0] != n_west_tip + 1:
        west_tip_o = np.concatenate([outer[offs[5]:], outer[0:1]], axis=0)
    if west_tip_o.shape[0] != n_west_tip + 1:
        raise RuntimeError(f"west_tip_o len {west_tip_o.shape[0]} != {n_west_tip+1}")
    north_o = north_o_ccw[::-1].copy()             # kW → kE (+x), for vertical rays
    notes.append("O outer = offset_closed(inner) — wall-normal 1:1 O–H node match")
    notes.append(
        f"OH corners 45°: kSE={kSE} kE={kE} kW={kW} kSW={kSW} "
        f"n_inner={n_cav_n} n_north={n_north_n} n_west={n_west_stem} n_east={n_east_stem}"
    )

    ogrid = build_pitch_ogrid(inner, outer, n_rad, stretch)
    if min_cell_area_2d(ogrid) <= 0:
        ogrid = ogrid[::-1].copy()
    a2 = min_cell_area_2d(ogrid)
    if a2 <= 0:
        raise RuntimeError(f"folded offset-O: min quad area {a2:.3e} m2")
    sm = smooth_ogrid(ogrid, n_iter=12, omega=0.35)
    if min_cell_area_2d(sm) > 0:
        ogrid = sm
        a2 = min_cell_area_2d(ogrid)
    # (O vertical outer snap disabled — thin collar folds)

    # H-O-H: open O at TE. Periodic wrap across dual-arc TE makes same-side faces.
    # Rotate TE (wall max-x) to column 0; keep that column for wake south/north
    # nodes (H still expects the TE outer point); open cuts are the adjacent
    # columns so the wake has finite thickness.
    i_te = int(np.argmax(ogrid[:, 0, 0]))
    if i_te:
        ogrid = np.concatenate([ogrid[i_te:], ogrid[:i_te]], axis=0)
    if ogrid.shape[0] < 8:
        raise RuntimeError("O-grid too coarse to open at TE")
    te_col = ogrid[0].copy()
    ogrid = ogrid[1:, :, :].copy()
    dcut = float(np.linalg.norm(ogrid[0, 0, :] - ogrid[-1, 0, :]))
    if dcut < 1e-7:
        raise RuntimeError(f"TE wake cuts coincident after open (d={dcut:.3e})")
    te_w_old = te_col[0].copy()
    te_o_old = te_col[-1].copy()
    te_w = _project_to_seg(te_w_old, ogrid[0, 0], ogrid[-1, 0])
    te_o = _project_to_seg(te_o_old, ogrid[0, -1], ogrid[-1, -1])
    h_te_wake = te_wake_block(ogrid, te_w, te_o)
    a2 = min_cell_area_2d(ogrid)
    kink_um = float(np.linalg.norm(te_w_old - te_w)) * 1e6
    notes.append(
        f"H-O-H TE wake: open O, TE projected to cut chord "
        f"(unkink {kink_um:.0f} um), cut wall sep={dcut*1e3:.4f} mm"
    )

    # ---- H-blocks: axis-aligned rays, 1:1 O-edge nodes ----
    # south_o: left tip wrap (SW45→Lt, +x) + inner offset + right tip wrap (Rt→SE45).
    south_o = np.concatenate([west_tip_o[:-1], cav_o, east_tip_o[1:]], axis=0)
    # west/east stems only (already 45°-trimmed by kSE/kSW)
    west_s2n = west_stem_o[::-1].copy()   # SW 45° → kW (increasing y)
    east_s2n = east_stem_o               # SE 45° → kE (increasing y)
    pSW = west_s2n[0]
    pNW = west_s2n[-1]
    pSE = east_s2n[0]
    pNE = east_s2n[-1]
    pLt = cav_o[0]
    pRt = cav_o[-1]

    # Soft red joins: H first cell tracks O outer Δn (growth ≤~1.25, neighbor ratio ~≤4).
    dn_o = float(np.mean(np.linalg.norm(ogrid[:, -1, :] - ogrid[:, -2, :], axis=1)))
    dn_o = max(dn_o, 1e-9)
    x_join_w = float(pSW[0])
    x_join_e = float(pSE[0])
    L_in = max(float(x_join_w) - float(x_in), 1e-9)
    # Prefer inlet_stretch but retarget so LE Δx ≈ dn_o (soft LE collar↔passage H).
    r_in = _pack_r_for_d0(L_in, n_in, dn_o, r_max=min(1.25, max(float(r_inlet), 1.02)))
    if r_in <= 1.0 + 1e-12:
        r_in = _inlet_pack_r(n_in, r_inlet)
    xs_w = _xs_pack_hi(x_in, x_join_w, n_in, r_in)
    dx_le = float(xs_w[-1] - xs_w[-2]) if len(xs_w) >= 2 else L_in
    notes.append(
        f"inlet H axial pack toward LE (r={r_in:.3g}, n_inlet={n_in}); "
        f"LE Δx={dx_le*1e6:.2f} um / dn_o={dn_o*1e6:.2f} um ratio={dx_le/dn_o:.2f}"
    )
    # TE collar x — dump starts here (open-O outer / east stem), NOT a full-pitch
    # vertical AABB east wall at x_TE (that hard O→Cartesian cliff fakes Δp).
    xs_poly = [p[0] for p in poly0]
    c_use = max(max(xs_poly) - min(xs_poly), 1e-6) if xs_poly else max(float(x_out) - float(x_in), 1e-6)
    x_te_col = max(
        float(outer[:, 0].max()),
        float(outer_hi[:, 0].max()),
        float(pSE[0]),
        float(pNE[0]),
        float(te_o[0]),
    )
    # Dump first Δx = O outer Δn (soft TE collar↔dump), NOT wall first-cell.
    dx_near = min(max(dn_o, 1e-9), 0.05 * c_use)
    if dump_xs is not None:
        xs_dump = np.asarray(dump_xs, dtype=float).copy()
        xs_dump = xs_dump - float(xs_dump[0]) + float(x_te_col)
        if float(xs_dump[-1]) < float(x_te_col) + 1.0 * c_use - 1e-12:
            xs_dump = dump_xs_1c(
                x_te_col, c_use, dx_near, n_near=10, stretch_max=1.25, L_dump_c=1.0
            )
        notes.append(
            f"dump xs from TE collar n={len(xs_dump)-1} L={(xs_dump[-1]-xs_dump[0])*1e3:.2f} mm "
            f"first_Δx={(xs_dump[1]-xs_dump[0])*1e3:.4f} mm last_Δx={(xs_dump[-1]-xs_dump[-2])*1e3:.3f} mm"
        )
        x_out = float(xs_dump[-1])
    else:
        # Prefer dump_xs_1c packing (0.4c cluster, stretch≤1.25, ≥1.0c) over TE cliff.
        L_c = max(1.0, (float(x_out) - float(x_te_col)) / c_use)
        xs_dump = dump_xs_1c(
            x_te_col, c_use, dx_near, n_near=10, stretch_max=1.25, L_dump_c=L_c
        )
        # Optional dump_rx only lengthens / retargets last cell if caller asked.
        if dump_rx is not None and float(dump_rx) > 1.0 + 1e-12 and int(n_out_x) >= 2:
            # Keep dump_xs_1c near-TE; if shorter than requested x_out, rebuild with L_c.
            if float(x_out) > float(xs_dump[-1]) + 1e-9:
                L_c = max(1.0, (float(x_out) - float(x_te_col)) / c_use)
                xs_dump = dump_xs_1c(
                    x_te_col, c_use, dx_near, n_near=10, stretch_max=min(1.25, float(dump_rx)),
                    L_dump_c=L_c,
                )
        x_out = float(xs_dump[-1])
        notes.append(
            f"DUMP 1.0c from TE collar: x_TE_col={x_te_col:.6g} x_out={x_out:.6g} "
            f"n_dump={len(xs_dump)-1} L/c={(x_out-x_te_col)/c_use:.3g} "
            f"first_Δx={(xs_dump[1]-xs_dump[0]):.3e} (no TE-plane AABB east wall)"
        )

    y_sw = float(pSW[1])
    y_se = float(pSE[1])
    y_nw = float(pNW[1])
    y_ne = float(pNE[1])
    # Mid cyclic x = north O x's so N/S cyclics pair 1:1 with no fan.
    # Graded buffer: pack passage H toward O so first H cell ≈ dn_o (soft red joins).
    gap_s = float(max(float(np.min(south_o[:, 1])) - float(y_bot), 1e-9))
    gap_n = float(max(float(y_top) - float(np.max(north_o[:, 1])), 1e-9))
    # Repair: do NOT bump n_fill_h (changes H–H corner topology → leftover faces).
    # Pack within existing n_fill so first H cell tracks dn_o as far as r≤1.25 allows.
    r_south = _pack_r_for_d0(gap_s, n_fill_h, dn_o, r_max=1.25)
    # North pack_r>1 leaves 80 unclassified faces (nw TFI/smooth drifts from
    # north west edge). Keep r_north=1 until nw edge-pin is proven; south is OK.
    r_north = 1.0
    dy_s = gap_s * (r_south - 1.0) / (r_south ** max(n_fill_h, 1) - 1.0) if r_south > 1.0 + 1e-12 else gap_s / max(n_fill_h, 1)
    dy_n = gap_n / max(n_fill_h, 1)
    notes.append(
        f"soft H↔O joins: dn_o={dn_o*1e6:.2f} um n_fill={n_fill_h} "
        f"r_s={r_south:.3g} r_n={r_north:.3g}(pinned) "
        f"firstΔy_s/dn_o={dy_s/dn_o:.2f} firstΔy_n/dn_o={dy_n/dn_o:.2f} "
        f"gap_s={gap_s*1e3:.3f} mm gap_n={gap_n*1e3:.3f} mm"
    )
    h_west = _pos_block(
        _ray_block_horizontal(
            x_in, west_s2n[:, 0], west_s2n[:, 1], n_in, pack_r=r_in, dense_at="hi"
        ),
        "west",
        keep_edges=True,
    )
    h_south = _pos_block(
        _ray_block_vertical(
            y_bot, south_o[:, 1], south_o[:, 0], n_fill_h, pack_r=r_south, dense_at="hi"
        ),
        "south",
        keep_edges=True,
    )
    xs_s = south_o[:, 0]
    north_top = np.column_stack([xs_s, np.full(xs_s.shape[0], y_top)])
    if north_o.shape[0] != xs_s.shape[0]:
        north_o = _interp_arc_xs(
            north_o,
            np.linspace(float(north_o[0, 0]), float(north_o[-1, 0]), int(xs_s.shape[0])),
        )
    h_n_raw = np.zeros((xs_s.shape[0], n_fill_h + 1, 2), dtype=float)
    for _i in range(xs_s.shape[0]):
        h_n_raw[_i] = _lin_pack_start(north_o[_i], north_top[_i], n_fill_h, r_north)
    # Pin edges before/after smooth — packed O interface must stay conformal
    # (smooth drifting edges → duplicate boundary faces → leftover≠0).
    _e0 = h_n_raw[:, 0, :].copy()
    _e1 = h_n_raw[:, -1, :].copy()
    _ew = h_n_raw[0, :, :].copy()
    _ee = h_n_raw[-1, :, :].copy()
    sm = smooth_rect_block(h_n_raw, n_iter=80, omega=0.45)
    if min_cell_area_2d_rect(sm) > 0:
        h_n_raw = sm
        h_n_raw[:, 0, :] = _e0
        h_n_raw[:, -1, :] = _e1
        h_n_raw[0, :, :] = _ew
        h_n_raw[-1, :, :] = _ee
        # re-pin corners after side restores
        h_n_raw[0, 0, :] = _e0[0]
        h_n_raw[-1, 0, :] = _e0[-1]
        h_n_raw[0, -1, :] = _e1[0]
        h_n_raw[-1, -1, :] = _e1[-1]
    h_north = _pos_block(h_n_raw, "north", keep_edges=True)

    west_s = h_west[:, 0, :]
    west_n = h_west[:, -1, :]
    south_w = h_south[0, :, :]
    south_e = h_south[-1, :, :]
    north_w = h_north[0, :, :]
    north_e = h_north[-1, :, :]

    h_sw = _pos_block(
        tfi_block(
            np.column_stack([west_s[:, 0], np.full(west_s.shape[0], y_bot)]),
            west_s,
            np.column_stack([np.full(south_w.shape[0], x_in), south_w[:, 1]]),
            south_w,
        ),
        "sw",
        keep_edges=True,
    )
    nw_north = np.column_stack([west_s[:, 0], np.full(west_s.shape[0], y_top)])
    nw_east = _lin(west_n[-1], nw_north[-1], north_w.shape[0] - 1)
    nw_west = _lin(west_n[0], nw_north[0], north_w.shape[0] - 1)
    h_nw_raw = tfi_block(west_n, nw_north, nw_west, nw_east)
    _n0 = h_nw_raw[:, 0, :].copy(); _n1 = h_nw_raw[:, -1, :].copy()
    _nw = h_nw_raw[0, :, :].copy(); _ne = h_nw_raw[-1, :, :].copy()
    sm = smooth_rect_block(h_nw_raw, n_iter=80, omega=0.45)
    if min_cell_area_2d_rect(sm) > 0:
        h_nw_raw = sm
        h_nw_raw[:, 0, :] = _n0; h_nw_raw[:, -1, :] = _n1
        h_nw_raw[0, :, :] = _nw; h_nw_raw[-1, :, :] = _ne
        h_nw_raw[0, 0, :] = _n0[0]; h_nw_raw[-1, 0, :] = _n0[-1]
        h_nw_raw[0, -1, :] = _n1[0]; h_nw_raw[-1, -1, :] = _n1[-1]
    h_nw = _pos_block(h_nw_raw, "nw", keep_edges=True)

    # Dump west = south H east + O east stem + north H east (collar silhouette).
    # Removes h_east→x_cart AABB cliff. dump_xs_1c packing from TE collar.
    # (Full-pitch dump-west can still look bright on preview when stem≈vertical.)
    south_e = south_e.copy()
    north_e = north_e.copy()
    south_e[-1] = east_s2n[0]
    north_e[0] = east_s2n[-1]
    west_dump = _join_polylines(south_e, east_s2n, north_e)
    h_dump = _pos_block(_dump_block_from_west(west_dump, xs_dump, x_out), "dump", keep_edges=True)
    snap = [(te_w_old, te_w), (te_o_old, te_o)]
    h_rest = [
        _snap_points(hb, snap)
        for hb in [h_west, h_south, h_north, h_sw, h_nw, h_dump]
    ]
    h_blocks = [h_te_wake, *h_rest]
    notes.append(
        "H-blocks: vertical rays cavity+south and back→y_top (pack→O, growth≤1.25); "
        "inlet stems horizontal (pack→LE); dump from TE collar silhouette with "
        "dump_xs_1c first_Δx≈dn_o. Soft red joins — no chalk-line size cliffs. "
        "No h_east→x_cart AABB east wall / Cartesian cliff at TE."
    )
    notes.append("H TE nodes snapped to wake chord (unkinked).")
    notes.append("Cyclic x-nodes are shared top/bottom so 3-pitch stacking is conformal.")
    notes.append("NOT subsetMesh stairs. NOT AABB morph across the cavity. NOT Gmsh.")
    notes.append(
        f"TE collar dump: x_TE_col={x_te_col:.6g} x_out={float(x_out):.6g} n_dump={len(xs_dump)-1}"
    )
    return ogrid, h_blocks, a2, notes



def _shift_poly(poly: list[tuple[float, float]], dy: float) -> list[tuple[float, float]]:
    return [(p[0], p[1] + dy) for p in poly]


def _aabb_xy(pts: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(pts[:, 0].min()),
        float(pts[:, 0].max()),
        float(pts[:, 1].min()),
        float(pts[:, 1].max()),
    )


def _aabb_overlap(a, b, eps: float = 1e-9) -> bool:
    return not (
        a[1] <= b[0] + eps
        or b[1] <= a[0] + eps
        or a[3] <= b[2] + eps
        or b[3] <= a[2] + eps
    )


def point_in_closed_poly(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    """Even-odd PIP. Boundary counts as inside."""
    pts = poly[:-1] if poly and poly[0] == poly[-1] else list(poly)
    n = len(pts)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if abs(yi - y) < 1e-16 and abs(xi - x) < 1e-16:
            return True
        if (yi > y) != (yj > y):
            xing = (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi
            if abs(xing - x) < 1e-16:
                return True
            if xing > x:
                inside = not inside
        j = i
    return inside


def _ogrid_from_offset(
    poly0: list[tuple[float, float]],
    d_o: float,
    n_in: int,
    n_out: int,
    n_cyc: int,
    n_rad: int,
    stretch: float,
) -> tuple[np.ndarray, np.ndarray, float, list[str]]:
    """Offset-closed O-collar. Same hybrid/offset functions as body_fitted, no AABB morph.

    Outer ring is the wall-normal offset (not a pitch rectangle). 4-side counts
    match today's S/N=n_cyclic so a neighbor passage can TFI on those nodes.
    """
    notes: list[str] = []
    n_i = 2 * n_cyc + n_in + n_out
    inner = resample_closed(poly0, n_i)
    if polygon_signed_area([(float(x), float(y)) for x, y in inner]) < 0:
        inner = inner[::-1].copy()
    outer = offset_closed(inner, d_o, n_smooth=12)
    ogrid = build_pitch_ogrid(inner, outer, n_rad, stretch)
    if min_cell_area_2d(ogrid) <= 0:
        ogrid = ogrid[::-1].copy()
    a2 = min_cell_area_2d(ogrid)
    if a2 <= 0:
        og2 = build_hybrid_aabb_ogrid(inner, outer, n_rad, stretch, 0.90 * d_o)
        if min_cell_area_2d(og2) <= 0:
            og2 = og2[::-1].copy()
        if min_cell_area_2d(og2) > 0:
            ogrid = og2
            a2 = min_cell_area_2d(ogrid)
            notes.append("cassette O: hybrid AABB-morph fallback on offset ring")
    sm = smooth_ogrid(ogrid, n_iter=8, omega=0.35)
    if min_cell_area_2d(sm) > 0:
        ogrid = sm
        a2 = min_cell_area_2d(ogrid)
    if a2 <= 0:
        raise RuntimeError(f"folded cassette O-grid: min quad area {a2:.3e} m2")
    notes.append(f"cassette O-collar d_o={d_o:.3g} m n_i={ogrid.shape[0]} n_rad={n_rad}")
    return ogrid, ogrid[:, -1, :].copy(), a2, notes


def _ring_ps_ss(ring: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    poly = [(float(x), float(y)) for x, y in ring] + [(float(ring[0, 0]), float(ring[0, 1]))]
    ps, ss, _, _ = split_ps_ss(poly)
    return np.array(ps, dtype=float), np.array(ss, dtype=float)


def _passage_tfi(south: np.ndarray, north: np.ndarray, n_st: int, n_span: int, name: str) -> np.ndarray:
    s = resample_open_arclength(south, n_st + 1)
    n = resample_open_arclength(north, n_st + 1)
    west = _lin(s[0], n[0], n_span)
    east = _lin(s[-1], n[-1], n_span)
    return _pos_block(tfi_block(s, n, west, east), name)


def build_cassette_oh(
    poly0: list[tuple[float, float]],
    *,
    pitch: float,
    n_blades: int,
    x_in: float,
    x_out: float,
    n_in: int,
    n_out: int,
    n_cyc: int,
    n_rad: int,
    n_fill: int,
    n_out_x: int,
    stretch: float,
    d_o: float,
    g_min: float,
    chord_m: float | None = None,
    dump_xs: np.ndarray | None = None,
) -> dict:
    """Three physical C's in one polyMesh. Fluid outside every C.

    Pitch-matching translational cyclics on bottom/top (separation = n_blades*pitch,
    matching nFaces via shared axial nodes). Full-height inlet/outlet ducts span the
    three-blade stack — no y-stubs, no stepped L/R lid cutoffs.
    """
    notes: list[str] = []
    blades = [_shift_poly(poly0, k * pitch) for k in range(n_blades)]
    Y = float(n_blades) * float(pitch)
    d_use = min(float(d_o), 0.28 * max(float(g_min), 1e-6), 0.00045)
    if d_use < float(d_o) - 1e-16:
        notes.append(f"d_o {d_o:.3g} → {d_use:.3g} m so collars clear Gate-0 gap")
    ogrids = []
    outers = []
    aabbs = []
    a2_min = 1e99
    for k, pk in enumerate(blades):
        og, outer, a2, onotes = _ogrid_from_offset(
            pk, d_use, n_in, n_out, n_cyc, n_rad, stretch
        )
        ogrids.append(og)
        outers.append(outer)
        aabbs.append(_aabb_xy(outer))
        a2_min = min(a2_min, a2)
        notes.extend(onotes)
    for k in range(n_blades - 1):
        if _aabb_overlap(aabbs[k], aabbs[k + 1]):
            notes.append(
                f"O AABB overlap blade{k}/blade{k+1} "
                f"{aabbs[k]} vs {aabbs[k+1]} — nested C; H is arc-length passage TFI "
                "not Cartesian-minus-AABB"
            )
    all_y0 = min(a[2] for a in aabbs)
    all_y1 = max(a[3] for a in aabbs)
    margin = max(0.5 * d_use, 0.15e-3)
    y_bot = float(all_y0) - margin
    y_top = y_bot + Y
    if float(all_y1) + 1e-9 >= y_top - margin:
        y_bot = 0.5 * (float(all_y0) + float(all_y1)) - 0.5 * Y
        y_top = y_bot + Y
        notes.append(
            f"cyclic strip centered on O stack; clearance bot={all_y0 - y_bot:.3e} "
            f"top={y_top - all_y1:.3e} m"
        )
    if float(all_y0) <= y_bot + 1e-9 or float(all_y1) >= y_top - 1e-9:
        raise RuntimeError(
            f"cassette metal/O does not fit in n_blades*pitch cyclic strip "
            f"Y={Y:.6g} y_bot={y_bot:.6g} y_top={y_top:.6g} "
            f"O_y=[{all_y0:.6g},{all_y1:.6g}]"
        )
    x_te = max(max(p[0] for p in bp) for bp in blades)
    c_use = float(chord_m) if chord_m not in (None, "") else max(
        x_te - min(min(p[0] for p in bp) for bp in blades), 1e-6
    )
    x_out = max(float(x_out), float(x_te) + 1.0 * c_use)
    if dump_xs is not None:
        _dump_xs = np.asarray(dump_xs, dtype=float).copy()
        _dump_xs = _dump_xs - float(_dump_xs[0]) + float(x_te)
        _dump_xs[-1] = x_out
    else:
        dx_te = max(
            float(d_use) * (float(stretch) - 1.0) / max(float(stretch) ** max(int(n_rad), 1) - 1.0, 1e-12),
            1e-7,
        )
        dx_te = min(dx_te, 0.05 * c_use)
        _dump_xs = dump_xs_1c(x_te, c_use, dx_te, n_near=10, stretch_max=1.25, L_dump_c=1.0)
        x_out = float(_dump_xs[-1])
    notes.append(
        f"DUMP 1.0c: x_TE={x_te:.6g} x_out={x_out:.6g} n_dump={len(_dump_xs)-1} "
        f"L_dump/c={(x_out-x_te)/c_use:.3g} first_Δx={(_dump_xs[1]-_dump_xs[0]):.3e}"
    )
    n_st = max(int(n_cyc) * 2, 32, int(n_fill) // 2)
    xs_le = np.linspace(float(x_in), float(x_te), n_st + 1)
    xs_cyc = np.unique(np.concatenate([xs_le[:-1], np.asarray(_dump_xs, dtype=float)]))
    xs_cyc[0] = float(x_in)
    xs_cyc[-1] = float(x_out)
    P_bot = np.column_stack([xs_cyc, np.full(xs_cyc.shape[0], y_bot)])
    P_top = np.column_stack([xs_cyc, np.full(xs_cyc.shape[0], y_top)])
    n_cyc_faces = int(xs_cyc.shape[0] - 1)

    h_blocks: list[np.ndarray] = []
    n_span = max(int(n_fill), 10)
    n_span_io = max(n_span, 12)

    for k in range(n_blades - 1):
        _ps_a, ss_a = _ring_ps_ss(outers[k])
        ps_b, _ss_b = _ring_ps_ss(outers[k + 1])
        h_blocks.append(_passage_tfi(ss_a, ps_b, n_st, n_span, f"pass{k}"))
        notes.append(f"passage TFI blade{k} SS-offset vs blade{k+1} PS-offset (arc length)")
        s0 = resample_open_arclength(ss_a, n_st + 1)
        n0 = resample_open_arclength(ps_b, n_st + 1)
        s_head = _lin_pack_end((x_in, float(s0[0, 1])), s0[0], n_in, max(stretch, 1.0))
        n_head = _lin_pack_end((x_in, float(n0[0, 1])), n0[0], n_in, max(stretch, 1.0))
        west_in = _lin(s_head[0], n_head[0], n_span_io)
        east_in = _lin(s0[0], n0[0], n_span_io)
        h_blocks.append(_pos_block(tfi_block(s_head, n_head, west_in, east_in), f"in{k}"))
        ys_s = float(s0[-1, 1])
        ys_n = float(n0[-1, 1])
        s_tail = np.column_stack([_dump_xs, np.full(_dump_xs.shape[0], ys_s)])
        n_tail = np.column_stack([_dump_xs, np.full(_dump_xs.shape[0], ys_n)])
        west_out = _lin(s0[-1], n0[-1], n_span_io)
        east_out = _lin(s_tail[-1], n_tail[-1], n_span_io)
        h_blocks.append(_pos_block(tfi_block(s_tail, n_tail, west_out, east_out), f"out{k}"))

    ps0, _ss0 = _ring_ps_ss(outers[0])
    _psN, ssN = _ring_ps_ss(outers[-1])
    n_floor = max(n_span // 2, 6)
    # REPAIR (1/1): mid-gap translational cyclics — P_top = P_bot+(0,Y).
    # y=const floor↔PS TFI folds on nested C (min_area<0). Mid-gap matches nFaces.
    def _y_at_x_ring(ring: np.ndarray, x: float, which: str) -> float:
        rx, ry = ring[:, 0], ring[:, 1]
        ys = []
        for i in range(len(rx) - 1):
            x0, x1 = float(rx[i]), float(rx[i + 1])
            if (x0 - x) * (x1 - x) > 0 and abs(x0 - x) > 1e-14 and abs(x1 - x) > 1e-14:
                continue
            if abs(x1 - x0) < 1e-16:
                if abs(x0 - x) < 1e-12:
                    ys.append(float(ry[i]))
                continue
            t = (x - x0) / (x1 - x0)
            if -1e-9 <= t <= 1.0 + 1e-9:
                ys.append(float(ry[i] + t * (ry[i + 1] - ry[i])))
        if not ys:
            j = int(np.argmin(np.abs(rx - x)))
            return float(ry[j])
        return float(min(ys) if which == "min" else max(ys))

    o0 = outers[0]
    img = outers[-1].copy()
    img[:, 1] = img[:, 1] - Y  # periodic image of last blade below blade0
    x_le_m = float(min(o0[:, 0].min(), img[:, 0].min()))
    x_te_m = float(max(o0[:, 0].max(), img[:, 0].max()))
    xs_m = np.linspace(float(x_in), float(x_out), n_cyc_faces + 1)
    P_bot = np.zeros((xs_m.shape[0], 2))
    for i, xv in enumerate(xs_m):
        if x_le_m - 1e-9 <= xv <= x_te_m + 1e-9:
            y = 0.5 * (_y_at_x_ring(o0, float(xv), "min") + _y_at_x_ring(img, float(xv), "max"))
        elif xv < x_le_m:
            y = 0.5 * (_y_at_x_ring(o0, x_le_m, "min") + _y_at_x_ring(img, x_le_m, "max"))
        else:
            y = 0.5 * (_y_at_x_ring(o0, x_te_m, "min") + _y_at_x_ring(img, x_te_m, "max"))
        # Keep mid-gap inside the cyclic strip
        y = min(max(float(y), y_bot + 1e-6), y_top - Y + 1e-6)
        P_bot[i] = (float(xv), float(y))
    P_top = P_bot.copy()
    P_top[:, 1] = P_bot[:, 1] + Y
    # Floor: mid-gap → PS0 offset (same count, arc-resample PS to n)
    ps0r = resample_open_arclength(ps0, P_bot.shape[0])
    # Align ps0r x toward P_bot x by rebuilding north as envelope at P_bot x
    north_f = np.column_stack([P_bot[:, 0], [_y_at_x_ring(o0, float(x), "min") for x in P_bot[:, 0]]])
    h_blocks.append(_passage_tfi(P_bot, north_f, int(P_bot.shape[0] - 1), n_floor, "floor"))
    south_l = np.column_stack([P_top[:, 0], [_y_at_x_ring(outers[-1], float(x), "max") for x in P_top[:, 0]]])
    h_blocks.append(_passage_tfi(south_l, P_top, int(P_top.shape[0] - 1), n_floor, "lid"))
    n_cyc_faces = int(P_bot.shape[0] - 1)
    notes.append(f"REPAIR: mid-gap cyclics nFaces={n_cyc_faces} Y={Y:.6g}")

    y_in_lo = float(resample_open_arclength(ps0, n_st + 1)[0, 1])
    y_in_hi = float(resample_open_arclength(ssN, n_st + 1)[0, 1])
    x_le_o = min(float(a[0]) for a in aabbs)
    x_le_o = max(x_le_o, float(x_in) + 1e-6)
    if y_in_lo - y_bot > 2e-5:
        n_ysw = max(n_floor, 6)
        west_sw = _lin((x_in, y_bot), (x_in, y_in_lo), n_ysw)
        east_sw = _lin((x_le_o, y_bot), (x_le_o, y_in_lo), n_ysw)
        south_sw = _lin(west_sw[0], east_sw[0], n_in)
        north_sw = _lin(west_sw[-1], east_sw[-1], n_in)
        try:
            h_blocks.append(_pos_block(tfi_block(south_sw, north_sw, west_sw, east_sw), "in_sw"))
            notes.append("full-height inlet: SW corner duct (no y-stub)")
        except Exception as exc:
            notes.append(f"in_sw skipped: {exc}")
    if y_top - y_in_hi > 2e-5:
        n_ynw = max(n_floor, 6)
        west_nw = _lin((x_in, y_in_hi), (x_in, y_top), n_ynw)
        east_nw = _lin((x_le_o, y_in_hi), (x_le_o, y_top), n_ynw)
        south_nw = _lin(west_nw[0], east_nw[0], n_in)
        north_nw = _lin(west_nw[-1], east_nw[-1], n_in)
        try:
            h_blocks.append(_pos_block(tfi_block(south_nw, north_nw, west_nw, east_nw), "in_nw"))
            notes.append("full-height inlet: NW corner duct (no y-stub)")
        except Exception as exc:
            notes.append(f"in_nw skipped: {exc}")

    y_out_lo = float(resample_open_arclength(ps0, n_st + 1)[-1, 1])
    y_out_hi = float(resample_open_arclength(ssN, n_st + 1)[-1, 1])
    if y_out_lo - y_bot > 2e-5:
        n_yse = max(n_floor, 6)
        s_se = np.column_stack([_dump_xs, np.full(_dump_xs.shape[0], y_bot)])
        n_se = np.column_stack([_dump_xs, np.full(_dump_xs.shape[0], y_out_lo)])
        west_se = _lin(s_se[0], n_se[0], n_yse)
        east_se = _lin(s_se[-1], n_se[-1], n_yse)
        try:
            h_blocks.append(_pos_block(tfi_block(s_se, n_se, west_se, east_se), "out_se"))
            notes.append("full-height outlet: SE corner dump (no y-stub)")
        except Exception as exc:
            notes.append(f"out_se skipped: {exc}")
    if y_top - y_out_hi > 2e-5:
        n_yne = max(n_floor, 6)
        s_ne = np.column_stack([_dump_xs, np.full(_dump_xs.shape[0], y_out_hi)])
        n_ne = np.column_stack([_dump_xs, np.full(_dump_xs.shape[0], y_top)])
        west_ne = _lin(s_ne[0], n_ne[0], n_yne)
        east_ne = _lin(s_ne[-1], n_ne[-1], n_yne)
        try:
            h_blocks.append(_pos_block(tfi_block(s_ne, n_ne, west_ne, east_ne), "out_ne"))
            notes.append("full-height outlet: NE corner dump (no y-stub)")
        except Exception as exc:
            notes.append(f"out_ne skipped: {exc}")

    spl = _down_u_splits(outers[-1])
    if spl is not None:
        try:
            n_stem = max(int(n_in), int(n_cyc), 12)
            inner_idx = spl["inner"]
            kNW, kNE = spl["kNW"], spl["kNE"]
            off_hi = outers[-1]
            cav_w = _resample_xy(inner_idx[: kNW + 1], off_hi, n_stem)
            cav_n = _resample_xy(inner_idx[kNW : kNE + 1], off_hi, n_cyc)
            cav_e = _resample_xy(inner_idx[kNE:], off_hi, n_stem)[::-1].copy()
            pLt, pRt = cav_w[0], cav_e[0]
            mouth = _lin(pLt, pRt, n_cyc)
            h_cav = _pos_block(tfi_block(mouth, cav_n, cav_w, cav_e), "cavity_top")
            h_blocks.append(h_cav)
            notes.append("cavity TFI on top C only (lower cups nest the neighbor)")
        except Exception as exc:
            notes.append(f"top cavity TFI skipped: {exc}")

    kept = []
    for hb in h_blocks:
        ni, nj, _ = hb.shape
        hit = False
        for i in range(ni - 1):
            for j in range(nj - 1):
                c = 0.25 * (hb[i, j] + hb[i + 1, j] + hb[i + 1, j + 1] + hb[i, j + 1])
                for bp in blades:
                    if point_in_closed_poly(float(c[0]), float(c[1]), bp):
                        hit = True
                        break
                if hit:
                    break
            if hit:
                break
        if hit:
            notes.append(f"dropped H-block shape={hb.shape} (PIP inside C)")
        else:
            kept.append(hb)
    h_blocks = kept
    first_cell = float(np.mean(np.linalg.norm(ogrids[0][:, 1, :] - ogrids[0][:, 0, :], axis=1)))
    notes.append(
        "mesh_kind cassette_OH: 3 closed metals, fluid outside every C, "
        f"bottom/top translational cyclic Y={Y:.6g} m (matching nFaces via shared xs)."
    )
    notes.append("NOT Gmsh. NOT y(x) passage_OH. NOT subsetMesh stairs. NOT lid/floor WALL.")
    notes.append(f"cyclic n_axial_faces={n_cyc_faces} inlet/outlet full-height ducts (corner fills). repair=vertical-or-midgap-cyclics")
    return {
        "ogrids": ogrids,
        "h_blocks": h_blocks,
        "blades": blades,
        "y_bot": y_bot,
        "y_top": y_top,
        "d_o": d_use,
        "a2": float(a2_min),
        "first_cell": first_cell,
        "notes": notes,
        "aabbs": aabbs,
        "n_i": int(ogrids[0].shape[0]),
        "x_in": float(x_in),
        "x_te": float(x_te),
        "x_out": float(x_out),
        "n_dump": int(len(_dump_xs) - 1),
        "dump_xs": _dump_xs,
        "cyclic_Y": float(Y),
        "n_cyc_faces": n_cyc_faces,
        "P_bot": P_bot,
        "P_top": P_top,
    }



def _tri_area2(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])


def _hybrid_size_m(
    x: float,
    y: float,
    *,
    le: tuple[float, float],
    pass_xy: tuple[float, float],
    h_le: float,
    h_pass: float,
    h_far: float,
    growth: float,
    chord: float,
    x_dense_c: float = 0.25,
    hole_closed: list[tuple[float, float]] | None = None,
) -> float:
    """Soft size field with axial inlet ramp + LE/passage attractors.

    Far upstream of LE (x <= le_x - x_dense_c*chord) stays sparse at h_far.
    From that station → LE, size blends smoothly toward h_le; LE/passage
    attractors and O-outer wall distance still force fine cells where needed.
    """
    g = max(float(growth), 1.0 + 1e-9)
    hl = max(float(h_le), 1e-9)
    hp = max(float(h_pass), 1e-9)
    hf = max(float(h_far), hp)
    c = max(float(chord), 1e-9)
    le_x = float(le[0])
    le_y = float(le[1])
    xd_c = max(float(x_dense_c), 0.0)
    x_dense = le_x - xd_c * c

    d_le = math.hypot(x - le_x, y - le_y)
    d_p = math.hypot(x - float(pass_xy[0]), y - float(pass_xy[1]))
    # LE stays tight; passage uses a wider length scale so mid-gap stays near h_pass
    # without requiring a tiny isotropic ball (old 4*h_pass cap had densified everything).
    L_pass = max(3.0 * hp, 0.08 * c)
    h_attr = min(hl * (g ** (d_le / hl)), hp * (g ** (d_p / L_pass)), hf)

    # Axial ramp (Laser mark ~0.25c upstream of LE): sparse | gradient | dense.
    if x <= x_dense:
        h_ax = hf
    elif x < le_x:
        t = (x - x_dense) / max(le_x - x_dense, 1e-15)
        t = max(0.0, min(1.0, t))
        t = t * t * (3.0 - 2.0 * t)  # smoothstep
        h_ax = hf + (hl - hf) * t
    else:
        h_ax = hf

    h = min(h_ax, h_attr)
    # Hole/O-outer spacing comes from constrained edges + LE attractor near the
    # collar; an isotropic wall term fights the axial inlet ramp (Laser mark).
    _ = hole_closed
    return float(max(h, 0.45 * hl))


def _min_dist_to_poly(x: float, y: float, poly: list[tuple[float, float]]) -> float:
    """Min distance from point to polygon boundary edges."""
    best = 1e300
    n = len(poly)
    if n < 2:
        return best
    for i in range(n - 1):
        x0, y0 = poly[i]
        x1, y1 = poly[i + 1]
        dx, dy = x1 - x0, y1 - y0
        L2 = dx * dx + dy * dy
        if L2 < 1e-30:
            d = math.hypot(x - x0, y - y0)
        else:
            t = max(0.0, min(1.0, ((x - x0) * dx + (y - y0) * dy) / L2))
            d = math.hypot(x - (x0 + t * dx), y - (y0 + t * dy))
        if d < best:
            best = d
    return float(best)


def _seed_size_field(
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    size_fn,
    hole_closed: list[tuple[float, float]],
    *,
    max_depth: int = 18,
    max_pts: int = 60000,
) -> list[list[float]]:
    """Quadtree seeds outside the O+wake hole; spacing tracks local size field."""
    out: list[list[float]] = []

    def rec(xa, xb, ya, yb, depth: int) -> None:
        if len(out) >= max_pts:
            return
        cx, cy = 0.5 * (xa + xb), 0.5 * (ya + yb)
        in_hole = point_in_closed_poly(cx, cy, hole_closed)
        hx = xb - xa
        hy = yb - ya
        if in_hole:
            if depth < max_depth and max(hx, hy) > 2e-5:
                xm, ym = cx, cy
                rec(xa, xm, ya, ym, depth + 1)
                rec(xm, xb, ya, ym, depth + 1)
                rec(xa, xm, ym, yb, depth + 1)
                rec(xm, xb, ym, yb, depth + 1)
            return
        h = float(size_fn(cx, cy))
        if (hx <= 1.35 * h and hy <= 1.35 * h) or depth >= max_depth:
            # Keep a soft halo outside the O-hole so interface tris are not cliffs.
            if _min_dist_to_poly(cx, cy, hole_closed) < 0.55 * h:
                return
            out.append([cx, cy])
            return
        xm, ym = cx, cy
        rec(xa, xm, ya, ym, depth + 1)
        rec(xm, xb, ya, ym, depth + 1)
        rec(xa, xm, ym, yb, depth + 1)
        rec(xm, xb, ym, yb, depth + 1)

    rec(x0, x1, y0, y1, 0)
    # Thin ONLY coarse far-field ghosts. Fine LE/passage seeds keep full density.
    # Quadtree children from a fine parent otherwise litter the sparse inlet at <<h.
    if not out:
        return out
    # Infer far scale from the largest requested sizes in this cloud.
    hs_all = [float(size_fn(p[0], p[1])) for p in out]
    h_cap = max(hs_all) if hs_all else 1e-3
    coarse_cut = 0.35 * h_cap
    fine = [p for p, h in zip(out, hs_all) if h < coarse_cut]
    coarse = [p for p, h in zip(out, hs_all) if h >= coarse_cut]
    scored = sorted(coarse, key=lambda p: -float(size_fn(p[0], p[1])))
    kept_c: list[list[float]] = []
    bins: dict[tuple[int, int], list[int]] = {}
    for p in scored:
        hp = float(size_fn(p[0], p[1]))
        min_d = 0.60 * hp
        inv = 1.0 / max(min_d, 1e-9)
        ix, iy = int(math.floor(p[0] * inv)), int(math.floor(p[1] * inv))
        ok = True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for qi in bins.get((ix + dx, iy + dy), ()):
                    q = kept_c[qi]
                    if math.hypot(p[0] - q[0], p[1] - q[1]) < min_d:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            bins.setdefault((ix, iy), []).append(len(kept_c))
            kept_c.append(p)
    return fine + kept_c


def _wake_outer_row(wake: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Wake row whose endpoints match O-outer cut nodes A,B."""
    eps = 1e-8
    for j in (0, -1):
        row = np.asarray(wake[:, j, :], dtype=float)
        d00 = float(np.linalg.norm(row[0] - a))
        d01 = float(np.linalg.norm(row[0] - b))
        d10 = float(np.linalg.norm(row[-1] - a))
        d11 = float(np.linalg.norm(row[-1] - b))
        if d00 < eps and d11 < eps:
            return row
        if d01 < eps and d10 < eps:
            return row
    raise RuntimeError("hybrid: TE wake outer row does not match O-outer cuts")


def _hole_loop_o_wake(ogrid: np.ndarray, wake: np.ndarray) -> np.ndarray:
    """Closed O-outer + wake-outer loop (hole = metal + O collar + TE wake)."""
    oo = np.asarray(ogrid[:, -1, :], dtype=float)
    a, b = oo[0], oo[-1]
    wrow = _wake_outer_row(wake, a, b)
    eps = 1e-8
    if float(np.linalg.norm(wrow[0] - b)) < eps and float(np.linalg.norm(wrow[-1] - a)) < eps:
        mid = wrow[1:-1]
    elif float(np.linalg.norm(wrow[0] - a)) < eps and float(np.linalg.norm(wrow[-1] - b)) < eps:
        mid = wrow[-2:0:-1]
    else:
        raise RuntimeError("hybrid: cannot close O-outer with wake outer")
    if mid.size:
        return np.vstack([oo, mid])
    return oo.copy()


def _build_passage_triangles(
    *,
    ogrid: np.ndarray,
    wake: np.ndarray,
    dump: np.ndarray,
    poly0: list[tuple[float, float]],
    x_in: float,
    y_bot: float,
    y_top: float,
    h_le: float,
    h_pass: float,
    h_far: float,
    growth: float,
    x_dense_c: float = 0.25,
) -> tuple[np.ndarray, dict[str, float], list[str]]:
    """Constrained Delaunay triangles west of dump, outside O+wake. No Gmsh."""
    try:
        import triangle as tr
    except ImportError as e:
        raise RuntimeError(
            "hybrid_OH_tri requires the 'triangle' package in .venv "
            "(pip install triangle). Gmsh is not used."
        ) from e

    notes: list[str] = []
    hole = _hole_loop_o_wake(ogrid, wake)
    hole_closed = [(float(x), float(y)) for x, y in hole] + [
        (float(hole[0, 0]), float(hole[0, 1]))
    ]
    x_cart = float(dump[0, 0, 0])
    west = np.asarray(dump[0, :, :], dtype=float)
    if float(west[0, 1]) > float(west[-1, 1]):
        west = west[::-1].copy()

    le = min(poly0, key=lambda p: p[0])
    xs_m = [p[0] for p in poly0]
    chord_x = max(max(xs_m) - min(xs_m), 1e-6)
    x_dense = float(le[0]) - float(x_dense_c) * chord_x
    pass_xy = (0.5 * (min(xs_m) + max(xs_m)), 0.0)

    def size_fn(x: float, y: float) -> float:
        return _hybrid_size_m(
            x,
            y,
            le=le,
            pass_xy=pass_xy,
            h_le=h_le,
            h_pass=h_pass,
            h_far=h_far,
            growth=growth,
            chord=chord_x,
            x_dense_c=x_dense_c,
            hole_closed=hole_closed,
        )

    # Matched cyclic x-nodes (true one-pitch cyclic).
    xs_c: list[float] = []
    x = float(x_in)
    guard = 0
    while x < x_cart - 1e-12 and guard < 100000:
        xs_c.append(x)
        x += max(size_fn(x, y_bot), 0.5 * h_le)
        guard += 1
    if not xs_c or abs(xs_c[-1] - x_cart) > 1e-12:
        xs_c.append(x_cart)
    else:
        xs_c[-1] = x_cart
    bottom = [(float(xx), float(y_bot)) for xx in xs_c]
    top = [(float(xx), float(y_top)) for xx in xs_c]
    n_in_s = max(4, int(math.ceil((y_top - y_bot) / max(h_far, 1e-9))))
    inlet = [(float(x_in), float(yy)) for yy in np.linspace(y_top, y_bot, n_in_s + 1)]
    dump_west = [(float(p[0]), float(p[1])) for p in west]

    outer: list[tuple[float, float]] = []
    outer += bottom[:-1]
    outer += dump_west[:-1]
    outer += list(reversed(top))[:-1]
    outer += inlet[:-1]

    verts: list[list[float]] = []
    segs: list[list[int]] = []

    def add_closed(pts: list[tuple[float, float]]) -> None:
        i0 = len(verts)
        for p in pts:
            verts.append([float(p[0]), float(p[1])])
        n = len(pts)
        for i in range(n):
            segs.append([i0 + i, i0 + ((i + 1) % n)])

    add_closed(outer)
    add_closed([(float(x), float(y)) for x, y in hole])

    cx, cy = polygon_centroid(poly0)
    if not point_in_closed_poly(cx, cy, hole_closed):
        # Fallback: average of O-wall ring (inside collar/metal side of hole).
        cx = float(np.mean(ogrid[:, 0, 0]))
        cy = float(np.mean(ogrid[:, 0, 1]))
    holes = np.asarray([[cx, cy]], dtype=float)

    seeds = _seed_size_field(x_in, x_cart, y_bot, y_top, size_fn, hole_closed)
    verts_arr = np.asarray(verts + seeds, dtype=float)
    seg_arr = np.asarray(segs, dtype=np.int32)
    a_max = 0.5 * (1.5 * h_far) ** 2
    # Y: no Steiner on constrained O-outer / dump-west (conformal to hex).
    # q28: a bit softer than q33 to avoid knife tris on long dump/cyclic edges.
    opts = f"pq28a{a_max:.8e}Y"
    res = tr.triangulate(
        {"vertices": verts_arr, "segments": seg_arr, "holes": holes},
        opts,
    )
    vout = np.asarray(res["vertices"], dtype=float)
    tris_i = np.asarray(res["triangles"], dtype=np.int32)
    if tris_i.size == 0:
        raise RuntimeError("hybrid: triangulation produced zero triangles")
    tris = vout[tris_i]  # (n, 3, 2)
    # Positive CCW
    areas = np.array([_tri_area2(t[0], t[1], t[2]) for t in tris], dtype=float)
    flip = areas < 0
    if np.any(flip):
        tris = tris.copy()
        tris[flip] = tris[flip][:, ::-1, :]
        areas = np.abs(areas)

    # Edge length stats
    def edge_lens(t):
        return (
            math.hypot(t[1, 0] - t[0, 0], t[1, 1] - t[0, 1]),
            math.hypot(t[2, 0] - t[1, 0], t[2, 1] - t[1, 1]),
            math.hypot(t[0, 0] - t[2, 0], t[0, 1] - t[2, 1]),
        )

    # Passage / LE / far-inlet clouds for size-field diagnostics
    ax, ay = pass_xy
    pass_hs: list[float] = []
    le_hs: list[float] = []
    far_hs: list[float] = []
    ramp_hs: list[float] = []
    for t in tris:
        cx_t = float(t[:, 0].mean())
        cy_t = float(t[:, 1].mean())
        el = edge_lens(t)
        hmed = float(np.median(el))
        if abs(cx_t - ax) < 0.25 * chord_x and abs(cy_t - ay) < 0.35 * (y_top - y_bot):
            pass_hs.append(hmed)
        # LE fan cloud: forward of mid-chord, near LE x (fluid side of O).
        if cx_t < le[0] + 0.15 * chord_x and math.hypot(cx_t - le[0], cy_t - le[1]) < max(8.0 * h_le, 0.08 * chord_x):
            le_hs.append(hmed)
        # Deep sparse inlet (well west of Laser mark) — exclude the densifying edge band.
        if cx_t <= x_dense - 0.05 * chord_x:
            far_hs.append(hmed)
        elif x_dense < cx_t < float(le[0]):
            ramp_hs.append(hmed)

    # Neighbor size ratio via shared edges (approx from edge length pairs at verts)
    max_ratio = 1.0
    # Build edge → lengths from adjacent tris
    edge_h: dict[tuple[int, int], list[float]] = {}
    for ti, idx in enumerate(tris_i):
        el = edge_lens(tris[ti])
        for a, b, eh in ((int(idx[0]), int(idx[1]), el[0]), (int(idx[1]), int(idx[2]), el[1]), (int(idx[2]), int(idx[0]), el[2])):
            key = (a, b) if a < b else (b, a)
            edge_h.setdefault(key, []).append(eh)
    for hs in edge_h.values():
        if len(hs) >= 2:
            lo, hi = min(hs), max(hs)
            if lo > 1e-16:
                max_ratio = max(max_ratio, hi / lo)
        # also compare to vertex-incident — skip heavy

    # Soft vertex size ratio from incident edge lengths
    vert_len: dict[int, list[float]] = {}
    for (a, b), hs in edge_h.items():
        h = float(np.mean(hs))
        vert_len.setdefault(a, []).append(h)
        vert_len.setdefault(b, []).append(h)
    for hs in vert_len.values():
        if not hs:
            continue
        lo, hi = min(hs), max(hs)
        if lo > 1e-16:
            max_ratio = max(max_ratio, hi / lo)

    def _med_mm(vals: list[float]) -> float:
        return float(np.median(vals) * 1e3) if vals else float("nan")

    stats = {
        "n_tri": float(len(tris)),
        "passage_h_m": float(np.median(pass_hs)) if pass_hs else float("nan"),
        "le_h_m": float(np.median(le_hs)) if le_hs else float("nan"),
        "far_inlet_h_m": float(np.median(far_hs)) if far_hs else float("nan"),
        "ramp_h_m": float(np.median(ramp_hs)) if ramp_hs else float("nan"),
        "x_dense_m": float(x_dense),
        "x_dense_c": float(x_dense_c),
        "max_size_ratio": float(max_ratio),
        "h_le": float(h_le),
        "h_pass": float(h_pass),
        "h_far": float(h_far),
        "growth": float(growth),
    }
    notes.append(
        f"hybrid triangles: n_tri={len(tris)} seeds={len(seeds)} "
        f"h_le={h_le*1e3:.3f}mm h_pass={h_pass*1e3:.3f}mm h_far={h_far*1e3:.3f}mm "
        f"growth={growth:.3g} x_dense_c={x_dense_c:.3g}"
    )
    notes.append(
        f"tri size field axial ramp; deep-far (x<=x_dense-0.05c) median h="
        f"{_med_mm(far_hs):.3f} mm, ramp h={_med_mm(ramp_hs):.3f} mm, "
        f"LE cloud h={_med_mm(le_hs):.3f} mm, "
        f"passage median h={_med_mm(pass_hs):.3f} mm, "
        f"max neighbor size ratio≈{max_ratio:.2f}"
    )
    notes.append("O-outer + dump-west are constrained edges (triangle -Y); no LE holes.")
    notes.append(f"triangle opts={opts} (Shewchuk; not Gmsh)")
    return tris, stats, notes


def build_hybrid_oh_tri(
    poly0: list[tuple[float, float]],
    *,
    x_in: float,
    x_out: float,
    y_bot: float,
    y_top: float,
    n_in: int,
    n_out: int,
    n_cyc: int,
    n_rad: int,
    n_fill: int,
    stretch: float,
    d_o: float,
    n_out_x: int | None = None,
    le_cluster: float = 1.0,
    inlet_stretch: float | None = None,
    h_le: float | None = None,
    h_pass: float | None = None,
    h_far: float | None = None,
    growth: float = 1.25,
    g_min: float | None = None,
    x_dense_c: float = 0.25,
) -> dict[str, Any]:
    """O-collar (quads) + TE wake H + dump H + passage/LE triangles. One pitch."""
    ogrid, h_all, a2, notes = build_offset_oh(
        poly0,
        x_in=x_in,
        x_out=x_out,
        y_bot=y_bot,
        y_top=y_top,
        n_in=n_in,
        n_out=n_out,
        n_cyc=n_cyc,
        n_rad=n_rad,
        n_fill=n_fill,
        stretch=stretch,
        d_o=d_o,
        n_out_x=n_out_x,
        le_cluster=le_cluster,
        inlet_stretch=inlet_stretch,
    )
    wake = h_all[0]
    dump = h_all[-1]
    gm = float(g_min) if g_min is not None else 8e-4
    hp = float(h_pass) if h_pass is not None else float(min(0.12e-3, max(0.05e-3, 0.12 * gm)))
    hl = float(h_le) if h_le is not None else float(0.45 * hp)
    # Far inlet: ~0.15–0.25c (1.5–3 mm class), not 4*h_pass (~0.4 mm).
    xs_poly = [p[0] for p in poly0]
    chord_x = max(max(xs_poly) - min(xs_poly), 1e-6) if xs_poly else 1e-2
    hf_auto = float(max(1.5e-3, min(3.0e-3, 0.20 * chord_x)))
    hf = float(h_far) if h_far is not None else hf_auto
    gr = max(float(growth), 1.05)
    xd_c = max(float(x_dense_c), 0.0)
    tris, stats, tnotes = _build_passage_triangles(
        ogrid=ogrid,
        wake=wake,
        dump=dump,
        poly0=poly0,
        x_in=x_in,
        y_bot=y_bot,
        y_top=y_top,
        h_le=hl,
        h_pass=hp,
        h_far=hf,
        growth=gr,
        x_dense_c=xd_c,
    )
    notes = list(notes) + tnotes
    notes.append(
        "mesh_kind hybrid_OH_tri: keep post-blade TE open-O + dump H; "
        "replace LE/passage Cartesian H with constrained triangles; one pitch true cyclic."
    )
    notes.append("NOT 3-blade premesh stack. NOT Gmsh. NOT cassette.")
    first_cell = float(np.mean(np.linalg.norm(ogrid[:, 1, :] - ogrid[:, 0, :], axis=1)))
    n_quad = int(ogrid.shape[0] * (ogrid.shape[1] - 1))
    # wake cells + dump cells
    n_quad += int((wake.shape[0] - 1) * (wake.shape[1] - 1))
    n_quad += int((dump.shape[0] - 1) * (dump.shape[1] - 1))
    return {
        "ogrid": ogrid,
        "h_blocks": [wake, dump],
        "tris": tris,
        "a2": a2,
        "notes": notes,
        "first_cell": first_cell,
        "n_i": int(ogrid.shape[0]),
        "d_o": float(d_o),
        "stats": stats,
        "n_quad": n_quad,
        "n_tri": int(len(tris)),
    }



def build_body_fitted_oh_shock(
    poly0,
    *,
    x_in,
    x_out,
    y_bot,
    y_top,
    n_in,
    n_out,
    n_cyc,
    n_rad,
    n_fill,
    stretch,
    d_o,
    n_out_x=None,
    le_cluster=2.5,
    inlet_stretch=None,
    beta1_deg=65.0,
    beta2_deg=-65.0,
    chord_m=0.014,
    dump_rx=1.18,
    dump_dx_last=1.4e-3,
    wake_cx=1.0,
    x_dense_c=0.25,
    te_angular_min=10,
    n_pitchwise_throat=40,
    y1_m=None,
):
    """Goldman shock O–H wrapper around conformal build_offset_oh."""
    from .oh_shock import count_te_angular_cells, dump_xs_rx
    import math

    notes = []
    metrics = {}
    c = max(float(chord_m), 1e-6)
    n_pw = max(int(n_pitchwise_throat), int(n_fill), 8)
    n_fill_use = n_pw
    n_in_use = max(int(n_in), 8)
    n_out_use = max(int(n_out), int(te_angular_min) + 8, 18)
    n_cyc_use = max(int(n_cyc), n_pw, 24)
    r_wall = max(float(stretch), 1.0)
    r_inlet = max(float(inlet_stretch if inlet_stretch is not None else 1.12), 1.0)
    rx = max(float(dump_rx), 1.0 + 1e-9)
    dx_last = max(float(dump_dx_last), 1e-6)
    x_te = max(p[0] for p in poly0)
    # dump_xs_1c from TE (no Cartesian cliff). First Δx ≈ O outer Δn estimate.
    # build_offset_oh retargets to measured dn_o; this seed must not chalk-line.
    dn_est = float(d_o) * (r_wall - 1.0) * (r_wall ** max(int(n_rad) - 1, 0)) / max(
        (r_wall ** max(int(n_rad), 1) - 1.0), 1e-12
    ) if r_wall > 1.0 + 1e-12 else float(d_o) / max(int(n_rad), 1)
    if y1_m is not None and float(y1_m) > 0:
        dn_est = max(dn_est, float(y1_m) * (r_wall ** max(int(n_rad) - 1, 0)))
    dx0 = max(float(dn_est), 1e-9)
    L_c = max(1.0, float(wake_cx), (float(x_out) - float(x_te)) / c)
    xs_full = dump_xs_1c(
        float(x_te), c, dx0, n_near=10, stretch_max=min(1.25, float(rx)), L_dump_c=L_c
    )
    # Optionally nudge outlet so last Δx is not far below dump_dx_last (still no TE cliff).
    if float(xs_full[-1] - xs_full[-2]) < 0.65 * dx_last:
        xs_full = np.append(xs_full, float(xs_full[-1]) + dx_last)
    n_dump = int(len(xs_full) - 1)
    ogrid, h_blocks, a2, oh_notes = build_offset_oh(
        poly0,
        x_in=x_in,
        x_out=float(xs_full[-1]),
        y_bot=y_bot,
        y_top=y_top,
        n_in=n_in_use,
        n_out=n_out_use,
        n_cyc=n_cyc_use,
        n_rad=int(n_rad),
        n_fill=n_fill_use,
        stretch=r_wall,
        d_o=float(d_o),
        n_out_x=int(n_dump),
        le_cluster=max(float(le_cluster), 1.0),
        inlet_stretch=r_inlet,
        dump_xs=xs_full,
        dump_rx=rx,
        beta1_deg=float(beta1_deg),
        beta2_deg=float(beta2_deg),
    )
    notes.extend(oh_notes)
    dump = h_blocks[-1]
    xs_d = dump[:, 0, 0]
    last_dx = float(abs(xs_d[-1] - xs_d[-2]))
    metrics.update(
        {
            "dump_last_dx_m": last_dx,
            "dump_L_m": float(abs(xs_d[-1] - xs_d[0])),
            "x_out_m": float(xs_d[-1]),
            "n_pitchwise_throat": int(n_fill_use),
            "n_radial": int(n_rad),
            "stretch": float(r_wall),
            "d_o_m": float(d_o),
        }
    )
    te_ang = count_te_angular_cells(
        ogrid[:, 0, :], te_a=ogrid[0, 0], te_b=ogrid[-1, 0], max_deg=20.0
    )
    metrics["te_angular_cells_in_20deg"] = int(te_ang)
    if te_ang < int(te_angular_min):
        raise RuntimeError(f"TE angular cells in 20° = {te_ang} < {te_angular_min}")
    wall = ogrid[:, 0, :]
    i_le = int(np.argmin(wall[:, 0]))
    ds = np.linalg.norm(np.diff(wall, axis=0), axis=1)
    le_ds = [float(ds[k]) for k in range(max(0, i_le - 3), min(len(ds), i_le + 3))]
    metrics["le_ds_min_m"] = float(min(le_ds)) if le_ds else None
    metrics["le_ds_target_m"] = 0.001 * c
    first_cell = float(np.mean(np.linalg.norm(ogrid[:, 1, :] - ogrid[:, 0, :], axis=1)))
    metrics["first_cell_m"] = first_cell
    if y1_m is not None:
        metrics["y1_m"] = float(y1_m)
    notes.append(
        "mesh_kind body_fitted_OH Goldman shock: conformal O + H; all quads; NOT hybrid_OH_tri."
    )
    return {
        "ogrid": ogrid,
        "h_blocks": h_blocks,
        "a2": float(a2),
        "notes": notes,
        "first_cell": first_cell,
        "n_i": int(ogrid.shape[0]),
        "d_o": float(d_o),
        "metrics": metrics,
        "x_out": float(xs_d[-1]),
        "n_in_use": n_in_use,
        "n_fill": n_fill_use,
        "n_out": int(n_out_use),
        "n_cyc": int(n_cyc_use),
    }



@dataclass
class MeshBuild:
    n_cells: int
    n_points: int
    n_faces: int
    patches: dict[str, int]
    first_cell_m: float
    min_area_2d: float
    check_notes: list[str] = field(default_factory=list)
    y_shift_m: float = 0.0
    pitch_m: float = 0.0
    z_thick_m: float = 0.001
    blade_polys: list[list[tuple[float, float]]] = field(default_factory=list)
    x_in: float = 0.0
    x_out: float = 0.0
    y_min: float = 0.0
    y_max: float = 0.0
    n_around: int = 0
    n_radial: int = 0
    d_o_m: float = 0.0
    mesh_kind: str = "body_fitted_OH"
    n_tri: int = 0
    n_quad: int = 0
    passage_h_m: float = 0.0
    le_h_m: float = 0.0
    far_inlet_h_m: float = 0.0
    max_size_ratio: float = 0.0


def _point_key(x: float, y: float, z: float) -> tuple[int, int, int]:
    return (round(x * 1e12), round(y * 1e12), round(z * 1e12))


def write_polymesh(
    case_dir: Path,
    job: dict[str, Any],
    spec: BladeSpec,
    poly: list[tuple[float, float]] | None = None,
) -> MeshBuild:
    g = job["geometry"]
    cfd = job["cfd"]
    pitch = cascade_pitch_m(job)
    n_blades = int(g["n_blades_cascade"])
    if str(cfd.get("mesh") or "").strip().lower() in ("hoh", "HOH", "mesh_hoh"):
        from .hoh_mesher import build_hoh_mesh
        from .meanline import compute_meanline
        dest = Path(case_dir)
        try:
            res = build_hoh_mesh(job, compute_meanline(job), n_blades=3, tier="balanced", out_dir=dest)
            if not getattr(res, "solvable", False):
                raise RuntimeError(
                    f"HOH not solvable n_cells={res.n_cells} maxNO={res.max_nonortho_deg}"
                )
            pitch_v = float(pitch)
            polys = []
            try:
                from .geometry import profile_from_job as _pfj
                base = list(_pfj(job, spec))
                for k in range(3):
                    polys.append([(x, y + k * pitch_v) for x, y in base])
            except Exception:
                polys = []
            notes = list(res.notes)
            notes.append(f"HOH solvable={res.solvable} success={res.success} maxNO={res.max_nonortho_deg:.2f} skew={res.max_skew:.2f}")
            return MeshBuild(
                n_cells=res.n_cells,
                n_points=0,
                n_faces=0,
                patches={"inlet": [0], "outlet": [0], "bottom": [0], "top": [0], "blades": [0], "frontAndBack": [0]},
                first_cell_m=0.0,
                min_area_2d=0.0,
                check_notes=notes,
                pitch_m=pitch_v,
                mesh_kind="hoh",
                n_quad=res.n_cells,
                blade_polys=polys,
                y_min=min((p[1] for poly in polys for p in poly), default=0.0),
                y_max=max((p[1] for poly in polys for p in poly), default=0.0) + 0.0,
            )
        except Exception as exc:
            # Freeze B: HOH refuse → 1-pitch body_fitted_OH. No 3-blade cassette remesh. No HOH loop.
            job["_hoh_fallback"] = f"HOH writer refuse → body_fitted_OH 1-pitch once: {exc}"
            cfd["mesh"] = "body_fitted_OH"
            n_blades = 1
    x_in = -float(cfd["x_up_c"]) * spec.chord_m
    x_out = spec.chord_m + float(cfd["x_dn_c"]) * spec.chord_m
    n_in = int(cfd["n_inlet"])
    n_out_x = int(cfd["n_outlet"])
    # O-grid east count stays at the ring default; extra n_outlet is dump Δx only.
    n_out = int(cfd.get("n_outlet_ring") or min(n_out_x, 14))
    n_cyc = int(cfd["n_cyclic"])
    n_rad = int(cfd["n_radial"])
    n_around_req = int(cfd.get("n_around", 48))
    n_fill = int(cfd.get("n_pitch_fill", 6))
    stretch = float(cfd["stretch"])
    # Inlet H axial pack ratio; dump still uses stretch on xs_east.
    inlet_stretch = float(cfd.get("inlet_stretch") or stretch)
    zth = float(cfd["z_thick_m"])
    le_cluster = max(float(cfd.get("le_cluster", 2.5) or 2.5), 1.0)
    n_le = int(cfd.get("n_le") or 0)
    mesh_req = str(cfd.get("mesh") or "body_fitted_OH").strip()
    use_hybrid = mesh_req in ("hybrid_OH_tri", "hybrid_oh_tri", "hybrid")
    use_body = (not use_hybrid) and mesh_req in ("body_fitted_OH", "body_fitted", "oh_shock", "")
    h_le_knob = cfd.get("h_le")
    h_pass_knob = cfd.get("h_pass")
    h_far_knob = cfd.get("h_far")
    growth_knob = float(cfd.get("growth") or 1.25)
    x_dense_c_knob = float(cfd.get("x_dense_c") if cfd.get("x_dense_c") not in (None, "") else 0.25)
    hybrid_tris: np.ndarray | None = None
    hybrid_stats: dict[str, float] = {}
    n_tri_cells = 0
    n_quad_cells = 0

    if n_le > n_in:
        # Optional west/LE share boost (inlet-side arc of the O ring).
        n_in = n_le
    # Freeze B: live solve is always 1-pitch translational cyclic (period = 1×pitch).
    # Viz post-stacks blade polys ×N for Mesh/Fields cassette look — do not remesh ×N.
    viz_n = max(int(g.get("n_blades_cascade") or 3), 1)
    job["_viz_stack_blades"] = viz_n
    n_blades = 1
    y_bot, y_top = -0.5 * pitch, 0.5 * pitch

    # 4-side ring: S/N = n_cyclic, W = n_inlet, E = n_outlet.
    # Not 2*(n_cyclic + n_inlet) and not an impulse-cup special case.
    n_around_base = 2 * n_cyc + n_in + n_out
    extra = int(n_around_req) - n_around_base
    if extra >= 2:
        add = extra // 2
        n_cyc += add

    poly0 = list(poly) if poly is not None else profile_from_job(job, spec)
    if polygon_signed_area(poly0) <= 0:
        raise RuntimeError("profile is not a CCW metal interior (zero or negative area)")
    ys0 = [p[1] for p in poly0]
    yspan = max(ys0) - min(ys0)
    fam0 = str((job.get("geometry") or {}).get("profile_family") or "")
    # Nested C is legal when g_min>0 (solids miss). Cassette/H-O-H hosts yspan>=s.
    # Only refuse true intersection. Never flatten outer arc to force a strip fit.
    use_passage = False  # Const. passage OFF. passage.py is dead. Ignore green button.
    passage = None
    cas = None
    y_shift = 0.0
    gap0 = passage_gap(poly0, pitch)
    if float(gap0["g_min"]) <= 0.0:
        raise RuntimeError(
            f"INTERSECTING METAL: passage_gap g_min={float(gap0['g_min'])*1e3:.4f} mm <= 0 "
            f"(SS0 vs PS0+(0,s={pitch*1e3:.3f} mm), arc-length n_hat, not y(x)). "
            "Refuse mesh/solve."
        )
    d_o_gate = min(0.00045, 0.06 * spec.chord_m)
    # Freeze B: never remesh a live 3-blade cassette. Redirect cassette/HOH-fallback → body_fitted.
    _mesh_ask = str(cfd.get("mesh") or "").strip()
    if _mesh_ask in ("cassette_OH", "cassette") or bool(job.get("_hoh_fallback")):
        cfd["mesh"] = "body_fitted_OH"
        use_body = True
        use_hybrid = False
        if not job.get("_hoh_fallback"):
            job["_cassette_redirect"] = "Freeze B: cassette_OH → body_fitted_OH 1-pitch cyclic"
    want_cassette = False
    # Nested yspan>=s used to force cassette; Freeze B keeps 1-pitch if metal fits (g_min>0).
    if want_cassette or ((not use_hybrid) and (not use_body) and yspan + 2.0 * d_o_gate >= float(pitch)):
        # Force dump ≥ TE+1c before cassette write (job x_dn_c may be shorter).
        x_te_metal = max(p[0] for p in poly0)
        x_out = max(float(x_out), float(x_te_metal) + 1.0 * float(spec.chord_m))
        cfd["x_dn_c"] = max(float(cfd.get("x_dn_c") or 0), (x_out - float(spec.chord_m)) / max(float(spec.chord_m), 1e-9))
        cas = build_cassette_oh(
            poly0,
            pitch=pitch,
            n_blades=n_blades,
            x_in=x_in,
            x_out=x_out,
            n_in=n_in,
            n_out=n_out,
            n_cyc=n_cyc,
            n_rad=n_rad,
            n_fill=n_fill,
            n_out_x=n_out_x,
            stretch=stretch,
            d_o=d_o_gate,
            g_min=float(gap0["g_min"]),
            chord_m=float(spec.chord_m),
        )
        y_shift = 0.0
        ogrid = cas["ogrids"][0]
        h_blocks = cas["h_blocks"]
        a2 = cas["a2"]
        first_cell = cas["first_cell"]
        n_i = cas["n_i"]
        oh_notes = list(cas["notes"])
        d_o = cas["d_o"]
        y_bot, y_top = cas["y_bot"], cas["y_top"]
        x_out = float(cas["x_out"])
        x_in = float(cas.get("x_in", x_in))
        oh_notes.append(
            f"Gate 0 g_min={float(gap0['g_min'])*1e3:.3f} mm > 0; mesh_kind cassette_OH "
            f"(yspan={yspan*1e3:.2f} mm + 2 d_o >= s={pitch*1e3:.2f} mm)"
        )
    if use_passage:
        from .passage import build_passage_oh
        passage = build_passage_oh(
            poly0,
            pitch=pitch,
            x_in=x_in,
            x_out=x_out,
            n_in=n_in,
            n_out=n_out_x,
            n_stream=max(n_cyc * 4, 64),
            n_span=max(n_fill, 8),
            n_rad=n_rad,
            stretch=stretch,
            d_o=min(0.00045, 0.06 * spec.chord_m),
        )
        y_shift = 0.0
        n_blades = 2
        ogrid = passage.o_south
        h_blocks = [passage.h_core, passage.h_inlet, passage.h_outlet]
        a2 = passage.min_area_2d
        first_cell = passage.first_cell_m
        n_i = ogrid.shape[0]
        oh_notes = list(passage.notes)
        d_o = passage.d_o
        y_bot, y_top = passage.y_min, passage.y_max
    if passage is None and cas is None:
        poly0, y_shift = center_in_pitch(poly0, pitch)
    xs = [p[0] for p in poly0]
    ys = [p[1] for p in poly0]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    clearance_y = min(y_top - ymax, ymin - y_bot) if (passage is None and cas is None) else 1.0
    if passage is None and cas is None and clearance_y <= 1e-9:
        raise RuntimeError(
            "profile clearance to cyclic is non-positive after center; "
            "expected cassette path when yspan+2*d_o >= s — check Gate 0 / write_kind"
        )
    if passage is None and cas is None:
        d_o = min(0.00045, 0.22 * max(clearance_y, 2e-6), 0.06 * spec.chord_m)
    fam = str((job.get("geometry") or {}).get("profile_family") or "")
    use_cavity = profile_has_cavity(poly0) and fam in (
        "impulse_bucket", "goldman_impulse", "goldman", "goldman_vortex",
    )
    do_cap_note = ""
    if passage is None and cas is None:
        # Mirror cassette Gate-0 collar cap: keep O thin enough to clear g_min.
        d_o_req = float(d_o)
        if use_cavity:
            # Tight pack: spend most of the cyclic gap on the O-collar, keep ≥8 µm.
            d_o_req = min(0.00045, max(2e-6, 0.55 * clearance_y), 0.06 * spec.chord_m)
        d_cap = 0.28 * max(float(gap0["g_min"]), 1e-6)
        d_o = min(d_o_req, d_cap, 0.00045)
        if d_o < d_o_req - 1e-16:
            do_cap_note = (
                f"d_o capped {d_o_req:.3g} → {d_o:.3g} m by 0.28*g_min "
                f"(g_min={float(gap0['g_min'])*1e3:.3f} mm); O not thickened for shocks"
            )
    shock_metrics: dict = {}
    if passage is None and cas is None and use_cavity and use_body:
        from .oh_shock import d_o_from_y1, y1_wall_m
        gas = job.get("gas") or {}
        y1, u_tau, y1_note = y1_wall_m(
            mu_pa_s=float(gas.get("mu_pa_s") or 1e-5),
            rho1_kg_m3=float(gas.get("rho1_kg_m3") or gas.get("rho1") or 1.0),
            w1_m_s=float(gas.get("w1_m_s") or 1.0),
            yplus_target=float(cfd.get("yplus_target") or 1.0),
            u_tau_frac_w1=float(cfd.get("u_tau_frac_w1") or 0.05),
        )
        shock_metrics["y1_m"] = y1
        shock_metrics["u_tau"] = u_tau
        shock_metrics["y1_note"] = y1_note
        # Restore real wall inflation: y1 from yplus_target, n_rad∈[15,25], growth≤1.25.
        r_use = min(max(float(stretch), 1.05), 1.25)
        n_rad = int(n_rad) if int(n_rad) > 0 else 20
        n_rad = max(15, min(25, n_rad))
        d_cap = min(
            0.28 * max(float(gap0["g_min"]), 1e-6),
            0.45 * max(clearance_y, 2e-6),
            0.06 * spec.chord_m,
            0.00045,
        )
        # Prefer keeping y1: shrink n_rad before crushing first-cell below target.
        d_o_req = d_o_from_y1(y1, n_rad, r_use)
        while n_rad > 15 and d_o_req > d_cap * 1.001:
            n_rad -= 1
            d_o_req = d_o_from_y1(y1, n_rad, r_use)
        d_o = min(d_o_req, d_cap)
        stretch = r_use
        cfd["n_radial"] = n_rad
        cfd["stretch"] = stretch
        shock_metrics["n_radial"] = n_rad
        shock_metrics["stretch"] = stretch
        shock_metrics["d_o_m"] = d_o
        shock_metrics["d_o_req_m"] = d_o_req
        beta1 = float(g.get("beta1_flow_deg") or 65.0)
        beta2 = float(g.get("beta2_flow_deg") or -65.0)
        n_pw = int(cfd.get("n_pitchwise_throat") or max(n_fill, 40))
        n_fill = max(n_fill, n_pw)
        hy = build_body_fitted_oh_shock(
            poly0,
            x_in=x_in,
            x_out=x_out,
            y_bot=y_bot,
            y_top=y_top,
            n_in=n_in,
            n_out=n_out,
            n_cyc=n_cyc,
            n_rad=n_rad,
            n_fill=n_fill,
            stretch=stretch,
            d_o=d_o,
            n_out_x=n_out_x,
            le_cluster=le_cluster,
            inlet_stretch=inlet_stretch,
            beta1_deg=beta1,
            beta2_deg=beta2,
            chord_m=spec.chord_m,
            dump_rx=float(cfd.get("dump_rx") or 1.18),
            dump_dx_last=float(cfd.get("dump_dx_last_m") or 1.4e-3),
            wake_cx=float(cfd.get("wake_cx") or 1.0),
            x_dense_c=float(cfd.get("x_dense_c") or 0.25),
            te_angular_min=int(cfd.get("te_angular_min") or 10),
            n_pitchwise_throat=n_pw,
            y1_m=y1,
        )
        ogrid = hy["ogrid"]
        h_blocks = hy["h_blocks"]
        a2 = hy["a2"]
        oh_notes = list(hy["notes"])
        first_cell = hy["first_cell"]
        n_i = hy["n_i"]
        d_o = hy["d_o"]
        x_out = float(hy.get("x_out") or x_out)
        shock_metrics.update(hy.get("metrics") or {})
        shock_metrics["yplus_target"] = float(cfd.get("yplus_target") or 1.0)
        # Prove first-cell on metal (O j=0) tracks y1 within 2× (inflation restored).
        if y1 > 0 and first_cell > 0:
            ratio_fc = float(first_cell) / float(y1)
            shock_metrics["first_cell_over_y1"] = ratio_fc
            oh_notes.append(
                f"inflation: first_cell={first_cell:.3g} m y1={y1:.3g} m "
                f"ratio={ratio_fc:.3g} n_rad={n_rad} stretch={stretch:.3g} d_o={d_o:.3g} m "
                f"(first cell on metal O j=0)"
            )
            if ratio_fc > 2.5 or ratio_fc < 0.35:
                oh_notes.append(
                    f"WARN first_cell/y1={ratio_fc:.3g} outside ~[0.35,2.5] — inflation soft"
                )
        n_quad_cells = int(n_i * n_rad)
        for hb in h_blocks:
            n_quad_cells += int((hb.shape[0] - 1) * (hb.shape[1] - 1))
        n_quad_cells *= int(n_blades)
        job["_oh_shock"] = dict(shock_metrics)
        oh_notes.append(
            f"body_fitted_OH child n_blades_cascade={n_blades}; "
            f"stack_after_child_ok={bool(cfd.get('stack_after_child_ok'))}"
        )
    elif passage is None and cas is None and use_cavity and use_hybrid:
        hy = build_hybrid_oh_tri(
            poly0,
            x_in=x_in,
            x_out=x_out,
            y_bot=y_bot,
            y_top=y_top,
            n_in=n_in,
            n_out=n_out,
            n_cyc=n_cyc,
            n_rad=n_rad,
            n_fill=n_fill,
            stretch=stretch,
            d_o=d_o,
            n_out_x=n_out_x,
            le_cluster=le_cluster,
            inlet_stretch=inlet_stretch,
            h_le=(float(h_le_knob) if h_le_knob not in (None, "") else None),
            h_pass=(float(h_pass_knob) if h_pass_knob not in (None, "") else None),
            h_far=(float(h_far_knob) if h_far_knob not in (None, "") else None),
            growth=growth_knob,
            g_min=float(gap0["g_min"]),
            x_dense_c=x_dense_c_knob,
        )
        ogrid = hy["ogrid"]
        h_blocks = hy["h_blocks"]
        hybrid_tris = hy["tris"]
        a2 = hy["a2"]
        oh_notes = list(hy["notes"])
        first_cell = hy["first_cell"]
        n_i = hy["n_i"]
        n_tri_cells = int(hy["n_tri"])
        n_quad_cells = int(hy["n_quad"])
        hybrid_stats = dict(hy["stats"])
        if do_cap_note:
            oh_notes.append(do_cap_note)
        if le_cluster > 1.0 + 1e-12:
            oh_notes.append(f"le_cluster={le_cluster:.3g} n_le={n_le or n_in} (west W=n_inlet={n_in})")
    elif passage is None and cas is None and use_cavity:
        ogrid, h_blocks, a2, oh_notes = build_offset_oh(
            poly0,
            x_in=x_in,
            x_out=x_out,
            y_bot=y_bot,
            y_top=y_top,
            n_in=n_in,
            n_out=n_out,
            n_cyc=n_cyc,
            n_rad=n_rad,
            n_fill=n_fill,
            stretch=stretch,
            d_o=d_o,
            n_out_x=n_out_x,
            le_cluster=le_cluster,
            inlet_stretch=inlet_stretch,
        )
        first_cell = float(np.mean(np.linalg.norm(ogrid[:, 1, :] - ogrid[:, 0, :], axis=1)))
        n_i = ogrid.shape[0]
        if do_cap_note:
            oh_notes.append(do_cap_note)
        if le_cluster > 1.0 + 1e-12:
            oh_notes.append(f"le_cluster={le_cluster:.3g} n_le={n_le or n_in} (west W=n_inlet={n_in})")
    elif passage is None and cas is None:
        # Tight AABB around a wall-normal offset (not metal+d_o). Cartesian H-blocks
        # need a vertical west edge; wrapping a C-shaped LE in one TFI H-block folds.
        inner_hi = resample_closed(poly0, max(4 * (2 * n_cyc + n_in + n_out), 200))
        if polygon_signed_area([(float(x), float(y)) for x, y in inner_hi]) < 0:
            inner_hi = inner_hi[::-1].copy()
        off_hi = offset_closed(inner_hi, d_o, n_smooth=12)
        bx0, bx1 = float(off_hi[:, 0].min()), float(off_hi[:, 0].max())
        by0, by1 = float(off_hi[:, 1].min()), float(off_hi[:, 1].max())
        pad_y = 0.72 * min(by0 - (y_bot + 1e-6), (y_top - 1e-6) - by1)
        pad_x = max(pad_y, 0.05 * spec.chord_m)
        pad_y = max(pad_y, 0.0)
        bx0 -= pad_x
        bx1 += pad_x
        by0 -= pad_y
        by1 += pad_y
        if by0 <= y_bot + 1e-6 or by1 >= y_top - 1e-6:
            raise RuntimeError("O-grid AABB collides with cyclic pitch boundary")
        if bx0 <= x_in + 1e-7 or bx1 >= x_out - 1e-7:
            raise RuntimeError("O-grid AABB collides with inlet/outlet")
        outer, ranges = outer_rectangle(bx0, bx1, by0, by1, n_cyc, n_out, n_cyc, n_in)

        def _positive_ogrid(inner_ring: np.ndarray) -> np.ndarray:
            og = build_hybrid_aabb_ogrid(inner_ring, outer, n_rad, stretch, 0.90 * d_o)
            if min_cell_area_2d(og) <= 0:
                og = og[::-1].copy()
            if min_cell_area_2d(og) <= 0:
                og = build_pitch_ogrid(inner_ring, outer, n_rad, stretch)
                if min_cell_area_2d(og) <= 0:
                    og = og[::-1].copy()
            return og

        if le_cluster > 1.0 + 1e-12:
            inner = inner_match_by_arclength(poly0, outer, le_cluster=le_cluster)
            ogrid = _positive_ogrid(inner)
            used = f"LE-clustered arc-length inner (le_cluster={le_cluster:.3g})"
            if min_cell_area_2d(ogrid) <= 0:
                inner = inner_match_by_angle(poly0, outer)
                ogrid = _positive_ogrid(inner)
                used = "centroid-angle inner (LE arc-length folded)"
        else:
            inner = inner_match_by_angle(poly0, outer)
            ogrid = _positive_ogrid(inner)
            used = "centroid-angle inner"
        if min_cell_area_2d(ogrid) <= 0:
            inner = inner_from_profile(poly0, outer, ranges)
            ogrid = _positive_ogrid(inner)
            used = "4-side inner (angle match folded)"
        a2 = min_cell_area_2d(ogrid)
        if a2 <= 0:
            raise RuntimeError(f"folded O-grid: min quad area {a2:.3e} m2")
        sm = smooth_ogrid(ogrid, n_iter=12, omega=0.35)
        if min_cell_area_2d(sm) > 0:
            ogrid = sm
            a2 = min_cell_area_2d(ogrid)
        first_cell = float(np.mean(np.linalg.norm(ogrid[:, 1, :] - ogrid[:, 0, :], axis=1)))
        n_i = ogrid.shape[0]
        # Inlet H axial pack toward AABB west face (LE / blade), mirror xs_east dump stretch.
        xs_west = _xs_pack_hi(x_in, bx0, n_in, max(inlet_stretch, 1.0))
        h_west = _pos_block(cartesian_block_xy(xs_west, np.linspace(by0, by1, n_in + 1)), "west")
        xs_east = np.array([bx1 + (x_out - bx1) * _stretch(j, n_out_x, max(stretch, 1.0)) for j in range(n_out_x + 1)])
        h_east = _pos_block(cartesian_block_xy(xs_east, np.linspace(by0, by1, n_out + 1)), "east")
        h_south = _pos_block(cartesian_block(bx0, bx1, y_bot, by0, n_cyc, n_fill), "south")
        h_north = _pos_block(cartesian_block(bx0, bx1, by1, y_top, n_cyc, n_fill), "north")
        h_sw = _pos_block(cartesian_block_xy(xs_west, np.linspace(y_bot, by0, n_fill + 1)), "sw")
        h_se = _pos_block(cartesian_block_xy(xs_east, np.linspace(y_bot, by0, n_fill + 1)), "se")
        h_nw = _pos_block(cartesian_block_xy(xs_west, np.linspace(by1, y_top, n_fill + 1)), "nw")
        h_ne = _pos_block(cartesian_block_xy(xs_east, np.linspace(by1, y_top, n_fill + 1)), "ne")
        h_blocks = [h_west, h_east, h_south, h_north, h_sw, h_se, h_nw, h_ne]
        oh_notes = [
            "O-outer is a padded AABB around the wall-normal offset; H-blocks Cartesian (cyclics conformal).",
            "Inner ring prefers centroid-angle matching (4-side AABB LE/TE fans only if angle match folds).",
            "Near-wall O is offset; only the last 1–2 layers morph to the AABB.",
            "Last morph layer wrap at AABB corners is shortened by extra AABB pad (skew 4.68 hole).",
            "NOT subsetMesh stairs. One-block TFI from a wrapped LE to the inlet folds; not used.",
            f"4-side n_around: S/N=n_cyclic={n_cyc} E=n_outlet={n_out} W=n_inlet={n_in} (not 2*(n_cyc+n_in)). {used}",
            f"AABB pad_x={pad_x:.3g} m pad_y={pad_y:.3g} m (room to morph oval→rectangle).",
            f"inlet H axial pack toward LE (inlet_stretch={max(inlet_stretch, 1.0):.3g}, n_inlet={n_in})",
        ]
        if do_cap_note:
            oh_notes.append(do_cap_note)
        if le_cluster > 1.0 + 1e-12:
            oh_notes.append(f"le_cluster={le_cluster:.3g} n_le={n_le or n_in} (west W=n_inlet={n_in})")

    if cas is not None:
        y_min, y_max = cas["y_bot"], cas["y_top"]
        blade_polys = [list(b) for b in cas["blades"]]
    elif passage is not None:
        y_min, y_max = passage.y_min, passage.y_max
        blade_polys = [list(poly0), [(xy[0], xy[1] + pitch) for xy in poly0]]
    else:
        y_min = y_bot
        y_max = y_bot + n_blades * pitch
        blade_polys = [[(xy[0], xy[1] + k * pitch) for xy in poly0] for k in range(n_blades)]

    key_to_id: dict[tuple[int, int, int], int] = {}
    points: list[tuple[float, float, float]] = []

    def pid_xy(x: float, y: float, kz: int) -> int:
        z = kz * zth
        key = _point_key(x, y, z)
        if key not in key_to_id:
            key_to_id[key] = len(points)
            points.append((x, y, z))
        return key_to_id[key]

    HEX_FACES = (
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (3, 7, 6, 2),
        (0, 4, 7, 3),
        (1, 2, 6, 5),
    )
    # Prism (extruded triangle): verts 0,1,2 @ z0 and 3,4,5 @ z1.
    PRISM_FACES = (
        (0, 2, 1),
        (3, 4, 5),
        (0, 1, 4, 3),
        (1, 2, 5, 4),
        (2, 0, 3, 5),
    )
    WALL_FACE = 2
    face_owner: dict[frozenset[int], tuple[list[int], int]] = {}
    face_neigh: dict[frozenset[int], int] = {}
    wall_keys: dict[frozenset[int], str] = {}
    n_cells = 0

    def add_hex(verts: list[int], wall_patch: str | None = None) -> None:
        nonlocal n_cells
        ci = n_cells
        n_cells += 1
        for fi, fs in enumerate(HEX_FACES):
            fverts = [verts[q] for q in fs]
            key = frozenset(fverts)
            if wall_patch is not None and fi == WALL_FACE:
                wall_keys[key] = wall_patch
            if key in face_owner:
                face_neigh[key] = ci
            else:
                face_owner[key] = (fverts, ci)

    def add_prism(corners_xy: list[tuple[float, float]]) -> None:
        """Extrude a CCW triangle to a prism (empty frontAndBack)."""
        nonlocal n_cells
        ci = n_cells
        n_cells += 1
        bots = [pid_xy(x, y, 0) for x, y in corners_xy]
        tops = [pid_xy(x, y, 1) for x, y in corners_xy]
        verts = bots + tops
        for fs in PRISM_FACES:
            fverts = [verts[q] for q in fs]
            key = frozenset(fverts)
            if key in face_owner:
                face_neigh[key] = ci
            else:
                face_owner[key] = (fverts, ci)

    def add_struct(pts: np.ndarray, dy: float, periodic_i: bool, wall_patch: str | None = None, wall_hi: str | None = None, flip_neg: bool = True) -> None:
        ni, nj = pts.shape[0], pts.shape[1]
        n_ic = ni if periodic_i else ni - 1
        n_jc = nj - 1
        for i in range(n_ic):
            i2 = (i + 1) % ni if periodic_i else i + 1
            for j in range(n_jc):
                corners = [
                    (float(pts[i, j, 0]), float(pts[i, j, 1]) + dy),
                    (float(pts[i2, j, 0]), float(pts[i2, j, 1]) + dy),
                    (float(pts[i2, j + 1, 0]), float(pts[i2, j + 1, 1]) + dy),
                    (float(pts[i, j + 1, 0]), float(pts[i, j + 1, 1]) + dy),
                ]
                if flip_neg and _quad_area(corners[0], corners[1], corners[2], corners[3]) <= 0:
                    corners = [corners[1], corners[0], corners[3], corners[2]]
                verts = [pid_xy(x, y, 0) for x, y in corners] + [pid_xy(x, y, 1) for x, y in corners]
                wp = wall_patch if (j == 0 and wall_patch) else (wall_hi if (j == n_jc - 1 and wall_hi) else None)
                add_hex(verts, wall_patch=wp)

    cell_xy: list[tuple[float, float]] = []
    _add_hex0 = add_hex

    def add_hex(verts: list[int], wall_patch: str | None = None) -> None:
        _add_hex0(verts, wall_patch=wall_patch)
        xs = [points[v][0] for v in verts[:4]]
        ys = [points[v][1] for v in verts[:4]]
        cell_xy.append((sum(xs) / 4.0, sum(ys) / 4.0))

    if cas is not None:
        for k, og in enumerate(cas["ogrids"]):
            add_struct(og, 0.0, True, wall_patch=f"blade{k}")
        for hb in cas["h_blocks"]:
            add_struct(hb, 0.0, False)
        shrunk = []
        inset = max(0.35 * float(cas["d_o"]), 2e-6)
        for bp in blade_polys:
            ring = resample_closed(bp, max(len(bp), 120))
            if polygon_signed_area([(float(x), float(y)) for x, y in ring]) < 0:
                ring = ring[::-1].copy()
            sh = offset_closed(ring, -inset, n_smooth=4)
            shrunk.append([(float(x), float(y)) for x, y in sh] + [(float(sh[0, 0]), float(sh[0, 1]))])
        n_og_cells = sum(int(og.shape[0] * (og.shape[1] - 1)) for og in cas["ogrids"])
        n_in_c = 0
        for cx, cy in cell_xy[n_og_cells:]:
            for bp in shrunk:
                if point_in_closed_poly(cx, cy, bp):
                    n_in_c += 1
                    break
        if n_in_c:
            raise RuntimeError(f"{n_in_c} H-block cell centres lie inside a closed C (cassette abort)")
    elif passage is not None:
        add_struct(passage.h_core, 0.0, False, wall_patch="blade0", wall_hi="blade1")
        add_struct(passage.h_inlet, 0.0, False)
        add_struct(passage.h_outlet, 0.0, False)
    else:
        # H-O-H: O open at TE (matching wake cuts); cavity-O periodic wrap is dead.
        o_periodic = not any("H-O-H TE wake" in n for n in (oh_notes or []))
        hoh_wake = o_periodic is False
        for k in range(n_blades):
            dy = k * pitch
            add_struct(ogrid, dy, o_periodic, wall_patch=f"blade{k}")
            for bi, hb in enumerate(h_blocks):
                wp = f"blade{k}" if (hoh_wake and bi == 0) else None
                # Wake: no 2D flip — flip was making OF face pyramids disagree at TE.
                flip = not (hoh_wake and bi == 0)
                add_struct(hb, dy, False, wall_patch=wp, flip_neg=flip)
        if hybrid_tris is not None:
            for tri in hybrid_tris:
                corners = [(float(tri[i, 0]), float(tri[i, 1])) for i in range(3)]
                if _tri_area2(corners[0], corners[1], corners[2]) < 0:
                    corners = [corners[0], corners[2], corners[1]]
                add_prism(corners)

    internal = []
    boundary = []
    for key, (fverts, owner) in face_owner.items():
        if key in face_neigh:
            nb = face_neigh[key]
            if owner > nb:
                owner, nb = nb, owner
                fverts = list(reversed(fverts))
            internal.append((fverts, owner, nb))
        else:
            boundary.append((fverts, owner, key))
    # checkMesh "Faces not in upper triangular order" without renumberMesh.
    internal.sort(key=lambda t: (t[1], t[2]))

    def fcent_t(fverts: list[int]) -> tuple[float, float, float]:
        n = len(fverts)
        return (
            sum(points[i][0] for i in fverts) / n,
            sum(points[i][1] for i in fverts) / n,
            sum(points[i][2] for i in fverts) / n,
        )

    tol = 1e-8
    blade_names = [f"blade{k}" for k in range(n_blades)]
    # Classic cascade: translational cyclics on the FULL pitch strip (including x < LE).
    # No ductBottom/ductTop horizontal lids (those straightened the β1 jet).

    buckets: dict[str, list[tuple[list[int], int, tuple[float, float, float]]]] = {
        "inlet": [], "outlet": [], "bottom": [], "top": [],
        "frontAndBack": [], **{n: [] for n in blade_names},
    }
    unclassified = 0
    for fverts, owner, key in boundary:
        c = fcent_t(fverts)
        x, y, z = c
        if abs(z - 0.0) < tol or abs(z - zth) < tol:
            buckets["frontAndBack"].append((fverts, owner, c))
        elif key in wall_keys:
            buckets[wall_keys[key]].append((fverts, owner, c))
        elif abs(x - x_in) < 5e-4:
            buckets["inlet"].append((fverts, owner, c))
        elif abs(x - x_out) < 5e-4:
            buckets["outlet"].append((fverts, owner, c))
        elif passage is not None and passage.cyclic and (
            abs(y - passage.o_south[0, 0, 1]) < 1e-6
            or abs(y - passage.o_south[-1, 0, 1]) < 1e-6
        ):
            buckets["bottom"].append((fverts, owner, c))
        elif passage is not None and passage.cyclic and (
            abs(y - (passage.o_south[0, 0, 1] + pitch)) < 1e-6
            or abs(y - (passage.o_south[-1, 0, 1] + pitch)) < 1e-6
        ):
            buckets["top"].append((fverts, owner, c))
        elif cas is not None and cas.get("P_bot") is not None:
            # Mid-gap cyclics: tag by nearest point on P_bot / P_top (not y=const).
            import numpy as _np
            pb = _np.asarray(cas["P_bot"], dtype=float)
            pt = _np.asarray(cas["P_top"], dtype=float)
            db = float(_np.min((pb[:, 0] - x) ** 2 + (pb[:, 1] - y) ** 2)) ** 0.5
            dt = float(_np.min((pt[:, 0] - x) ** 2 + (pt[:, 1] - y) ** 2)) ** 0.5
            tol_cyc = max(2.5e-4, 0.02 * float(cas.get("cyclic_Y") or pitch))
            if db <= tol_cyc or dt <= tol_cyc:
                if db <= dt:
                    buckets["bottom"].append((fverts, owner, c))
                else:
                    buckets["top"].append((fverts, owner, c))
            else:
                unclassified += 1
        elif (passage is None) and abs(y - y_min) < 1e-7:
            buckets["bottom"].append((fverts, owner, c))
        elif (passage is None) and abs(y - y_max) < 1e-7:
            buckets["top"].append((fverts, owner, c))
        else:
            if passage is not None:
                ymid = 0.5 * (passage.y_min + passage.y_max)
                name = "blade0" if y < ymid else "blade1"
                buckets[name].append((fverts, owner, c))
            else:
                unclassified += 1
    if unclassified:
        raise RuntimeError(f"{unclassified} boundary faces not on wall/inlet/outlet/cyclic/empty")
    if any(not buckets[n] for n in blade_names):
        raise RuntimeError("per-blade wall patches missing faces")
    if buckets["bottom"] or buckets["top"]:
        if len(buckets["bottom"]) != len(buckets["top"]):
            raise RuntimeError(
                f"cyclic face count mismatch bottom={len(buckets['bottom'])} top={len(buckets['top'])}"
            )

    buckets["bottom"].sort(key=lambda t: (round(t[2][0], 9), round(t[2][2], 9)))
    buckets["top"].sort(key=lambda t: (round(t[2][0], 9), round(t[2][2], 9)))

    def _canon_cyclic_outward(fverts: list[int], *, side: str) -> list[int]:
        """Match body_fitted cyclic quads: both start at (min x, z=0); opposite xz winding
        so each face is outward (bottom -y, top +y) and 0th vertices couple.
        bottom: (lo,z0)->(hi,z0)->(hi,z1)->(lo,z1)
        top:    (lo,z0)->(lo,z1)->(hi,z1)->(hi,z0)
        """
        if len(fverts) != 4:
            return fverts
        ps = [(points[i][0], points[i][2], i) for i in fverts]
        z0 = min(p[1] for p in ps)
        z1 = max(p[1] for p in ps)
        bot = sorted([p for p in ps if abs(p[1] - z0) <= abs(p[1] - z1)], key=lambda p: p[0])
        top = sorted([p for p in ps if abs(p[1] - z1) < abs(p[1] - z0)], key=lambda p: p[0])
        # robust split by z clusters
        zs = sorted(set(round(p[1], 15) for p in ps))
        if len(zs) >= 2:
            z0, z1 = zs[0], zs[-1]
            bot = sorted([p for p in ps if abs(p[1] - z0) < 1e-12], key=lambda p: p[0])
            top = sorted([p for p in ps if abs(p[1] - z1) < 1e-12], key=lambda p: p[0])
        if len(bot) != 2 or len(top) != 2:
            return fverts
        lo_z0, hi_z0 = bot[0][2], bot[1][2]
        lo_z1, hi_z1 = top[0][2], top[1][2]
        if side == "bottom":
            return [lo_z0, hi_z0, hi_z1, lo_z1]
        return [lo_z0, lo_z1, hi_z1, hi_z0]

    if buckets["bottom"] and buckets["top"]:
        buckets["bottom"] = [
            (_canon_cyclic_outward(fv, side="bottom"), ow, c) for fv, ow, c in buckets["bottom"]
        ]
        buckets["top"] = [
            (_canon_cyclic_outward(fv, side="top"), ow, c) for fv, ow, c in buckets["top"]
        ]

    patch_order = ["inlet", "outlet"]
    if buckets["bottom"] or buckets["top"]:
        patch_order.extend(["bottom", "top"])
    patch_order.extend(["frontAndBack", *blade_names])

    all_faces: list[list[int]] = []
    owners: list[int] = []
    neighs: list[int] = []
    for fverts, ow, nb in internal:
        all_faces.append(fverts)
        owners.append(ow)
        neighs.append(nb)
    start: dict[str, int] = {}
    counts: dict[str, int] = {}
    for name in patch_order:
        start[name] = len(all_faces)
        for fverts, ow, _c in buckets[name]:
            all_faces.append(fverts)
            owners.append(ow)
        counts[name] = len(buckets[name])

    mesh_dir = case_dir / "constant" / "polyMesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    n_internal = len(neighs)
    # Translational cyclic period = exactly 1×pitch for the 1-pitch strip (Freeze B).
    sep = float(pitch) if cas is None else float(cas.get("cyclic_Y") or (n_blades * pitch))

    def w(name: str, body: str, cls: str, obj: str, note: str = "") -> None:
        hdr = _foam_header(cls, obj, note=note) if note else _foam_header(cls, obj)
        (mesh_dir / name).write_text(hdr + body + "\n", encoding="utf-8")

    w("points", f"{len(points)}\n(\n" + "".join(f"({x:.12g} {y:.12g} {z:.12g})\n" for x, y, z in points) + ")\n", "vectorField", "points")
    w("faces", f"{len(all_faces)}\n(\n" + "".join(f"{len(fv)}(" + " ".join(str(i) for i in fv) + ")\n" for fv in all_faces) + ")\n", "faceList", "faces")
    w("owner", f"{len(owners)}\n(\n" + "".join(f"{i}\n" for i in owners) + ")\n", "labelList", "owner",
      note=f"nPoints:{len(points)}  nCells:{n_cells}  nFaces:{len(all_faces)}  nInternalFaces:{n_internal}")
    w("neighbour", f"{len(neighs)}\n(\n" + "".join(f"{i}\n" for i in neighs) + ")\n", "labelList", "neighbour",
      note=f"nPoints:{len(points)}  nCells:{n_cells}  nFaces:{len(all_faces)}  nInternalFaces:{n_internal}")

    btxt = [f"{len(patch_order)}\n(\n"]
    for name in patch_order:
        if name in ("bottom", "top"):
            neigh = "top" if name == "bottom" else "bottom"
            svec = sep if name == "bottom" else -sep
            extra = (
                "        inGroups        1(cyclic);\n"
                "        matchTolerance  0.0001;\n"
                "        transform       translational;\n"
                f"        neighbourPatch  {neigh};\n"
                f"        separationVector (0 {svec:.12g} 0);\n"
            )
            ptype = "cyclic"
        elif name == "frontAndBack":
            ptype = "empty"
            extra = "        inGroups        1(empty);\n"
        elif name.startswith("blade"):
            ptype = "wall"
            extra = "        inGroups        1(wall);\n"
        else:
            ptype = "patch"
            extra = ""
        btxt.append(
            f"    {name}\n    {{\n        type            {ptype};\n{extra}"
            f"        nFaces          {counts[name]};\n        startFace       {start[name]};\n    }}\n"
        )
    btxt.append(")\n")
    w("boundary", "".join(btxt), "polyBoundaryMesh", "boundary")

    if cas is not None:
        kind = "cassette_OH"
        kind_note = (
            "mesh: cassette_OH — three closed C metals, fluid outside every C, "
            "bottom/top translational cyclics (n_blades*pitch), full-height inlet/outlet."
        )
    elif passage is not None:
        kind = "passage_OH"
        kind_note = "mesh: Goldman/Katsanis passage O+H (SS0 vs PS0+s). Not a pitch rectangle."
    elif hybrid_tris is not None:
        kind = "hybrid_OH_tri"
        kind_note = (
            "mesh: hybrid_OH_tri — O-collar quads + TE wake/dump H + constrained Delaunay "
            "triangles in LE/passage; one pitch true cyclic (Shewchuk triangle, not Gmsh)."
        )
    else:
        kind = "body_fitted_OH"
        kind_note = "mesh: body-fitted O-grid on the metal + Cartesian H-blocks to the pitch rectangle."
    if n_quad_cells <= 0 and hybrid_tris is None:
        # legacy count: all hex cells are quads extruded
        n_quad_cells = int(n_cells)
    notes = [
        kind_note,
        "NOT Cartesian subsetMesh stair-step.",
        f"{n_blades} blade(s) in polyMesh (viz_stack={job.get('_viz_stack_blades', 1)}), "
        f"cyclic period={sep:.6g} m (=1×pitch), empty frontAndBack, slab {zth} m.",
        f"O n_around={n_i} (JSON n_around={n_around_req}; S/N=n_cyclic={n_cyc} E=n_outlet={n_out} W=n_inlet={n_in}) n_radial={n_rad}",
        f"n_cells={n_cells} first_cell≈{first_cell:.3g} m d_o={d_o:.3g} m min O-quad {a2:.3e} m2",
        f"y_shift to centre blade in pitch: {y_shift:.6g} m (rigid; metal angles unchanged).",
        "Wall faces tagged from the O-grid j=0 ring (per-blade patches).",
        (
            f"classic cascade cyclics: full pitch strip bottom/top translational "
            f"(incl. x < LE); nFaces={len(buckets['bottom'])}/{len(buckets['top'])}. "
            "No ductBottom/ductTop lids."
        ),
        *oh_notes,
        *([job["_hoh_fallback"]] if job.get("_hoh_fallback") else []),
    ]
    if kind == "hybrid_OH_tri":
        notes.append(
            f"counts n_tri={n_tri_cells} n_quad={n_quad_cells}; "
            f"cyclic separationVector = one pitch ({pitch:.6g} m)."
        )
    return MeshBuild(
        n_cells=n_cells, n_points=len(points), n_faces=len(all_faces), patches=counts,
        first_cell_m=first_cell, min_area_2d=a2, check_notes=notes, y_shift_m=y_shift,
        pitch_m=pitch, z_thick_m=zth, blade_polys=blade_polys, x_in=x_in, x_out=x_out,
        y_min=y_min, y_max=y_max, n_around=n_i, n_radial=n_rad, d_o_m=d_o,
        mesh_kind=kind,
        n_tri=n_tri_cells,
        n_quad=n_quad_cells if n_quad_cells else (n_cells - n_tri_cells),
        passage_h_m=float(hybrid_stats.get("passage_h_m") or 0.0),
        le_h_m=float(hybrid_stats.get("le_h_m") or 0.0),
        far_inlet_h_m=float(hybrid_stats.get("far_inlet_h_m") or 0.0),
        max_size_ratio=float(hybrid_stats.get("max_size_ratio") or 0.0),
    )


def _parse_boundary_patches(mesh_dir: Path) -> dict[str, tuple[int, int]]:
    """name → (startFace, nFaces) from ascii polyMesh/boundary."""
    text = (mesh_dir / "boundary").read_text(encoding="utf-8", errors="replace")
    out: dict[str, tuple[int, int]] = {}
    for m in re.finditer(
        r"(\w+)\s*\{\s*[^}]*?nFaces\s+(\d+)\s*;\s*[^}]*?startFace\s+(\d+)\s*;",
        text,
        flags=re.S,
    ):
        out[m.group(1)] = (int(m.group(3)), int(m.group(2)))
    if "frontAndBack" not in out:
        for m in re.finditer(
            r"(\w+)\s*\{\s*[^}]*?startFace\s+(\d+)\s*;\s*[^}]*?nFaces\s+(\d+)\s*;",
            text,
            flags=re.S,
        ):
            out[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    return out


def _read_foam_points(mesh_dir: Path) -> np.ndarray:
    text = (mesh_dir / "points").read_text(encoding="utf-8", errors="replace")
    m = re.search(r"FoamFile\s*\{.*?\}\s*", text, flags=re.S)
    if m:
        text = text[m.end() :]
    m = re.search(r"(\d+)\s*\(", text)
    if not m:
        raise RuntimeError("polyMesh points: no count")
    n = int(m.group(1))
    body = text[m.end() :]
    pts: list[tuple[float, float, float]] = []
    for m2 in re.finditer(r"\(([^)]+)\)", body):
        nums = m2.group(1).split()
        if len(nums) >= 3:
            pts.append((float(nums[0]), float(nums[1]), float(nums[2])))
        if len(pts) >= n:
            break
    return np.asarray(pts, dtype=float)


def _read_foam_faces(mesh_dir: Path) -> list[list[int]]:
    text = (mesh_dir / "faces").read_text(encoding="utf-8", errors="replace")
    m = re.search(r"FoamFile\s*\{.*?\}\s*", text, flags=re.S)
    if m:
        text = text[m.end() :]
    m = re.search(r"(\d+)\s*\(", text)
    if not m:
        raise RuntimeError("polyMesh faces: no count")
    n = int(m.group(1))
    body = text[m.end() :]
    faces: list[list[int]] = []
    for m2 in re.finditer(r"(\d+)\(([^)]+)\)", body):
        faces.append([int(x) for x in m2.group(2).split()])
        if len(faces) >= n:
            break
    return faces


def write_mesh_preview_png(
    path: Path,
    job: dict[str, Any],
    spec: BladeSpec,
    mesh: MeshBuild,
) -> None:
    """Gmsh/NASA-style polyMesh wire: 2D cell quads from empty frontAndBack faces.

    Not the cascade metal outline. Metal is a solid cutout; fluid shows every
    polygon edge. Reads case_dir/constant/polyMesh next to path.
    """
    # Freeze B: tile 1-pitch polyMesh wire × viz_stack for cassette look (no remesh).
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection, PolyCollection
    except Exception:
        return

    path = Path(path)
    case_dir = path.parent
    try:
        from .job import pitch_m as _pitch_m
        _pitch = float(_pitch_m(job))
    except Exception:
        _pitch = float(getattr(mesh, "pitch_m", None) or 0.01)
    _viz_n = int(job.get("_viz_stack_blades") or (job.get("geometry") or {}).get("n_blades_cascade") or 1)
    _viz_n = max(_viz_n, 1)
    _ = spec
    mesh_dir = case_dir / "constant" / "polyMesh"
    if not (mesh_dir / "points").is_file() or not (mesh_dir / "faces").is_file():
        return

    try:
        pts = _read_foam_points(mesh_dir)
        faces = _read_foam_faces(mesh_dir)
        patches = _parse_boundary_patches(mesh_dir)
    except Exception:
        return

    fab = patches.get("frontAndBack")
    if not fab:
        return
    start, nfaces = fab
    z_all = pts[:, 2]
    z_front = float(np.min(z_all))
    z_tol = max(1e-12, 1e-6 * (float(np.max(z_all)) - z_front + 1e-16))

    palette = ["#5b4b8a", "#3d6b7a", "#6b5e2e", "#6a3d5c", "#2f5d4a", "#4a4e6b"]
    segs: list[np.ndarray] = []
    polys_xy: list[np.ndarray] = []
    colors: list[str] = []
    end = min(start + nfaces, len(faces))
    for fi in range(start, end):
        fv = faces[fi]
        if len(fv) < 3:
            continue
        xyz = pts[fv]
        if float(np.mean(xyz[:, 2])) > z_front + z_tol:
            continue
        xy = xyz[:, :2] * 1000.0
        polys_xy.append(xy)
        closed = np.vstack([xy, xy[:1]])
        for i in range(len(xy)):
            segs.append(closed[i : i + 2])
        cx, cy = float(xy[:, 0].mean()), float(xy[:, 1].mean())
        colors.append(palette[int(abs(hash((round(cx, 1), round(cy, 1)))) % len(palette))])

    fig, ax = plt.subplots(figsize=(6.2, 8.0), dpi=140)
    ax.set_facecolor("#0b0b0f")
    fig.patch.set_facecolor("#0b0b0f")

    # Tile fluid wire + metals in y by k*pitch for UI cassette (polyMesh stays 1-pitch).
    pitch_mm = _pitch * 1000.0
    tiled_polys: list[np.ndarray] = []
    tiled_colors: list[str] = []
    tiled_segs: list[np.ndarray] = []
    for k in range(_viz_n):
        dy = k * pitch_mm
        for xy, col in zip(polys_xy, colors):
            shifted = xy.copy()
            shifted[:, 1] = shifted[:, 1] + dy
            tiled_polys.append(shifted)
            tiled_colors.append(col)
        for sg in segs:
            s2 = sg.copy()
            s2[:, 1] = s2[:, 1] + dy
            tiled_segs.append(s2)
    if tiled_polys:
        ax.add_collection(
            PolyCollection(tiled_polys, facecolors=tiled_colors, edgecolors="none", alpha=0.35, zorder=1)
        )
    if tiled_segs:
        ax.add_collection(
            LineCollection(tiled_segs, colors="#d8d4e8", linewidths=0.22, alpha=0.85, zorder=2)
        )

    metal_face = "#1a1a22"
    metal_edge = "#e8e6f2"
    base_polys = list(mesh.blade_polys or [])
    if len(base_polys) == 1 and _viz_n > 1:
        bp0 = base_polys[0]
        base_polys = [[(p[0], p[1] + k * _pitch) for p in bp0] for k in range(_viz_n)]
    for k, poly in enumerate(base_polys):
        xs = [p[0] * 1000 for p in poly]
        ys = [p[1] * 1000 for p in poly]
        ax.fill(
            xs,
            ys,
            color=metal_face,
            ec=metal_edge,
            lw=0.9,
            zorder=3,
            label=("metal" if k == 0 else None),
        )

    for k in range(_viz_n):
        ax.axhline((mesh.y_min + k * _pitch) * 1000, color="#6e6a82", ls="--", lw=0.5, zorder=4)
        ax.axhline((mesh.y_max + k * _pitch) * 1000, color="#6e6a82", ls="--", lw=0.5, zorder=4)
    ax.set_aspect("equal")
    ax.autoscale()
    # Crop dump length — NASA/Gmsh blade-to-blade view, not the full outlet H-block.
    if base_polys:
        mx = [p[0] * 1000 for poly in base_polys for p in poly]
        my = [p[1] * 1000 for poly in base_polys for p in poly]
        x0, x1 = min(mx), max(mx)
        y0, y1 = min(my), max(my)
        c = max(x1 - x0, 1.0)
        ax.set_xlim(x0 - 0.55 * c, x1 + 1.15 * c)
        pad = 0.12 * max(y1 - y0, c)
        ax.set_ylim(y0 - pad, y1 + pad)
    ax.set_xlabel("x axial [mm]  → flow")
    ax.set_ylabel("y pitch [mm]  ↑ stack")
    ax.set_title(
        f"polyMesh wire · {mesh.mesh_kind} · n_cells={mesh.n_cells} · "
        f"front faces={len(polys_xy)}"
        + (f" · tri={mesh.n_tri} quad={mesh.n_quad}" if mesh.n_tri else ""),
        fontsize=9,
        color="#e8e6f2",
    )
    for spine in ax.spines.values():
        spine.set_color("#9b97b0")
    ax.tick_params(colors="#e8e6f2", labelsize=8)
    ax.xaxis.label.set_color("#e8e6f2")
    ax.yaxis.label.set_color("#e8e6f2")
    if mesh.blade_polys:
        ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="#0b0b0f", edgecolor="none")
    plt.close(fig)
