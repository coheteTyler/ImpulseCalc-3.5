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
    ks = np.where(ys >= 0.92 * ymax)[0]
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


def _pos_block(hb: np.ndarray, name: str) -> np.ndarray:
    """Orientation with strictly positive min 2D area. No pinched-quad waiver.

    Do not Laplacian-untangle a block that is already positive: thin north
    layers (offset back almost on the cyclic) invert under that smoother.
    """
    amin0 = min_cell_area_2d_rect(hb)
    if amin0 > 0:
        return hb
    hb = _untangle_block(hb)
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
) -> tuple[np.ndarray, list[np.ndarray], float, list[str]]:
    """Offset O-collar + cavity TFI + outer TFI H-blocks to the pitch rectangle.

    Long pointed stems make AABB-morph O-cells cross the U (duplicate verts,
    zero-area faces, 1e145 skew). The cavity is its own H-block; outer H-blocks
    only see the convex back and the mouth chord. Cyclic top/bottom share x.
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
    kNW, kNE = spl["kNW"], spl["kNE"]
    kE, kW = spl["kE"], spl["kW"]
    kE_top, kW_top = spl["kE_top"], spl["kW_top"]
    notes.append(
        f"U-cavity depth {spl['depth']*1e3:.2f} mm / y-span {spl['yspan']*1e3:.2f} mm; "
        f"cavity TFI + offset collar (not AABB morph across the stems)"
    )

    n_stem = max(int(n_in), int(n_cyc), 12)
    n_cav_x = int(n_cyc)
    n_cav_y = n_stem
    n_up = max(int(n_fill), 4)
    le_r = max(float(le_cluster), 1.0)
    # Local per-arc x-range: global blade x makes the whole west stem look "near LE"
    # (flat weights). Local range densifies toward min-x *on that arc*.
    if le_r > 1.0 + 1e-12:
        notes.append(
            f"streamwise LE cluster le_cluster={le_r:.3g} on wall/offset arcs "
            "(Δs biased to min-x per arc; O wall-normal stretch unchanged)"
        )

    def _side(ring, idx, nseg):
        return _resample_xy(idx, ring, nseg, le_r)

    cav_w = _side(outer_hi, inner_idx[: kNW + 1], n_cav_y)
    cav_n = _side(outer_hi, inner_idx[kNW : kNE + 1], n_cav_x)
    cav_e_n2s = _side(outer_hi, inner_idx[kNE :], n_cav_y)
    cav_e = cav_e_n2s[::-1].copy()
    east_o = _side(outer_hi, outer_idx[: kE + 1], n_out)                 # Rt → stem NE
    east_up = _side(outer_hi, outer_idx[kE : kE_top + 1], n_up)          # stem NE → top NE
    north_o = _side(outer_hi, outer_idx[kE_top : kW_top + 1], n_cyc)     # top NE → top NW
    west_up = _side(outer_hi, outer_idx[kW_top : kW + 1], n_up)          # top NW → stem NW
    west_o = _side(outer_hi, outer_idx[kW :], n_in)                      # stem NW → Lt

    met_w = _side(inner_hi, inner_idx[: kNW + 1], n_cav_y)
    met_n = _side(inner_hi, inner_idx[kNW : kNE + 1], n_cav_x)
    met_e = _side(inner_hi, inner_idx[kNE :], n_cav_y)[::-1].copy()
    met_east_o = _side(inner_hi, outer_idx[: kE + 1], n_out)
    met_east_up = _side(inner_hi, outer_idx[kE : kE_top + 1], n_up)
    met_north_o = _side(inner_hi, outer_idx[kE_top : kW_top + 1], n_cyc)
    met_west_up = _side(inner_hi, outer_idx[kW_top : kW + 1], n_up)
    met_west_o = _side(inner_hi, outer_idx[kW :], n_in)

    inner = np.concatenate(
        [
            met_w[:-1], met_n[:-1], met_e[::-1][:-1],
            met_east_o[:-1], met_east_up[:-1], met_north_o[:-1], met_west_up[:-1], met_west_o[:-1],
        ],
        axis=0,
    )
    outer = np.concatenate(
        [
            cav_w[:-1], cav_n[:-1], cav_e_n2s[:-1],
            east_o[:-1], east_up[:-1], north_o[:-1], west_up[:-1], west_o[:-1],
        ],
        axis=0,
    )
    if inner.shape[0] != outer.shape[0]:
        raise RuntimeError(f"U offset-O ring mismatch inner={inner.shape[0]} outer={outer.shape[0]}")

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

    pLt = cav_w[0]
    pRt = cav_e[0]
    pNW_o = west_o[0]          # stem NW
    pNE_o = east_o[-1]         # stem NE
    pNW_top = north_o[-1]
    pNE_top = north_o[0]
    west_s2n = west_o[::-1].copy()
    east_s2n = east_o
    west_up_s2n = west_up[::-1].copy()  # stem NW → top NW
    east_up_s2n = east_up               # stem NE → top NE
    north_ltr = north_o[::-1].copy()    # top NW → top NE
    mouth = _lin(pLt, pRt, n_cyc)
    cav_south = mouth
    cav_north = cav_n
    cav_west = cav_w
    cav_east = cav_e
    if cav_north.shape[0] != cav_south.shape[0] or cav_west.shape[0] != cav_east.shape[0]:
        raise RuntimeError("cavity TFI edge mismatch")
    h_cav_raw = tfi_block(cav_south, cav_north, cav_west, cav_east)
    if min_cell_area_2d_rect(h_cav_raw) <= 0:
        h_cav_raw = tfi_block(cav_south, cav_north, cav_east, cav_west)
        notes.append("cavity TFI: swapped east/west to clear inverted quads")
    h_cav = _pos_block(h_cav_raw, "cavity")

    x_join_w = float(pLt[0])
    x_join_e = float(pRt[0])
    # Inlet H axial: pack toward LE join (high-x), mirror dump stretch toward blade.
    r_in = _inlet_pack_r(n_in, r_inlet)
    xs_w = _xs_pack_hi(x_in, x_join_w, n_in, r_in)
    xs_s = np.linspace(x_join_w, x_join_e, n_cyc + 1)
    notes.append(
        f"inlet H axial pack toward LE (inlet_stretch={r_inlet:.3g}→{r_in:.3g}, n_inlet={n_in}); "
        "Δx smaller near LE join than at x_in"
    )
    x_cart = min(x_out - 1e-6, max(float(outer_hi[:, 0].max()), x_join_e) + 2e-4)
    xs_near = np.linspace(x_join_e, x_cart, n_out + 1)
    xs_dump = np.linspace(x_cart, x_out, n_out_x + 1)
    xs_e = xs_near
    cyc_s = np.column_stack([xs_s, np.full(xs_s.shape[0], y_bot)])
    cyc_n = np.column_stack([xs_s, np.full(xs_s.shape[0], y_top)])
    join_w_top = np.array([x_join_w, y_top], dtype=float)
    join_e_top = np.array([x_join_e, y_top], dtype=float)
    join_w_bot = np.array([x_join_w, y_bot], dtype=float)
    join_e_bot = np.array([x_join_e, y_bot], dtype=float)

    # Shared wake interface: cavity uses mouth Lt→Rt as south; south H uses the
    # same point sequence as its north edge (conformal). Face-normal repair later
    # orients owner→neighbour; do not reverse here or nodes will duplicate.
    h_south = _pos_block(
        tfi_block(cyc_s, mouth, _lin(join_w_bot, pLt, n_fill), _lin(join_e_bot, pRt, n_fill)),
        "south",
    )
    h_north = _pos_block(
        tfi_block(north_ltr, cyc_n, _lin(pNW_top, join_w_top, n_fill), _lin(pNE_top, join_e_top, n_fill)),
        "north",
    )
    h_west = _pos_block(
        tfi_block(
            _lin_pack_end((x_in, pLt[1]), pLt, n_in, r_in),
            _lin_pack_end((x_in, pNW_o[1]), pNW_o, n_in, r_in),
            np.column_stack([np.full(west_s2n.shape[0], x_in), west_s2n[:, 1]]),
            west_s2n,
        ),
        "west",
    )
    h_west_up = _pos_block(
        tfi_block(
            _lin_pack_end((x_in, pNW_o[1]), pNW_o, n_in, r_in),
            _lin_pack_end((x_in, pNW_top[1]), pNW_top, n_in, r_in),
            np.column_stack([np.full(west_up_s2n.shape[0], x_in), west_up_s2n[:, 1]]),
            west_up_s2n,
        ),
        "west_up",
    )
    h_east = _pos_block(
        tfi_block(
            _lin(pRt, (x_cart, pRt[1]), n_out),
            _lin(pNE_o, (x_cart, pNE_o[1]), n_out),
            east_s2n,
            np.column_stack([np.full(east_s2n.shape[0], x_cart), east_s2n[:, 1]]),
        ),
        "east",
    )
    h_east_up = _pos_block(
        tfi_block(
            _lin(pNE_o, (x_cart, pNE_o[1]), n_out),
            _lin(pNE_top, (x_cart, pNE_top[1]), n_out),
            east_up_s2n,
            np.column_stack([np.full(east_up_s2n.shape[0], x_cart), east_up_s2n[:, 1]]),
        ),
        "east_up",
    )
    h_sw = _pos_block(
        tfi_block(
            np.column_stack([xs_w, np.full(xs_w.shape[0], y_bot)]),
            _lin_pack_end((x_in, pLt[1]), pLt, n_in, r_in),
            _lin((x_in, y_bot), (x_in, pLt[1]), n_fill),
            _lin(join_w_bot, pLt, n_fill),
        ),
        "sw",
    )
    h_se = _pos_block(
        tfi_block(
            np.column_stack([xs_e, np.full(xs_e.shape[0], y_bot)]),
            np.column_stack([xs_e, np.full(xs_e.shape[0], pRt[1])]),
            _lin(join_e_bot, pRt, n_fill),
            _lin((x_cart, y_bot), (x_cart, pRt[1]), n_fill),
        ),
        "se",
    )
    h_nw = _pos_block(
        tfi_block(
            _lin_pack_end((x_in, pNW_top[1]), pNW_top, n_in, r_in),
            np.column_stack([xs_w, np.full(xs_w.shape[0], y_top)]),
            _lin((x_in, pNW_top[1]), (x_in, y_top), n_fill),
            _lin(pNW_top, join_w_top, n_fill),
        ),
        "nw",
    )
    h_ne = _pos_block(
        tfi_block(
            _lin(pNE_top, (x_cart, pNE_top[1]), n_out),
            np.column_stack([xs_e, np.full(xs_e.shape[0], y_top)]),
            _lin(pNE_top, join_e_top, n_fill),
            _lin((x_cart, pNE_top[1]), (x_cart, y_top), n_fill),
        ),
        "ne",
    )
    ys_dump = np.concatenate(
        [
            np.linspace(y_bot, float(pRt[1]), n_fill + 1)[:-1],
            east_s2n[:-1, 1],
            east_up_s2n[:-1, 1],
            np.linspace(float(pNE_top[1]), y_top, n_fill + 1),
        ]
    )
    h_dump = _pos_block(cartesian_block_xy(xs_dump, ys_dump), "dump")
    snap = [(te_w_old, te_w), (te_o_old, te_o)]
    h_rest = [_snap_points(hb, snap) for hb in [h_cav, h_west, h_west_up, h_east, h_east_up, h_south, h_north, h_sw, h_se, h_nw, h_ne, h_dump]]
    h_blocks = [h_te_wake, *h_rest]
    notes.append("H-blocks: cavity TFI inside the U; outer TFI to the pitch rectangle.")
    notes.append("H TE nodes snapped to wake chord (unkinked).")
    notes.append("Cyclic x-nodes are shared top/bottom so 3-pitch stacking is conformal.")
    notes.append("NOT subsetMesh stairs. NOT AABB morph across the cavity. NOT Gmsh.")
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
) -> dict:
    """Three physical C's in one polyMesh. Fluid outside every C. No cyclic. No Gmsh."""
    notes: list[str] = []
    blades = [_shift_poly(poly0, k * pitch) for k in range(n_blades)]
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
    lid = max(0.5 * d_use, 0.0005)
    y_bot = all_y0 - lid
    y_top = all_y1 + lid
    h_blocks: list[np.ndarray] = []
    n_st = max(n_cyc, 24)
    n_span = max(n_fill, 8)
    for k in range(n_blades - 1):
        _ps_a, ss_a = _ring_ps_ss(outers[k])
        ps_b, _ss_b = _ring_ps_ss(outers[k + 1])
        h_blocks.append(_passage_tfi(ss_a, ps_b, n_st, n_span, f"pass{k}"))
        notes.append(f"passage TFI blade{k} SS-offset vs blade{k+1} PS-offset (arc length)")
        # inlet / outlet of this channel
        s0 = resample_open_arclength(ss_a, n_st + 1)
        n0 = resample_open_arclength(ps_b, n_st + 1)
        s_head = _lin_pack_end((x_in, float(s0[0, 1])), s0[0], n_in, max(stretch, 1.0))
        n_head = _lin_pack_end((x_in, float(n0[0, 1])), n0[0], n_in, max(stretch, 1.0))
        west_in = _lin(s_head[0], n_head[0], n_span)
        east_in = _lin(s0[0], n0[0], n_span)
        h_blocks.append(_pos_block(tfi_block(s_head, n_head, west_in, east_in), f"in{k}"))
        s_tail = _lin(s0[-1], (x_out, float(s0[-1, 1])), n_out_x)
        n_tail = _lin(n0[-1], (x_out, float(n0[-1, 1])), n_out_x)
        west_out = _lin(s0[-1], n0[-1], n_span)
        east_out = _lin(s_tail[-1], n_tail[-1], n_span)
        h_blocks.append(_pos_block(tfi_block(s_tail, n_tail, west_out, east_out), f"out{k}"))
    # floor under blade0 PS-offset; lid above last SS-offset
    ps0, ss0 = _ring_ps_ss(outers[0])
    _psN, ssN = _ring_ps_ss(outers[-1])
    ps0r = resample_open_arclength(ps0, n_st + 1)
    ssNr = resample_open_arclength(ssN, n_st + 1)
    floor_s = np.column_stack([ps0r[:, 0], np.full(ps0r.shape[0], y_bot)])
    # keep floor x in domain
    floor_s[:, 0] = np.clip(floor_s[:, 0], x_in, x_out)
    h_blocks.append(_passage_tfi(floor_s, ps0r, n_st, max(n_fill, 4), "floor"))
    lid_n = np.column_stack([ssNr[:, 0], np.full(ssNr.shape[0], y_top)])
    lid_n[:, 0] = np.clip(lid_n[:, 0], x_in, x_out)
    h_blocks.append(_passage_tfi(ssNr, lid_n, n_st, max(n_fill, 4), "lid"))
    # cavity of the TOP C only (lower U's hold the neighbor)
    spl = _down_u_splits(outers[-1])
    if spl is not None:
        inner_hi = resample_closed(blades[-1], max(4 * (2 * n_cyc + n_in + n_out), 200))
        if polygon_signed_area([(float(x), float(y)) for x, y in inner_hi]) < 0:
            inner_hi = inner_hi[::-1].copy()
        off_hi = outers[-1]
        # cavity on the offset ring of the last blade; splits are indices into that ring
        try:
            n_stem = max(int(n_in), int(n_cyc), 12)
            inner_idx = spl["inner"]
            kNW, kNE = spl["kNW"], spl["kNE"]
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
    notes.append("mesh_kind cassette_OH: 3 closed metals, fluid outside every C, lid/floor WALL not cyclic.")
    notes.append("NOT Gmsh. NOT y(x) passage_OH. NOT subsetMesh stairs.")
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
    # hybrid_OH_tri: one pitch, true cyclic — do not stack 3 premeshed blades.
    if use_hybrid:
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
    use_passage = False
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
    # Hybrid refuses cassette stacking (one-pitch strip only).
    if (not use_hybrid) and yspan + 2.0 * d_o_gate >= float(pitch):
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
    if passage is None and cas is None and use_cavity and use_hybrid:
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
        elif abs(x - x_in) < 2e-4:
            buckets["inlet"].append((fverts, owner, c))
        elif abs(x - x_out) < 2e-4:
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
        elif (passage is None) and abs(y - y_min) < 1e-7:
            buckets["bottom"].append((fverts, owner, c))
        elif (passage is None) and abs(y - y_max) < 1e-7:
            buckets["top"].append((fverts, owner, c))
        else:
            if passage is not None:
                ymid = 0.5 * (passage.y_min + passage.y_max)
                name = "blade0" if y < ymid else "blade1"
                buckets[name].append((fverts, owner, c))
            elif cas is not None:
                ymid = 0.5 * (y_min + y_max)
                name = "bottom" if y < ymid else "top"
                buckets[name].append((fverts, owner, c))
            else:
                unclassified += 1
    if unclassified:
        raise RuntimeError(f"{unclassified} boundary faces not on wall/inlet/outlet/cyclic/empty")
    if any(not buckets[n] for n in blade_names):
        raise RuntimeError("per-blade wall patches missing faces")
    if cas is None and (buckets["bottom"] or buckets["top"]):
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

    if hybrid_tris is not None and buckets["bottom"] and buckets["top"]:
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
    sep = n_blades * pitch

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
            if cas is not None:
                ptype = "wall"
                extra = "        inGroups        1(wall);\n"
            else:
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
        kind_note = "mesh: cassette_OH — three closed C metals, fluid outside every C, lid/floor walls."
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
        f"{n_blades} blades, pitch {pitch:.6g} m, empty frontAndBack, slab {zth} m.",
        f"O n_around={n_i} (JSON n_around={n_around_req}; S/N=n_cyclic={n_cyc} E=n_outlet={n_out} W=n_inlet={n_in}) n_radial={n_rad}",
        f"n_cells={n_cells} first_cell≈{first_cell:.3g} m d_o={d_o:.3g} m min O-quad {a2:.3e} m2",
        f"y_shift to centre blade in pitch: {y_shift:.6g} m (rigid; metal angles unchanged).",
        "Wall faces tagged from the O-grid j=0 ring (per-blade patches).",
        *oh_notes,
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
    del job, spec  # preview is polyMesh-driven; metal cutouts from MeshBuild
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection, PolyCollection
    except Exception:
        return

    path = Path(path)
    case_dir = path.parent
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

    if polys_xy:
        ax.add_collection(
            PolyCollection(polys_xy, facecolors=colors, edgecolors="none", alpha=0.35, zorder=1)
        )
    if segs:
        ax.add_collection(
            LineCollection(segs, colors="#d8d4e8", linewidths=0.22, alpha=0.85, zorder=2)
        )

    metal_face = "#1a1a22"
    metal_edge = "#e8e6f2"
    for k, poly in enumerate(mesh.blade_polys or []):
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

    ax.axhline(mesh.y_min * 1000, color="#6e6a82", ls="--", lw=0.5, zorder=4)
    ax.axhline(mesh.y_max * 1000, color="#6e6a82", ls="--", lw=0.5, zorder=4)
    ax.set_aspect("equal")
    ax.autoscale()
    # Crop dump length — NASA/Gmsh blade-to-blade view, not the full outlet H-block.
    if mesh.blade_polys:
        mx = [p[0] * 1000 for poly in mesh.blade_polys for p in poly]
        my = [p[1] * 1000 for poly in mesh.blade_polys for p in poly]
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
