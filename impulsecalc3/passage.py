"""Goldman/Katsanis blade-to-blade passage O+H.

Lower wall = SS of blade 0. Upper wall = PS of blade 0 translated by s.
Not a y-strip that owns one C. Translation (x, y) ~ (x, y+s), not a mirror.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .geometry import split_ps_ss


def _lin(p0, p1, n_seg: int):
    p0 = np.asarray(p0, dtype=float).reshape(2)
    p1 = np.asarray(p1, dtype=float).reshape(2)
    tt = np.linspace(0.0, 1.0, max(int(n_seg), 1) + 1)[:, None]
    return (1.0 - tt) * p0[None, :] + tt * p1[None, :]


def _stretch(j: int, n: int, r: float) -> float:
    if n <= 0:
        return 0.0
    if abs(r - 1.0) < 1e-9:
        return j / n
    return (r ** j - 1.0) / (r ** n - 1.0)


def _resample(chain: list[tuple[float, float]] | np.ndarray, n_seg: int) -> np.ndarray:
    pts = np.asarray(chain, dtype=float)
    if n_seg < 1 or len(pts) < 2:
        return pts.copy()
    s = [0.0]
    for i in range(1, len(pts)):
        s.append(s[-1] + float(np.hypot(pts[i, 0] - pts[i - 1, 0], pts[i, 1] - pts[i - 1, 1])))
    total = s[-1] if s[-1] > 0 else 1.0
    out = []
    for k in range(n_seg + 1):
        target = total * k / n_seg
        if target <= 0:
            out.append(pts[0])
            continue
        if target >= total:
            out.append(pts[-1])
            continue
        j = 0
        while j < len(s) - 1 and s[j + 1] < target:
            j += 1
        span = s[j + 1] - s[j] or 1e-16
        t = (target - s[j]) / span
        out.append((1.0 - t) * pts[j] + t * pts[j + 1])
    return np.asarray(out, dtype=float)



def _y_at_x(chain: np.ndarray, x: float) -> float:
    xs = chain[:, 0]
    if x <= float(xs.min()):
        i = int(np.argmin(xs))
        return float(chain[i, 1])
    if x >= float(xs.max()):
        i = int(np.argmax(xs))
        return float(chain[i, 1])
    for i in range(len(chain) - 1):
        x0, x1 = float(chain[i, 0]), float(chain[i + 1, 0])
        if (x0 - x) * (x1 - x) <= 0.0:
            if abs(x1 - x0) < 1e-16:
                return float(chain[i, 1])
            tt = (x - x0) / (x1 - x0)
            return float(chain[i, 1] + tt * (chain[i + 1, 1] - chain[i, 1]))
    return float(chain[int(np.argmin(np.abs(xs - x))), 1])


def _pair_by_x(south: np.ndarray, north: np.ndarray, n_st: int) -> tuple[np.ndarray, np.ndarray]:
    raise RuntimeError(
        "_pair_by_x is off the load path (ORBIT 2026-09-03). A C is not y(x). "
        "Goldman channel is SS0(s) vs PS0(s)+s by arc length."
    )
    x0 = max(float(south[:, 0].min()), float(north[:, 0].min()))
    x1 = min(float(south[:, 0].max()), float(north[:, 0].max()))
    xs = np.linspace(x0, x1, n_st + 1)
    s = np.column_stack([xs, np.array([_y_at_x(south, x) for x in xs])])
    n = np.column_stack([xs, np.array([_y_at_x(north, x) for x in xs])])
    return s, n


def offset_open(chain: np.ndarray, dist: float, toward: np.ndarray) -> np.ndarray:
    """Offset an open chain. `toward` is a sample interior point used to pick the fluid side."""
    n = chain.shape[0]
    normals = np.zeros_like(chain)
    for i in range(n):
        i0 = max(i - 1, 0)
        i1 = min(i + 1, n - 1)
        e = chain[i1] - chain[i0]
        L = float(np.hypot(e[0], e[1])) or 1e-16
        # right-handed normal of the walk
        nrm = np.array([e[1] / L, -e[0] / L])
        mid = 0.5 * (chain[max(i - 1, 0)] + chain[min(i + 1, n - 1)])
        to_in = toward - mid
        if float(nrm[0] * to_in[0] + nrm[1] * to_in[1]) < 0:
            nrm = -nrm
        normals[i] = nrm
    for _ in range(6):
        sm = normals.copy()
        sm[1:-1] = 0.5 * normals[1:-1] + 0.25 * normals[:-2] + 0.25 * normals[2:]
        ln = np.linalg.norm(sm, axis=1, keepdims=True)
        normals = sm / np.clip(ln, 1e-16, None)
    return chain + float(dist) * normals


def _layers(wall: np.ndarray, outer: np.ndarray, n_rad: int, stretch: float) -> np.ndarray:
    pts = np.zeros((wall.shape[0], n_rad + 1, 2), dtype=float)
    for j in range(n_rad + 1):
        t = _stretch(j, n_rad, stretch)
        pts[:, j, :] = (1.0 - t) * wall + t * outer
    return pts


def _extend_x(chain: np.ndarray, x_in: float, x_out: float, n_in: int, n_out: int) -> np.ndarray:
    p0, p1 = chain[0], chain[-1]
    head = _lin((x_in, float(p0[1])), p0, max(n_in, 2))[:-1]
    tail = _lin(p1, (x_out, float(p1[1])), max(n_out, 2))[1:]
    return np.concatenate([head, chain, tail], axis=0)


@dataclass
class PassageGrid:
    o_south: np.ndarray
    o_north: np.ndarray
    h_core: np.ndarray
    h_inlet: np.ndarray
    h_outlet: np.ndarray
    south_poly: list[tuple[float, float]]
    north_poly: list[tuple[float, float]]
    x_in: float
    x_out: float
    y_min: float
    y_max: float
    d_o: float
    first_cell_m: float
    min_area_2d: float
    notes: list[str] = field(default_factory=list)
    cyclic: bool = False


def build_passage_oh(
    poly: list[tuple[float, float]],
    *,
    pitch: float,
    x_in: float,
    x_out: float,
    n_in: int,
    n_out: int,
    n_stream: int,
    n_span: int,
    n_rad: int,
    stretch: float,
    d_o: float,
) -> PassageGrid:
    """SS0 vs PS0+s channel. Cyclics omitted unless inlet height is a pitch."""
    from .mesh import _pos_block, min_cell_area_2d_rect, tfi_block

    notes: list[str] = []
    ps, ss, _, _ = split_ps_ss(poly)
    south = np.array(ss, dtype=float)
    north = np.array([(p[0], p[1] + float(pitch)) for p in ps], dtype=float)
    # Fluid must sit between them. If north is below south, the labels were swapped vs +s.
    imid = min(len(south), len(north)) // 2
    if float(north[min(imid, len(north) - 1), 1]) < float(south[min(imid, len(south) - 1), 1]):
        south, north = north, south
        notes.append("passage walls swapped so +s is the upper surface")

    n_st = max(int(n_stream), 24)
    south_m, north_m = _pair_by_x(south, north, n_st)
    gap = np.linalg.norm(north_m - south_m, axis=1)
    gmin = float(gap.min())
    if gmin <= 1e-7:
        raise RuntimeError(
            f"passage walls meet (min gap {gmin:.3g} m). Solids intersect; not nesting."
        )
    d_use = min(float(d_o), 0.32 * gmin)
    if d_use < float(d_o) - 1e-16:
        notes.append(f"d_o {d_o:.3g} → {d_use:.3g} m so collars clear the channel")

    n_eta = max(int(n_rad) + int(n_span), 10)
    h_core = np.zeros((south_m.shape[0], n_eta + 1, 2), dtype=float)
    for j in range(n_eta + 1):
        # cosine clustering at both walls (O-like first cell without a second block)
        t = 0.5 - 0.5 * math.cos(math.pi * j / n_eta)
        h_core[:, j, :] = (1.0 - t) * south_m + t * north_m
    d_use = float(np.mean(np.linalg.norm(h_core[:, 1, :] - h_core[:, 0, :], axis=1)))

    n_sp = n_eta
    s_head = _lin((x_in, float(south_m[0, 1])), south_m[0], n_in)
    n_head = s_head.copy()
    n_head[:, 1] = s_head[:, 1] + float(pitch)
    # inlet east MUST be the core west (same nodes)
    east_in = h_core[0, :, :]
    west_in = _lin(s_head[0], n_head[0], n_sp)
    if east_in.shape[0] != west_in.shape[0]:
        west_in = _lin(s_head[0], n_head[0], east_in.shape[0] - 1)
        n_sp = east_in.shape[0] - 1
        s_head = _lin((x_in, float(south_m[0, 1])), south_m[0], n_in)
        n_head = s_head.copy()
        n_head[:, 1] = s_head[:, 1] + float(pitch)
    h_inlet = tfi_block(s_head, n_head, west_in, east_in)

    s_tail = _lin(south_m[-1], (x_out, float(south_m[-1, 1])), n_out)
    n_tail = s_tail.copy()
    n_tail[:, 1] = s_tail[:, 1] + float(pitch)
    west_out = h_core[-1, :, :]
    east_out = _lin(s_tail[-1], n_tail[-1], west_out.shape[0] - 1)
    h_outlet = tfi_block(s_tail, n_tail, west_out, east_out)

    o_s = h_core[:, :2, :]
    o_n = h_core[:, -2:, :]

    dy_in = abs(float(n_head[0, 1] - s_head[0, 1]))
    cyclic = abs(dy_in - float(pitch)) / max(float(pitch), 1e-12) < 0.08
    if cyclic:
        notes.append(f"inlet height {dy_in*1e3:.2f} mm ≈ s; leftover cuts tagged cyclic")
    else:
        notes.append(
            f"inlet height {dy_in*1e3:.2f} mm ≠ s={pitch*1e3:.2f} mm — leftover cuts are "
            "inlet/outlet duct, not cyclic (LE thickness). Channel is the Goldman object."
        )

    first = float(np.mean(np.linalg.norm(h_core[:, 1, :] - h_core[:, 0, :], axis=1)))
    areas = [
        min_cell_area_2d_rect(o_s),
        min_cell_area_2d_rect(o_n),
        min_cell_area_2d_rect(h_core),
        min_cell_area_2d_rect(h_inlet),
        min_cell_area_2d_rect(h_outlet),
    ]
    amin = min(areas)
    ys = np.concatenate([south_m[:, 1], north_m[:, 1], np.array([x_in * 0, x_out * 0])])
    y_min = float(min(south_m[:, 1].min(), north_m[:, 1].min(), s_head[:, 1].min(), n_head[:, 1].min()))
    y_max = float(max(south_m[:, 1].max(), north_m[:, 1].max(), s_head[:, 1].max(), n_head[:, 1].max()))
    notes.append(
        f"passage O+H: SS0 vs PS0+s, min gap {gmin*1e3:.2f} mm, d_o {d_use:.3g} m, "
        f"n_stream={n_st} n_span={n_sp} n_rad={n_rad}"
    )
    notes.append("NOT a pitch rectangle. NOT a mirror. NOT Gmsh.")
    return PassageGrid(
        o_south=o_s,
        o_north=o_n,
        h_core=h_core,
        h_inlet=h_inlet,
        h_outlet=h_outlet,
        south_poly=[(float(x), float(y)) for x, y in south_m],
        north_poly=[(float(x), float(y)) for x, y in north_m],
        x_in=float(x_in),
        x_out=float(x_out),
        y_min=y_min,
        y_max=y_max,
        d_o=d_use,
        first_cell_m=first,
        min_area_2d=amin,
        notes=notes,
        cyclic=cyclic,
    )
