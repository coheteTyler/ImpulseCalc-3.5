"""Curved-periodic O–H mesher for nested point-defined blades (profile_points).

When the blade y-span is >= pitch (nested cascade), straight y = ±s/2 cyclics
cut through metal and the classic body_fitted_OH path clips the blade (Gate 0).
This path keeps the metal exact and moves the cyclic instead:

* Periodic line T(x): equidistant (medial) curve between blade k and blade k+1
  (= blade k + (0, s)), smoothed, and blended to a flat line upstream of the LE
  and downstream of the TE (flat → hump → flat). Bottom cyclic = T − s exactly,
  so bottom/top are node-for-node translates: translational cyclic, not AMI.
* Blocks (all hex after extrusion, no triangles):
    U  upper passage: straight wall-normal lines from blade-k outer wall to T
    D  lower passage: straight wall-normal lines from blade-k inner wall to T−s
    NL / NT  nose O pieces around the LE / TE (wall-normal offsets, same layering)
    W  inlet block (x_in → LE composite), E  outlet block (TE composite → x_out)
  W/E are TFI + Laplacian smoothed with periodic rows sliding along T (paired).
* Wall layering: y1 (from yplus_target), growth (cfd.stretch), n_radial layers,
  then extension layers growing to the passage cell size.

Writer: explicit 2-D quads → 1-cell-thick hex slab, patches inlet / outlet /
bottom / top (cyclic translational ±pitch) / frontAndBack (empty) / blade0 (wall).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

PATCH_ORDER = ("inlet", "outlet", "bottom", "top", "frontAndBack", "blade0")


# --------------------------------------------------------------------------- writer
def _hdr(cls: str, obj: str, note: str = "") -> str:
    n = f'    note        "{note}";\n' if note else ""
    return (
        "FoamFile\n{\n    version     2.0;\n    format      ascii;\n"
        f"    class       {cls};\n{n}    location    \"constant/polyMesh\";\n"
        f"    object      {obj};\n}}\n"
        "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"
    )


def write_polymesh_2d(
    out_case: Path,
    xy: np.ndarray,
    quads: np.ndarray,
    patches: dict[str, list[tuple[int, int]]],
    z_thick: float,
    *,
    cyclic_pairs: tuple[str, str] | None = None,
    separation: float = 0.0,
    patch_types: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Extrude CCW 2-D quads to one hex layer and write constant/polyMesh (ascii).

    patches: name → list of directed 2-D boundary edges (a, b). For ordinary
    patches (a, b) may be in any direction (the owner quad's CCW edge is used).
    For the cyclic pair (bottom, top) edges must be given as (lo, hi) along the
    periodic line, in the same order on both sides, so face k ↔ face k.
    """
    xy = np.asarray(xy, dtype=float)
    quads = np.asarray(quads, dtype=np.int64)
    N = len(xy)
    nq = len(quads)
    # orientation check (must be CCW, positive area)
    a = xy[quads]
    area = 0.5 * (
        (a[:, 0, 0] * a[:, 1, 1] - a[:, 1, 0] * a[:, 0, 1])
        + (a[:, 1, 0] * a[:, 2, 1] - a[:, 2, 0] * a[:, 1, 1])
        + (a[:, 2, 0] * a[:, 3, 1] - a[:, 3, 0] * a[:, 2, 1])
        + (a[:, 3, 0] * a[:, 0, 1] - a[:, 0, 0] * a[:, 3, 1])
    )
    if np.any(area <= 0):
        raise RuntimeError(f"curved_periodic writer: {int((area <= 0).sum())} quads with area <= 0; refusing to write")
    # edge → (cell, local k)
    edge_cells: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for c in range(nq):
        q = quads[c]
        for k in range(4):
            p, r = int(q[k]), int(q[(k + 1) % 4])
            key = (p, r) if p < r else (r, p)
            edge_cells.setdefault(key, []).append((c, p, r))
    internal = []
    bnd_edges: dict[tuple[int, int], tuple[int, int, int]] = {}
    for key, lst in edge_cells.items():
        if len(lst) == 2:
            (c1, p1, r1), (c2, _p2, _r2) = lst
            if c1 < c2:
                own, nb, p, r = c1, c2, p1, r1
            else:
                own, nb = c2, c1
                p, r = lst[1][1], lst[1][2]
            internal.append((own, nb, [p, r, r + N, p + N]))
        elif len(lst) == 1:
            bnd_edges[key] = lst[0]
        else:
            raise RuntimeError(f"non-manifold 2-D edge {key}: {len(lst)} quads")
    internal.sort(key=lambda t: (t[0], t[1]))
    ptypes = dict(patch_types or {})
    faces: list[list[int]] = [f for _o, _n, f in internal]
    owner = [o for o, _n, _f in internal]
    neigh = [n for _o, n, _f in internal]
    start: dict[str, int] = {}
    count: dict[str, int] = {}
    used = set()
    order = [n for n in PATCH_ORDER if n in patches or n == "frontAndBack"] + [
        n for n in patches if n not in PATCH_ORDER
    ]
    for name in order:
        start[name] = len(faces)
        if name == "frontAndBack":
            for c in range(nq):
                q = [int(v) for v in quads[c]]
                faces.append([q[0], q[3], q[2], q[1]])
                owner.append(c)
            for c in range(nq):
                q = [int(v) for v in quads[c]]
                faces.append([q[0] + N, q[1] + N, q[2] + N, q[3] + N])
                owner.append(c)
            count[name] = 2 * nq
            continue
        for (ea, eb) in patches[name]:
            key = (ea, eb) if ea < eb else (eb, ea)
            if key not in bnd_edges:
                raise RuntimeError(f"patch {name}: edge {key} is not a boundary edge")
            if key in used:
                raise RuntimeError(f"patch {name}: edge {key} assigned twice")
            used.add(key)
            c, p, r = bnd_edges[key]
            if cyclic_pairs and name == cyclic_pairs[0]:
                lo, hi = ea, eb
                f = [lo, hi, hi + N, lo + N]
                # outward check vs owner CCW edge direction
                if (p, r) != (lo, hi):
                    raise RuntimeError(f"bottom cyclic edge ({lo},{hi}) not CCW for its owner")
            elif cyclic_pairs and name == cyclic_pairs[1]:
                lo, hi = ea, eb
                f = [lo, lo + N, hi + N, hi]
                if (p, r) != (hi, lo):
                    raise RuntimeError(f"top cyclic edge ({lo},{hi}) not CW for its owner")
            else:
                f = [p, r, r + N, p + N]
            faces.append(f)
            owner.append(c)
        count[name] = len(patches[name])
    if len(used) != len(bnd_edges):
        raise RuntimeError(f"{len(bnd_edges) - len(used)} boundary edges not in any patch")
    md = Path(out_case) / "constant" / "polyMesh"
    md.mkdir(parents=True, exist_ok=True)
    for stale in ("cellZones", "faceZones", "pointZones", "sets"):
        pth = md / stale
        if pth.is_dir():
            import shutil
            shutil.rmtree(pth, ignore_errors=True)
        elif pth.exists():
            pth.unlink()
    pts3 = np.vstack([np.c_[xy, np.zeros(N)], np.c_[xy, np.full(N, float(z_thick))]])
    note = f"nPoints:{2*N}  nCells:{nq}  nFaces:{len(faces)}  nInternalFaces:{len(internal)}"
    (md / "points").write_text(
        _hdr("vectorField", "points") + f"{2*N}\n(\n" + "".join(f"({x:.15g} {y:.15g} {z:.15g})\n" for x, y, z in pts3) + ")\n"
    )
    (md / "faces").write_text(
        _hdr("faceList", "faces") + f"{len(faces)}\n(\n" + "".join("4(" + " ".join(str(i) for i in f) + ")\n" for f in faces) + ")\n"
    )
    (md / "owner").write_text(_hdr("labelList", "owner", note) + f"{len(owner)}\n(\n" + "".join(f"{i}\n" for i in owner) + ")\n")
    (md / "neighbour").write_text(_hdr("labelList", "neighbour", note) + f"{len(neigh)}\n(\n" + "".join(f"{i}\n" for i in neigh) + ")\n")
    b = [f"{len(order)}\n(\n"]
    for name in order:
        if cyclic_pairs and name in cyclic_pairs:
            other = cyclic_pairs[1] if name == cyclic_pairs[0] else cyclic_pairs[0]
            sv = separation if name == cyclic_pairs[0] else -separation
            body = (
                "        type            cyclic;\n        inGroups        1(cyclic);\n"
                "        matchTolerance  0.0001;\n        transform       translational;\n"
                f"        neighbourPatch  {other};\n        separationVector (0 {sv:.15g} 0);\n"
            )
        elif name == "frontAndBack":
            body = "        type            empty;\n        inGroups        1(empty);\n"
        elif ptypes.get(name, "wall" if name.startswith("blade") else "patch") == "wall":
            body = "        type            wall;\n        inGroups        1(wall);\n"
        else:
            body = "        type            patch;\n"
        b.append(f"    {name}\n    {{\n{body}        nFaces          {count[name]};\n        startFace       {start[name]};\n    }}\n")
    b.append(")\n")
    (md / "boundary").write_text(_hdr("polyBoundaryMesh", "boundary") + "".join(b))
    return {"n_points": 2 * N, "n_cells": nq, "n_faces": len(faces), "n_internal": len(internal), "patches": count, "order": order}


# --------------------------------------------------------------------------- geometry helpers
def _dens(P: np.ndarray, h: float) -> np.ndarray:
    Q = np.vstack([P, P[:1]])
    out = []
    for a, b in zip(Q[:-1], Q[1:]):
        n = max(1, int(math.ceil(float(np.hypot(*(b - a))) / h)))
        out.append(a + np.outer(np.arange(n) / n, b - a))
    return np.vstack(out)


class _Loop:
    """Closed CCW polyline with arc-length parameter and nearest-point query."""

    def __init__(self, P: np.ndarray):
        self.P = np.asarray(P, dtype=float)
        Q = np.vstack([self.P, self.P[:1]])
        self.A = Q[:-1]
        self.B = Q[1:]
        self.AB = self.B - self.A
        self.L2 = (self.AB ** 2).sum(1)
        seg = np.sqrt(self.L2)
        self.s0 = np.r_[0.0, np.cumsum(seg)[:-1]]
        self.L = float(seg.sum())

    def nearest(self, q: np.ndarray) -> tuple[np.ndarray, float, float]:
        t = np.clip(((q - self.A) * self.AB).sum(1) / np.maximum(self.L2, 1e-300), 0.0, 1.0)
        C = self.A + t[:, None] * self.AB
        d = np.hypot(*(C - q).T)
        k = int(np.argmin(d))
        return C[k], float(self.s0[k] + t[k] * math.sqrt(self.L2[k])), float(d[k])

    def at(self, sig: np.ndarray) -> np.ndarray:
        sig = np.mod(np.asarray(sig, dtype=float), self.L)
        k = np.clip(np.searchsorted(self.s0, sig, side="right") - 1, 0, len(self.s0) - 1)
        t = (sig - self.s0[k]) / np.sqrt(np.maximum(self.L2[k], 1e-300))
        return self.A[k] + t[:, None] * self.AB[k]

    def normal(self, sig: np.ndarray, win: float) -> np.ndarray:
        """Outward unit normal (CCW loop → right of tangent), averaged over ±win arc."""
        sig = np.asarray(sig, dtype=float)
        a = self.at(sig - win)
        b = self.at(sig + win)
        t = b - a
        t /= np.hypot(*t.T)[:, None]
        return np.c_[t[:, 1], -t[:, 0]]


def layer_dist(y1: float, growth: float, n_wall: int, n_ext: int, g_ext: float) -> np.ndarray:
    """Distances from wall of layer nodes 0..n_wall+n_ext."""
    d = [0.0]
    h = y1
    for _ in range(n_wall):
        d.append(d[-1] + h)
        h *= growth
    h = (d[-1] - d[-2]) * g_ext
    for _ in range(n_ext):
        d.append(d[-1] + h)
        h *= g_ext
    return np.array(d)


def _fill_geometric(d0: float, h_last: float, L: float, m: int) -> np.ndarray:
    """m more nodes from d0 to d0+L, first cell h_last*q, ratio q (solved)."""
    lo, hi = 0.5, 3.0

    def tot(q):
        return h_last * sum(q ** k for k in range(1, m + 1))

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if tot(mid) > L:
            hi = mid
        else:
            lo = mid
    q = 0.5 * (lo + hi)
    out = []
    acc = d0
    h = h_last
    for _ in range(m):
        h *= q
        acc += h
        out.append(acc)
    out = np.array(out)
    out *= 1.0  # exact end
    out[-1] = d0 + L
    return out, q


def _smoothstep5(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (t * (6 * t - 15) + 10)


def _tfi(S, N, W, E):
    """S,N: (ni,2) along i; W,E: (nj,2) along j. Corners must agree."""
    ni, nj = len(S), len(W)
    u = np.linspace(0, 1, ni)[:, None, None]
    v = np.linspace(0, 1, nj)[None, :, None]
    # arc-length params improve TFI for graded edges
    def frac(c):
        d = np.r_[0, np.cumsum(np.hypot(*np.diff(c, axis=0).T))]
        return d / d[-1]
    us = frac(S)[:, None, None]
    un = frac(N)[:, None, None]
    vw = frac(W)[None, :, None]
    ve = frac(E)[None, :, None]
    uu = (1 - v) * us + v * un
    vv = (1 - u) * vw + u * ve
    S_ = S[:, None, :]
    N_ = N[:, None, :]
    W_ = W[None, :, :]
    E_ = E[None, :, :]
    X = (1 - vv) * S_ + vv * N_ + (1 - uu) * W_ + uu * E_ - (
        (1 - uu) * (1 - vv) * S[0] + uu * (1 - vv) * S[-1] + (1 - uu) * vv * N[0] + uu * vv * N[-1]
    )
    return X


def _bisect_scale(h0, cap, tot, floor=None):
    """c such that sum(clip(c*h0, floor, cap)) == tot (cap.sum() must exceed tot)."""
    fl = np.zeros_like(cap) if floor is None else np.minimum(floor, cap)
    f = lambda c: np.minimum(np.maximum(c * h0, fl), cap)
    lo, hi = 0.0, 1.0
    while f(hi).sum() < tot:
        hi *= 2.0
    for _ in range(100):
        c = 0.5 * (lo + hi)
        if f(c).sum() < tot:
            lo = c
        else:
            hi = c
    t = f(hi)
    return t * (tot / t.sum())


def _limit_ratio(t: np.ndarray, r: float) -> np.ndarray:
    """Forward/backward sweep so adjacent spacings differ by <= r (only shrinks cells)."""
    t = np.array(t, dtype=float)
    for k in range(1, len(t)):
        t[k] = min(t[k], r * t[k - 1])
    for k in range(len(t) - 2, -1, -1):
        t[k] = min(t[k], r * t[k + 1])
    return t


def _grade_ends(sg: np.ndarray, ha: float, hb: float, r: float = 1.2) -> np.ndarray:
    """Re-space a 1-D node distribution so the end cells are ha / hb and grow by <= r
    into the interior shape (min(c*h_orig, ha r^k, hb r^(n-1-k)), rescaled to the length)."""
    h0 = np.diff(sg)
    tot = float(sg[-1] - sg[0])
    n = len(h0)
    k = np.arange(n)
    cap = np.minimum(ha * r ** k, hb * r ** (n - 1 - k))
    if cap.sum() <= tot:
        return sg
    # floor: the end cells are at least ha / hb (a curvature-clustered tip end would otherwise stay
    # far below the passage wall cell it abuts), relaxing inward at r per cell
    flo = np.minimum(np.maximum(ha * r ** (-k.astype(float)), hb * r ** (-(n - 1 - k).astype(float))), cap)
    if flo.sum() >= tot:
        flo = None
    t = _bisect_scale(h0, cap, tot, floor=flo)
    # the curvature-weighted interior can itself jump by >3x per cell: limit, re-fit, re-cap
    for _ in range(400):
        t = _limit_ratio(t, r)
        t = np.minimum(t * (tot / t.sum()), cap)
        if flo is not None:
            t = np.maximum(t, flo)
        if abs(t.sum() - tot) <= 1e-12 * tot:
            break
    t *= tot / t.sum()
    out = sg[0] + np.r_[0.0, np.cumsum(t)]
    out[-1] = sg[-1]
    return out


def _graded_dist(h0: float, L: float, n: int, r: float) -> np.ndarray:
    """n cells over length L from the fine end: t_k = min(h0 r^k, cap), cap solved so sum = L
    (growth <= r, then a steady coarse plateau). Falls back to a pure geometric ratio > r only
    if n cells at r cannot reach L. Returns node arc positions 0..L."""
    k = np.arange(n)
    g = h0 * r ** k
    if g.sum() <= L:
        lo, hi = r, 4.0
        for _ in range(100):
            q = 0.5 * (lo + hi)
            if (h0 * q ** k).sum() < L:
                lo = q
            else:
                hi = q
        t = h0 * hi ** k
    else:
        lo, hi = 0.0, float(g.max())
        for _ in range(100):
            c = 0.5 * (lo + hi)
            if np.minimum(g, c).sum() < L:
                lo = c
            else:
                hi = c
        t = np.minimum(g, hi)
    t = t * (L / t.sum())
    out = np.r_[0.0, np.cumsum(t)]
    out[-1] = L
    return out


def _ctrl_from_spacing(h: np.ndarray) -> np.ndarray:
    """Thomas-Middlecoff-type 1-D source term at interior nodes: x_xixi + P x_xi = 0 for spacings h."""
    rr = h[1:] / h[:-1]
    return -2.0 * (rr - 1.0) / (rr + 1.0)


# --------------------------------------------------------------------------- main builder
@dataclass
class CurvedParams:
    n_pass: int = 120        # cells along U/D (Q → Q')
    n_nose: int = 72         # cells along each nose O piece
    n_ext: int = 13          # extension layers beyond n_wall (O-ring); ring outer (singular corner nodes) >= 0.3 mm off the wall
    g_ext: float = 1.2
    n_fill: int = 14         # layers from O outer to periodic line in U/D
    n_w: int = 40            # W cells inlet → LE composite (graded=False only; graded derives it)
    n_e: int = 96            # E cells TE composite → outlet (graded=False only; graded derives it)
    smooth_iter: int = 3000
    t_flat_up: float = -7.0e-3   # T flat for x <= this (relative to LE x=0 of chord)
    t_blend_up: float = -2.0e-3  # T equals smoothed medial curve for x >= this
    smooth_win: float = 0.6e-3   # medial smoothing half-window [m]
    nose_normal_win: float = 40e-6
    nose_curv_w: float = 1.0e-3   # curvature weight for nose wall node clustering [m]
    nose_blend_n: int = 1         # nodes over which nose normals blend into the U/D end lines
    # ---- physics-graded spacing (W/E: Winslow + Thomas-Middlecoff source terms)
    graded: bool = True
    nose_end_growth: float = 1.15 # nose end cells = adjacent passage wall cell, growth <= this
    w_growth: float = 1.10        # W/E growth away from the blade composites (to a coarse plateau)
    h0_ratio: float = 1.2         # max ratio of first W/E cell width between neighbouring lines
    nose_tip_fac: float = 0.7     # nose end cell at the LE/TE-circle (D) end, relative to the D wall cell
    corner_fac: float = 1.3       # first-cell widening at the end-line/nose-outer corners
    corner_w: float = 2.0         # corner widening half-width [lines]
    inlet_uniform: float = 0.7    # inlet/outlet j-distribution: 0 = composite arc fraction, 1 = uniform
    q_clip: float = 0.6
    respace_every: int = 50       # re-space W/E i-lines to the graded target every N smoothing sweeps
    feet_smooth: float = 3.0      # Gaussian width [lines] for U/D wall-foot de-bunching
    feet_ratio: float = 1.15      # max neighbour ratio of U/D wall-foot spacing
    corner_rot_deg: float = 20.0  # end-line fill part leaves the O-outer corner turned this far into the passage
    corner_rot_p: float = 6.0     # offset shape t (1-t)^p (concentrated near the corner)
    corner_rot_m: int = 10        # columns over which the offset decays (cosine)
    blend_k: float = 8.0          # W/E: per-line graded spacing near the composite -> common fraction far away
    ortho_corner_w: float = 5.0   # ... but released over this many lines around each corner
    ortho_k: float = 4.0          # W/E: rows pulled onto the composite normal (Gaussian width in rows)
    ring_wall_normal: bool = True # continuous O-ring: U/D ring lines wall-normal like the noses, Hermite fill beyond
    ring_win_fac: float = 0.8     # ring normal-averaging half-window grows by this x wall distance
    ring_win_cap: float = 1.0     # ... up to this wall distance (1 m = uncapped: grows through the whole ring)
    fill_tan: float = 0.6         # Hermite tangent length at the ring outer node (fraction of the fill chord)


@dataclass
class CurvedResult:
    xy: np.ndarray
    quads: np.ndarray
    patches: dict
    notes: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    T: np.ndarray | None = None
    blocks: dict | None = None


def build_curved_periodic(
    loop: list[tuple[float, float]],
    *,
    pitch: float,
    chord: float,
    x_in: float,
    x_out: float,
    y1: float,
    growth: float,
    n_wall: int,
    params: CurvedParams | None = None,
) -> CurvedResult:
    from scipy.spatial import cKDTree

    pr = params or CurvedParams()
    s = float(pitch)
    P = np.asarray([p for p in loop], dtype=float)
    if np.allclose(P[0], P[-1]):
        P = P[:-1]
    # CCW
    ar = 0.5 * float(np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1]))
    if ar < 0:
        P = P[::-1].copy()
    blade = _Loop(P)
    notes: list[str] = []
    x_le = float(P[:, 0].min())
    x_te = float(P[:, 0].max())
    xc0 = float(P[np.argmin(np.hypot(*(P - P[:, :].min(0)).T)), 0])
    _ = xc0
    # ---- periodic line T(x): medial (equidistant) curve blade k vs blade k+1
    D = _dens(P, 4e-6)
    tk, tk1 = cKDTree(D), cKDTree(D + [0.0, s])
    ymin_b, ymax_b = float(P[:, 1].min()), float(P[:, 1].max())
    xs = np.arange(x_in, x_out + 1e-12, 10e-6)
    ye = np.empty_like(xs)
    for i, x in enumerate(xs):
        lo, hi = ymin_b, ymax_b + s  # f(lo) < 0 (near k) ... f(hi) > 0
        ys = np.linspace(lo, hi, 240)
        q = np.c_[np.full_like(ys, x), ys]
        f = tk.query(q)[0] - tk1.query(q)[0]
        k = np.where((f[:-1] < 0) & (f[1:] >= 0))[0]
        if len(k) == 0:
            raise RuntimeError(f"medial curve not found at x={x:.5g}")
        k = int(k[-1])
        a_, b_ = ys[k], ys[k + 1]
        for _ in range(40):
            m_ = 0.5 * (a_ + b_)
            fm = tk.query([x, m_])[0] - tk1.query([x, m_])[0]
            if fm < 0:
                a_ = m_
            else:
                b_ = m_
        ye[i] = 0.5 * (a_ + b_)
    # smooth (moving average, reflect ends)
    w = max(1, int(round(pr.smooth_win / 10e-6)))
    ker = np.ones(2 * w + 1) / (2 * w + 1)
    ys_ = np.convolve(np.pad(ye, w, mode="edge"), ker, mode="valid")
    ys_ = np.convolve(np.pad(ys_, w, mode="edge"), ker, mode="valid")
    y_flat_up = float(np.interp(x_le + pr.t_flat_up, xs, ys_))
    y_flat_dn = float(np.interp(x_te - pr.t_flat_up, xs, ys_))
    y_flat = 0.5 * (y_flat_up + y_flat_dn)
    su = _smoothstep5((xs - (x_le + pr.t_flat_up)) / (pr.t_blend_up - pr.t_flat_up))
    sd = _smoothstep5(((x_te - pr.t_flat_up) - xs) / (pr.t_blend_up - pr.t_flat_up))
    wgt = np.minimum(su, sd)
    Ty = y_flat + (ys_ - y_flat) * wgt
    T = np.c_[xs, Ty]

    def Tof(x):
        return np.interp(x, xs, Ty)

    # clearance of periodic line to metal
    dT_k = tk.query(T)[0]
    dT_k1 = tk1.query(T)[0]
    clear = float(min(dT_k.min(), dT_k1.min()))
    notes.append(f"periodic line: medial curve blade k|k+1, smoothed ±{pr.smooth_win*1e3:.2f} mm, flat y={y_flat*1e3:.4f} mm "
                 f"for x<={ (x_le+pr.t_flat_up)*1e3:.2f} and x>={(x_te-pr.t_flat_up)*1e3:.2f} mm; min metal clearance {clear*1e3:.3f} mm")

    # ---- layering
    rO = layer_dist(y1, growth, n_wall, pr.n_ext, pr.g_ext)
    nO = len(rO) - 1
    d_o = float(rO[-1])
    h_lastO = float(rO[-1] - rO[-2])
    if clear < 2.0 * d_o:
        raise RuntimeError(f"periodic clearance {clear:.3g} m < 2*O thickness {d_o:.3g} m")

    def ring(sig):
        """Continuous O-ring nodes (m, nO+1, 2) above wall arc positions sig: layer k is stepped off
        layer k-1 along the wall normal averaged over a window that widens with wall distance
        (nose_normal_win + ring_win_fac * min(r_k, ring_win_cap)). Wall-normal at the wall; further out the LE/TE-circle
        curvature jump is spread over several columns, so the ring's outer contour and its column
        spacing stay smooth (a pure normal offset would jump by (1 + d kappa) at the circle tangency)."""
        sig = np.asarray(sig, dtype=float)
        X = np.empty((len(sig), nO + 1, 2))
        X[:, 0] = blade.at(sig)
        for k in range(1, nO + 1):
            win = pr.nose_normal_win + pr.ring_win_fac * min(float(rO[k]), pr.ring_win_cap)
            X[:, k] = X[:, k - 1] + float(rO[k] - rO[k - 1]) * blade.normal(sig, win)
        return X

    # ---- Q / Q': periodic node at passage entrance/exit = T point nearest midpoint of min-gap segment
    dk, ik = cKDTree(D + [0.0, s]).query(D)
    # entrance side (x < mid) and exit side
    xm = 0.5 * (x_le + x_te)
    le_side = D[:, 0] < xm
    k1 = int(np.argmin(np.where(le_side, dk, np.inf)))
    k2 = int(np.argmin(np.where(~le_side, dk, np.inf)))
    mid1 = 0.5 * (D[k1] + D[ik[k1]] + [0.0, s])
    mid2 = 0.5 * (D[k2] + D[ik[k2]] + [0.0, s])
    xQ = float(xs[np.argmin(np.hypot(xs - mid1[0], Ty - mid1[1]))])
    xQ2 = float(xs[np.argmin(np.hypot(xs - mid2[0], Ty - mid2[1]))])
    g_min_true = float(min(dk[k1], dk[k2]))
    notes.append(f"true min metal gap blade k→k+1 = {g_min_true*1e3:.3f} mm; passage Q x={xQ*1e3:.3f} mm, Q' x={xQ2*1e3:.3f} mm")

    # ---- periodic nodes along T: arc-length uniform in U/D part
    def t_arc(xa, xb, n):
        xx = xs[(xs >= xa) & (xs <= xb)]
        xx = np.r_[xa, xx, xb]
        yy = Tof(xx)
        d = np.r_[0, np.cumsum(np.hypot(np.diff(xx), np.diff(yy)))]
        return xx, d

    xx, dd = t_arc(xQ, xQ2, 0)
    tgt = np.linspace(0, dd[-1], pr.n_pass + 1)
    xP = np.interp(tgt, dd, xx)
    PT = np.c_[xP, Tof(xP)]
    PB = PT - [0.0, s]
    # feet
    footU = []
    footD = []
    for p in PT:
        c, sg, d = blade.nearest(p)
        footU.append((c, sg, d))
    for p in PB:
        c, sg, d = blade.nearest(p)
        footD.append((c, sg, d))
    sigU = np.array([f[1] for f in footU])
    sigD = np.array([f[1] for f in footD])
    L = blade.L
    # U feet must be on the outer surface (σ decreasing with i), D on inner (σ increasing)
    dsu = np.diff(np.unwrap(sigU / L * 2 * np.pi) * L / (2 * np.pi))
    dsd = np.diff(np.unwrap(sigD / L * 2 * np.pi) * L / (2 * np.pi))
    if not (np.all(dsu < 0) and np.all(dsd > 0)):
        raise RuntimeError(f"wall feet not monotone (U dec ok={bool(np.all(dsu<0))}, D inc ok={bool(np.all(dsd>0))})")

    if pr.graded:
        # foot de-bunching: nearest-point feet collapse onto convex wall regions (LE tip on the
        # D side: 13 / 4 / 26 / 107 um). Smooth the foot spacing (Gaussian in index, total arc kept),
        # then cap the neighbour ratio; the lines stay within a few degrees of wall-normal.
        def _smooth_feet(sig, PL):
            u = np.unwrap(sig / L * 2 * np.pi) * L / (2 * np.pi)
            d = np.diff(u)
            sgn = 1.0 if d[0] > 0 else -1.0
            d = np.abs(d)
            tot = float(d.sum())
            k = np.arange(-12, 13)
            ker = np.exp(-0.5 * (k / pr.feet_smooth) ** 2)
            ker /= ker.sum()
            t = np.convolve(np.pad(d, 12, mode="reflect"), ker, mode="valid")
            t *= tot / t.sum()
            for _ in range(500):
                t = _limit_ratio(t, pr.feet_ratio)
                t *= tot / t.sum()
                if np.max(np.maximum(t[1:] / t[:-1], t[:-1] / t[1:])) <= pr.feet_ratio * 1.001:
                    break
            un = u[0] + sgn * np.r_[0.0, np.cumsum(t)]
            un[-1] = u[-1]
            sg_ = np.mod(un, L)
            C = blade.at(sg_)
            return [(C[i], float(sg_[i]), float(np.hypot(*(PL[i] - C[i])))) for i in range(len(sg_))]

        footU = _smooth_feet(sigU, PT)
        footD = _smooth_feet(sigD, PB)
        sigU = np.array([f[1] for f in footU])
        sigD = np.array([f[1] for f in footD])

    def line_block(Pline, feet, n_fill):
        """(ni, nj) nodes: wall foot → O layers (rO) → fill → periodic node."""
        ni = len(Pline)
        nj = nO + n_fill + 1
        X = np.zeros((ni, nj, 2))
        qs = []
        RG = ring(np.array([f[1] for f in feet])) if pr.ring_wall_normal else None
        uu = np.linspace(0.0, 1.0, 1201)[:, None]
        h00, h10 = 2 * uu ** 3 - 3 * uu ** 2 + 1, uu ** 3 - 2 * uu ** 2 + uu
        h01, h11 = -2 * uu ** 3 + 3 * uu ** 2, uu ** 3 - uu ** 2
        for i, (p, (c, _sg, h)) in enumerate(zip(Pline, feet)):
            if RG is None:
                n = (p - c) / h
                fill, q = _fill_geometric(d_o, h_lastO, h - d_o, n_fill)
                qs.append(q)
                r = np.r_[rO, fill]
                X[i] = c[None, :] + r[:, None] * n[None, :]
            else:
                # continuous O-ring: the ring part of every U/D line is exactly wall-normal (the same
                # normal the nose pieces use), so ring lines run on unbroken around the noses; the fill
                # part leaves the ring outer node tangent to the normal (cubic Hermite) and bends
                # smoothly onto the periodic node (no kink at the ring outer edge)
                X[i, : nO + 1] = RG[i]
                F_ = X[i, nO]
                n = X[i, nO] - X[i, nO - 1]
                n = n / np.hypot(*n)
                v = p - F_
                Lc = float(np.hypot(*v))
                Cv = h00 * F_ + h10 * (n * Lc * pr.fill_tan) + h01 * p + h11 * v
                sa = np.r_[0.0, np.cumsum(np.hypot(*np.diff(Cv, axis=0).T))]
                fill, q = _fill_geometric(0.0, h_lastO, float(sa[-1]), n_fill)
                qs.append(q)
                X[i, nO + 1:] = np.c_[np.interp(fill, sa, Cv[:, 0]), np.interp(fill, sa, Cv[:, 1])]
            X[i, -1] = p  # exact periodic node
        return X, qs

    XU, qU = line_block(PT, footU, pr.n_fill)       # j=0 wall (outer), j=-1 top periodic
    XD, qD = line_block(PB, footD, pr.n_fill)       # j=0 wall (inner), j=-1 bottom periodic

    def _round_corner(X, at_end):
        """Turn the fill part of the end line (O-outer corner F -> periodic node) into the block by
        corner_rot_deg at F, so the inlet/outlet block gets ~90+rot deg at F instead of 90 (the
        valence-5 corner no longer squashes its two wedge cells). Offset L tan(rot) t (1-t)^p along
        the in-block direction, decayed over corner_rot_m columns; F, the O layers and the periodic
        node are untouched."""
        if pr.corner_rot_deg <= 0:
            return X
        ni = X.shape[0]
        i0, step = (ni - 1, -1) if at_end else (0, 1)
        F_, Qn = X[i0, nO], X[i0, -1]
        dv = Qn - F_
        Lf = float(np.hypot(*dv))
        dv = dv / Lf
        e = X[i0 + step, nO] - F_
        e = e - (e @ dv) * dv
        e = e / np.hypot(*e)
        A = Lf * math.tan(math.radians(pr.corner_rot_deg))
        p_ = pr.corner_rot_p
        for c in range(pr.corner_rot_m):
            i = i0 + step * c
            dec = 0.5 * (1 + math.cos(math.pi * c / pr.corner_rot_m))
            seg = X[i, nO:]
            t = np.r_[0, np.cumsum(np.hypot(*np.diff(seg, axis=0).T))]
            t = t / t[-1]
            X[i, nO:] = seg + (dec * A * t * (1 - t) ** p_)[:, None] * e[None, :]
        return X

    if pr.graded:
        for _X in (XU, XD):
            _round_corner(_X, False)
            _round_corner(_X, True)
    notes.append(f"U/D fill ratio q in [{min(qU+qD):.3f}, {max(qU+qD):.3f}]; O layers {nO} (n_wall {n_wall} growth {growth} + {pr.n_ext} ext @ {pr.g_ext}); d_o={d_o*1e6:.1f} um")

    # ---- nose O pieces (CCW on loop)
    def nose_block(sig_a, n_a, sig_b, n_b, n_cells, ha=None, hb=None):
        """Wall from σ_a CCW to σ_b; normals blended to n_a / n_b at ends."""
        sb = sig_b if sig_b > sig_a else sig_b + L
        # curvature-weighted distribution
        ss = np.linspace(sig_a, sb, 4001)
        nn = blade.normal(ss, pr.nose_normal_win)
        ang = np.unwrap(np.arctan2(nn[:, 1], nn[:, 0]))
        kap = np.abs(np.gradient(ang, ss))
        wts = 1.0 + pr.nose_curv_w * kap  # kap in 1/m; cap r≈0.18 mm → kap≈5.6e3
        cw = np.r_[0, np.cumsum(0.5 * (wts[1:] + wts[:-1]) * np.diff(ss))]
        sg = np.interp(np.linspace(0, cw[-1], n_cells + 1), cw, ss)
        if ha is not None and hb is not None:
            sg = _grade_ends(sg, ha, hb, pr.nose_end_growth)
        Wn = blade.at(sg)
        Nn = blade.normal(sg, pr.nose_normal_win)
        # blend ends to exact line directions
        m = len(sg)
        bl = np.clip(np.arange(m) / pr.nose_blend_n, 0, 1)
        br = np.clip((m - 1 - np.arange(m)) / pr.nose_blend_n, 0, 1)
        for i in range(m):
            wa = 1 - _smoothstep5(bl[i])
            wb = 1 - _smoothstep5(br[i])
            v = Nn[i] * (1 - wa - wb) + n_a * wa + n_b * wb if (wa + wb) <= 1 else (n_a * wa + n_b * wb) / (wa + wb)
            Nn[i] = v / np.hypot(*v)
        Wn[0] = XA0
        Wn[-1] = XB0
        X = Wn[:, None, :] + rO[None, :, None] * Nn[:, None, :]
        if pr.ring_wall_normal:
            X = ring(sg)   # same ring generator as the U/D ring parts: ring lines continue unbroken
        return X

    nU0 = (XU[0, 1] - XU[0, 0]); nU0 /= np.hypot(*nU0)
    nD0 = (XD[0, 1] - XD[0, 0]); nD0 /= np.hypot(*nD0)
    nUe = (XU[-1, 1] - XU[-1, 0]); nUe /= np.hypot(*nUe)
    nDe = (XD[-1, 1] - XD[-1, 0]); nDe /= np.hypot(*nDe)
    _dev = []
    for _sg, _n in ((sigU[0], nU0), (sigD[0], nD0), (sigU[-1], nUe), (sigD[-1], nDe)):
        _nn = blade.normal(np.array([_sg]), pr.nose_normal_win)[0]
        _dev.append(float(np.degrees(np.arccos(np.clip(_nn @ _n, -1, 1)))))
    notes.append("end-line vs wall-normal deviation deg (U0,D0,Ue,De): " + ", ".join(f"{d:.1f}" for d in _dev))
    # LE nose: from U foot 0 (outer) CCW to D foot 0 (inner)
    def _sarc(a_, b_):
        d_ = (a_ - b_) % L
        return float(min(d_, L - d_))

    if pr.graded:
        haL, hbL = _sarc(sigU[1], sigU[0]), _sarc(sigD[1], sigD[0])
        haT, hbT = _sarc(sigD[-1], sigD[-2]), _sarc(sigU[-1], sigU[-2])
        # the D end lines start on the convex LE/TE circle: the nose wall-normal fan widens the
        # end column outward by ~(1 + d_o/R) while the D column does not; balance wall vs O outer
        hbL *= pr.nose_tip_fac
        haT *= pr.nose_tip_fac
    else:
        haL = hbL = haT = hbT = None
    XA0, XB0 = XU[0, 0], XD[0, 0]
    XNL = nose_block(sigU[0], nU0, sigD[0], nD0, pr.n_nose, haL, hbL)
    # TE nose: from D foot end (inner) CCW to U foot end (outer)
    XA0, XB0 = XD[-1, 0], XU[-1, 0]
    XNT = nose_block(sigD[-1], nDe, sigU[-1], nUe, pr.n_nose, haT, hbT)
    # force exact shared radial lines
    XNL[0] = XU[0, : nO + 1]
    XNL[-1] = XD[0, : nO + 1]
    XNT[0] = XD[-1, : nO + 1]
    XNT[-1] = XU[-1, : nO + 1]

    # ---- W block: east composite (top → bottom): U line0 from Q down to O outer, nose outer CCW, D line0 O outer → Q-s
    eU = XU[0, nO:][::-1]            # Q ... F_U0
    eN = XNL[:, -1]                  # F_U0 ... F_D0
    eD = XD[0, nO:]                  # F_D0 ... Q-s
    east = np.vstack([eU, eN[1:], eD[1:]])[::-1]  # bottom → top
    nyW = len(east)
    # periodic rows (x from x_in to xQ): graded toward Q
    def per_x(xa, xb, n, h_end, at_end=True):
        """n cells from xa to xb, geometric with last cell ~h_end at xb (or at xa)."""
        Lx = xb - xa
        lo, hi = 1.0, 1.6
        for _ in range(200):
            q = 0.5 * (lo + hi)
            tot = h_end * sum(q ** k for k in range(n))
            if tot > Lx:
                hi = q
            else:
                lo = q
        q = 0.5 * (lo + hi)
        hs = h_end * q ** np.arange(n)
        hs *= Lx / hs.sum()
        if at_end:
            hs = hs[::-1]
        return xa + np.r_[0, np.cumsum(hs)], q

    # E block composite (bottom → top): D line end Q'-s → O outer, TE nose outer CCW, U line end O outer → Q'
    wD = XD[-1, nO:][::-1]           # Q'-s ... F_De
    wN = XNT[:, -1]                  # F_De ... F_Ue
    wU = XU[-1, nO:]                 # F_Ue ... Q'
    westE = np.vstack([wD, wN[1:], wU[1:]])
    nyE = len(westE)

    # ---- periodic-aware elliptic smoothing of W / E interiors (+ periodic rows slide on T)
    def smooth(X, fixed_i, n_iter, omega=0.8, Pc=None, Qc=None):
        """Winslow (inverse-Laplace) elliptic smoothing, Jacobi iterations, optional
        Thomas-Middlecoff source terms: a(x_xixi + P x_xi) - 2b x_xieta + g(x_etaeta + Q x_eta) = 0.
        Periodic rows (j=0 bottom, j=nj-1 top=bottom+s) slide along T via ghost rows from the
        opposite side; column fixed_i (inlet/outlet) and the blade-side column are held."""
        ni, nj, _ = X.shape
        Pi = None if Pc is None else Pc[1:-1, :, None]
        Qi = None if Qc is None else Qc[1:-1, :, None]
        for _ in range(n_iter):
            G = np.concatenate([(X[:, nj - 2] - [0.0, s])[:, None], X, (X[:, 1] + [0.0, s])[:, None]], axis=1)
            xp, xm = G[2:, 1:-1], G[:-2, 1:-1]
            yp, ym = G[1:-1, 2:], G[1:-1, :-2]
            xpyp, xpym = G[2:, 2:], G[2:, :-2]
            xmyp, xmym = G[:-2, 2:], G[:-2, :-2]
            Xi = 0.5 * (xp - xm)
            Et = 0.5 * (yp - ym)
            al = (Et ** 2).sum(-1)[..., None]
            ga = (Xi ** 2).sum(-1)[..., None]
            be = (Xi * Et).sum(-1)[..., None]
            num = al * (xp + xm) + ga * (yp + ym) - 0.5 * be * (xpyp - xpym - xmyp + xmym)
            if Pi is not None:
                num = num + al * Pi * Xi + ga * Qi * Et
            new = num / (2.0 * (al + ga))
            Y = X[1:-1] + omega * (new - X[1:-1])
            X[1:-1, 1:-1] = Y[:, 1:-1]
            xb = Y[:, 0, 0]
            lo_, hi_ = min(X[0, 0, 0], X[-1, 0, 0]), max(X[0, 0, 0], X[-1, 0, 0])
            xb = np.clip(xb, lo_, hi_)
            if X[-1, 0, 0] > X[0, 0, 0]:
                xb = np.maximum.accumulate(xb)
            X[1:-1, 0, 0] = xb
            X[1:-1, 0, 1] = Tof(xb) - s
            X[1:-1, nj - 1, 0] = xb
            X[1:-1, nj - 1, 1] = Tof(xb)
        return X

    def _ef(C):
        e_ = np.r_[0, np.cumsum(np.hypot(*np.diff(C, axis=0).T))]
        return e_ / e_[-1]

    if not pr.graded:
        h_Q = float(np.hypot(*(PT[1] - PT[0])))
        xW, qW = per_x(x_in, xQ, pr.n_w, h_Q, at_end=True)
        xE, qE = per_x(xQ2, x_out, pr.n_e, float(np.hypot(*(PT[-1] - PT[-2]))), at_end=False)
        XW = _tfi(np.c_[xW, Tof(xW) - s], np.c_[xW, Tof(xW)], np.c_[np.full(nyW, x_in), (Tof(x_in) - s) + _ef(east) * s], east)
        XE = _tfi(np.c_[xE, Tof(xE) - s], np.c_[xE, Tof(xE)], westE, np.c_[np.full(nyE, x_out), (Tof(x_out) - s) + _ef(westE) * s])
        XW = smooth(XW, 0, pr.smooth_iter)
        XE = smooth(XE, XE.shape[0] - 1, pr.smooth_iter)
    else:
        # first W/E cell across each composite node = the neighbour cell width on the other side
        # (passage streamwise cell along the U/D end lines, last O-layer thickness around the noses)
        def _wid(A_, B_):
            return np.hypot(*(A_ - B_).T)

        def _gm(a_, b_):
            return float(np.sqrt(a_ * b_))

        jc = (pr.n_fill, pr.n_fill + pr.n_nose)   # end-line / nose-outer corners (both W and E)

        def _h0_prep(h0):
            jj = np.arange(len(h0))
            f = np.ones(len(h0))
            for c in jc:
                f = f + (pr.corner_fac - 1.0) * np.exp(-(((jj - c) / pr.corner_w) ** 2))
            h0 = _limit_ratio(h0 * f, pr.h0_ratio)
            h0[0] = h0[-1] = min(h0[0], h0[-1])
            return h0

        hUw = _wid(XU[0, nO:], XU[1, nO:])[::-1]
        hNw = _wid(XNL[:, -1], XNL[:, -2])
        hDw = _wid(XD[0, nO:], XD[1, nO:])
        h0W = _h0_prep(np.r_[hUw[:-1], _gm(hUw[-1], hNw[0]), hNw[1:-1], _gm(hNw[-1], hDw[0]), hDw[1:]][::-1])
        hDe = _wid(XD[-1, nO:], XD[-2, nO:])[::-1]
        hNe = _wid(XNT[:, -1], XNT[:, -2])
        hUe = _wid(XU[-1, nO:], XU[-2, nO:])
        h0E = _h0_prep(np.r_[hDe[:-1], _gm(hDe[-1], hNe[0]), hNe[1:-1], _gm(hNe[-1], hUe[0]), hUe[1:]])
        if len(h0W) != nyW or len(h0E) != nyE:
            raise RuntimeError(f"graded h0 length mismatch W {len(h0W)}/{nyW} E {len(h0E)}/{nyE}")
        r = pr.w_growth

        def _n_needed(h0, Lj):
            return int(np.max(np.ceil(np.log1p(Lj * (r - 1.0) / h0) / np.log(r))))

        n_w = _n_needed(h0W, (east[:, 0] - x_in) * 1.08)
        n_e = _n_needed(h0E, (x_out - westE[:, 0]) * 1.08)
        xW = xQ - _graded_dist(h0W[0], xQ - x_in, n_w, r)[::-1]
        xE = xQ2 + _graded_dist(h0E[0], x_out - xQ2, n_e, r)
        qW = qE = r
        wu = pr.inlet_uniform
        XW = _tfi(np.c_[xW, Tof(xW) - s], np.c_[xW, Tof(xW)],
                  np.c_[np.full(nyW, x_in), (Tof(x_in) - s) + ((1 - wu) * _ef(east) + wu * np.linspace(0, 1, nyW)) * s], east)
        XE = _tfi(np.c_[xE, Tof(xE) - s], np.c_[xE, Tof(xE)], westE,
                  np.c_[np.full(nyE, x_out), (Tof(x_out) - s) + ((1 - wu) * _ef(westE) + wu * np.linspace(0, 1, nyE)) * s])

        def _respace(X, h0, at_end):
            """Re-space each j-line to the graded target and build P from it. Near the composite
            the per-line target (first cell = neighbour width, growth <= r) holds and the first rows
            sit on the composite normal; far away all lines share one spacing fraction (smooth
            j-lines in the coarse inlet/outlet region)."""
            ni, nj, _ = X.shape
            Pc = np.zeros((ni, nj))
            kk = np.arange(ni)
            Pl0 = X[::-1, 0] if at_end else X[:, 0]
            L0 = float(np.hypot(*np.diff(Pl0, axis=0).T).sum())
            F_ref = _graded_dist(h0[0], L0, ni - 1, r) / L0
            wk = np.exp(-((kk / pr.blend_k) ** 2))
            wo = np.exp(-((kk / pr.ortho_k) ** 2))[:, None]
            C = (X[-1] if at_end else X[0]).copy()
            u1 = np.diff(C, axis=0)
            u1 = u1 / np.hypot(*u1.T)[:, None]
            tg = np.r_[u1[:1], u1[:-1] + u1[1:], u1[-1:]]
            nrm = np.c_[tg[:, 1], -tg[:, 0]]
            nrm = nrm / np.hypot(*nrm.T)[:, None]
            inn = (X[-2] if at_end else X[1]) - C
            nrm = nrm * np.sign((nrm * inn).sum(1))[:, None]
            jj_ = np.arange(nj)
            wj = np.ones(nj)
            for c in jc:
                wj = wj * (1 - np.exp(-(((jj_ - c) / pr.ortho_corner_w) ** 2)))
            for j in range(nj):
                Pl = X[::-1, j] if at_end else X[:, j]
                d = np.hypot(*np.diff(Pl, axis=0).T)
                S = np.r_[0.0, np.cumsum(d)]
                G = _graded_dist(h0[j], S[-1], ni - 1, r)
                Sn = wk * G + (1 - wk) * S[-1] * F_ref
                if np.any(np.diff(Sn) <= 0):
                    Sn = G
                Pn = np.c_[np.interp(Sn, S, Pl[:, 0]), np.interp(Sn, S, Pl[:, 1])]
                if 0 < j < nj - 1 and wj[j] > 1e-3:
                    # (not at the convex end-line/nose corners: normals from both sides converge there)
                    Pn = wj[j] * wo * (C[j] + Sn[:, None] * nrm[j]) + (1 - wj[j] * wo) * Pn
                Pn[0], Pn[-1] = Pl[0], Pl[-1]
                X[:, j] = Pn[::-1] if at_end else Pn
                h = np.diff(Sn)
                Pc[1:-1, j] = _ctrl_from_spacing(h[::-1] if at_end else h)
            xb = X[1:-1, 0, 0]
            X[1:-1, 0, 1] = Tof(xb) - s
            X[1:-1, nj - 1, 0] = xb
            X[1:-1, nj - 1, 1] = Tof(xb)
            return X, Pc

        def _qline(C):
            d = np.hypot(*np.diff(C, axis=0).T)
            return np.clip(_ctrl_from_spacing(np.r_[d[-1], d, d[0]]), -pr.q_clip, pr.q_clip)

        def _qfield(X, comp_i):
            ni = X.shape[0]
            qc, qf = _qline(X[comp_i]), _qline(X[ni - 1 - comp_i])
            w_ = np.linspace(0.0, 1.0, ni)
            if comp_i == 0:
                w_ = w_[::-1]
            return w_[:, None] * qc[None, :] + (1 - w_[:, None]) * qf[None, :]

        XW, PW = _respace(XW, h0W, True)
        XE, PE = _respace(XE, h0E, False)
        QW, QE = _qfield(XW, XW.shape[0] - 1), _qfield(XE, 0)
        # elliptic smoothing with source terms, interleaved with spacing restoration along the
        # smoothed i-lines (boundary-normal spacing frozen); the last operation is a re-space
        for _ in range(max(1, pr.smooth_iter // pr.respace_every)):
            XW = smooth(XW, 0, pr.respace_every, Pc=PW, Qc=QW)
            XE = smooth(XE, XE.shape[0] - 1, pr.respace_every, Pc=PE, Qc=QE)
            XW, _ = _respace(XW, h0W, True)
            XE, _ = _respace(XE, h0E, False)
        notes.append(f"graded W/E: n_w={n_w} n_e={n_e}, growth {r} to a coarse plateau; first cell = neighbour width "
                     f"(W {h0W.min()*1e6:.1f}-{h0W.max()*1e6:.1f} um, E {h0E.min()*1e6:.1f}-{h0E.max()*1e6:.1f} um), corner_fac {pr.corner_fac}")
        for _nm, _X, _ha, _hb in (("LE", XNL, haL, hbL), ("TE", XNT, haT, hbT)):
            _w = np.hypot(*np.diff(_X[:, 0], axis=0).T)
            notes.append(f"{_nm} nose end cells {_w[0]*1e6:.1f}/{_w[-1]*1e6:.1f} um vs passage wall {_ha*1e6:.1f}/{_hb*1e6:.1f} um, "
                         f"max adjacent ratio {float(np.max(np.maximum(_w[1:]/_w[:-1], _w[:-1]/_w[1:]))):.3f}")

    # ---- assemble nodes/quads
    blocks = {"U": XU, "D": XD, "NL": XNL, "NT": XNT, "W": XW, "E": XE}
    key2id: dict[tuple[int, int], int] = {}
    pts: list[tuple[float, float]] = []

    def nid(p):
        k = (int(round(p[0] * 1e12)), int(round(p[1] * 1e12)))
        v = key2id.get(k)
        if v is None:
            v = len(pts)
            key2id[k] = v
            pts.append((float(p[0]), float(p[1])))
        return v

    ids = {}
    for name, X in blocks.items():
        ni, nj, _ = X.shape
        I = np.empty((ni, nj), dtype=np.int64)
        for i in range(ni):
            for j in range(nj):
                I[i, j] = nid(X[i, j])
        ids[name] = I
    xy = np.array(pts)
    quads = []
    for name, I in ids.items():
        ni, nj = I.shape
        for i in range(ni - 1):
            for j in range(nj - 1):
                q = [I[i, j], I[i + 1, j], I[i + 1, j + 1], I[i, j + 1]]
                a = xy[q]
                ar_ = 0.5 * sum(a[k, 0] * a[(k + 1) % 4, 1] - a[(k + 1) % 4, 0] * a[k, 1] for k in range(4))
                if ar_ < 0:
                    q = q[::-1]
                quads.append(q)
    quads = np.array(quads, dtype=np.int64)

    # ---- patches
    def edges_row(I, j, rev=False):
        e = [(int(I[i, j]), int(I[i + 1, j])) for i in range(I.shape[0] - 1)]
        return e

    def edges_col(I, i):
        return [(int(I[i, j]), int(I[i, j + 1])) for j in range(I.shape[1] - 1)]

    wall = edges_row(ids["U"], 0) + edges_row(ids["D"], 0) + edges_row(ids["NL"], 0) + edges_row(ids["NT"], 0)
    inlet = edges_col(ids["W"], 0)
    outlet = edges_col(ids["E"], ids["E"].shape[0] - 1)
    # periodic (lo→hi along +x): bottom = W south, D j=-1, E south ; top = W north, U j=-1, E north
    bottom = edges_row(ids["W"], 0) + edges_row(ids["D"], ids["D"].shape[1] - 1) + edges_row(ids["E"], 0)
    top = edges_row(ids["W"], ids["W"].shape[1] - 1) + edges_row(ids["U"], ids["U"].shape[1] - 1) + edges_row(ids["E"], ids["E"].shape[1] - 1)
    # ---- mandatory assertions
    if len(bottom) != len(top):
        raise RuntimeError(f"cyclic edge count mismatch {len(bottom)} vs {len(top)}")
    bn = np.array([[xy[a], xy[b]] for a, b in bottom])
    tn = np.array([[xy[a], xy[b]] for a, b in top])
    pair_err = float(np.abs(tn - bn - np.array([0.0, s])).max())
    if pair_err > 1e-12:
        raise RuntimeError(f"cyclic pairing error {pair_err:.3e} m > 1e-12")
    # midline never touches metal: min distance >= O thickness
    if clear < d_o:
        raise RuntimeError(f"periodic line clearance {clear:.3e} < O thickness {d_o:.3e}")
    # min O thickness available in throat: half distance wall→periodic along U/D lines
    hU = np.array([f[2] for f in footU])
    hD = np.array([f[2] for f in footD])
    metrics = {
        "n_cells": int(len(quads)),
        "n_points_2d": int(len(xy)),
        "cyclic_pair_max_err_m": pair_err,
        "periodic_clearance_min_m": clear,
        "O_thickness_m": d_o,
        "O_layers": nO,
        "wall_to_periodic_min_m": float(min(hU.min(), hD.min())),
        "true_min_gap_m": g_min_true,
        "y_flat_m": y_flat,
        "xQ_m": xQ,
        "xQ2_m": xQ2,
        "first_cell_m": float(rO[1]),
        "blocks": {k: list(v.shape[:2]) for k, v in blocks.items()},
        "W_ratio": qW,
        "E_ratio": qE,
    }
    notes.append(
        f"curved_periodic: cells={len(quads)} pair_err={pair_err:.1e} m, clearance {clear*1e3:.3f} mm >= d_o {d_o*1e3:.3f} mm, "
        f"wall→periodic min {metrics['wall_to_periodic_min_m']*1e3:.3f} mm"
    )
    return CurvedResult(
        xy=xy, quads=quads,
        patches={"inlet": inlet, "outlet": outlet, "bottom": bottom, "top": top, "blade0": wall},
        notes=notes, metrics=metrics, T=T, blocks=blocks,
    )
