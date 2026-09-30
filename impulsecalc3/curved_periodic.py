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


# --------------------------------------------------------------------------- main builder
@dataclass
class CurvedParams:
    n_pass: int = 120        # cells along U/D (Q → Q')
    n_nose: int = 64         # cells along each nose O piece
    n_ext: int = 10          # extension layers beyond n_wall (O part)
    g_ext: float = 1.15
    n_fill: int = 14         # layers from O outer to periodic line in U/D
    n_w: int = 40            # W cells inlet → LE composite
    n_e: int = 96            # E cells TE composite → outlet
    smooth_iter: int = 3000
    t_flat_up: float = -7.0e-3   # T flat for x <= this (relative to LE x=0 of chord)
    t_blend_up: float = -2.0e-3  # T equals smoothed medial curve for x >= this
    smooth_win: float = 0.6e-3   # medial smoothing half-window [m]
    nose_normal_win: float = 40e-6
    nose_curv_w: float = 1.0e-3   # curvature weight for nose wall node clustering [m]
    nose_blend_n: int = 3         # nodes over which nose normals blend into the U/D end lines


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

    def line_block(Pline, feet, n_fill):
        """(ni, nj) nodes: wall foot → O layers (rO) → fill → periodic node."""
        ni = len(Pline)
        nj = nO + n_fill + 1
        X = np.zeros((ni, nj, 2))
        qs = []
        for i, (p, (c, _sg, h)) in enumerate(zip(Pline, feet)):
            n = (p - c) / h
            fill, q = _fill_geometric(d_o, h_lastO, h - d_o, n_fill)
            qs.append(q)
            r = np.r_[rO, fill]
            X[i] = c[None, :] + r[:, None] * n[None, :]
            X[i, -1] = p  # exact periodic node
        return X, qs

    XU, qU = line_block(PT, footU, pr.n_fill)       # j=0 wall (outer), j=-1 top periodic
    XD, qD = line_block(PB, footD, pr.n_fill)       # j=0 wall (inner), j=-1 bottom periodic
    notes.append(f"U/D fill ratio q in [{min(qU+qD):.3f}, {max(qU+qD):.3f}]; O layers {nO} (n_wall {n_wall} growth {growth} + {pr.n_ext} ext @ {pr.g_ext}); d_o={d_o*1e6:.1f} um")

    # ---- nose O pieces (CCW on loop)
    def nose_block(sig_a, n_a, sig_b, n_b, n_cells):
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
    XA0, XB0 = XU[0, 0], XD[0, 0]
    XNL = nose_block(sigU[0], nU0, sigD[0], nD0, pr.n_nose)
    # TE nose: from D foot end (inner) CCW to U foot end (outer)
    XA0, XB0 = XD[-1, 0], XU[-1, 0]
    XNT = nose_block(sigD[-1], nDe, sigU[-1], nUe, pr.n_nose)
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

    h_Q = float(np.hypot(*(PT[1] - PT[0])))
    xW, qW = per_x(x_in, xQ, pr.n_w, h_Q, at_end=True)
    xE, qE = per_x(xQ2, x_out, pr.n_e, float(np.hypot(*(PT[-1] - PT[-2]))), at_end=False)
    south = np.c_[xW, Tof(xW) - s]
    north = np.c_[xW, Tof(xW)]
    ef = np.r_[0, np.cumsum(np.hypot(*np.diff(east, axis=0).T))]
    ef /= ef[-1]
    west = np.c_[np.full(nyW, x_in), (Tof(x_in) - s) + ef * s]
    XW = _tfi(south, north, west, east)  # (ni=n_w+1, nj=nyW)
    # E block: west composite (bottom → top): D line end Q'-s → O outer, TE nose outer CCW, U line end O outer → Q'
    wD = XD[-1, nO:][::-1]           # Q'-s ... F_De
    wN = XNT[:, -1]                  # F_De ... F_Ue
    wU = XU[-1, nO:]                 # F_Ue ... Q'
    westE = np.vstack([wD, wN[1:], wU[1:]])
    nyE = len(westE)
    southE = np.c_[xE, Tof(xE) - s]
    northE = np.c_[xE, Tof(xE)]
    ef = np.r_[0, np.cumsum(np.hypot(*np.diff(westE, axis=0).T))]
    ef /= ef[-1]
    eastE = np.c_[np.full(nyE, x_out), (Tof(x_out) - s) + ef * s]
    XE = _tfi(southE, northE, westE, eastE)

    # ---- periodic-aware Laplacian smoothing of W / E interiors (+ periodic rows slide on T)
    def smooth(X, fixed_i, n_iter, omega=0.8):
        """Winslow (inverse-Laplace) elliptic smoothing, Jacobi iterations.
        Periodic rows (j=0 bottom, j=nj-1 top=bottom+s) slide along T via ghost rows from the
        opposite side; column fixed_i (inlet/outlet) and the blade-side column are held."""
        ni, nj, _ = X.shape
        for _ in range(n_iter):
            # ghost-extended array in j: row -1 = row nj-2 - s ; row nj = row 1 + s
            G = np.concatenate([(X[:, nj - 2] - [0.0, s])[:, None], X, (X[:, 1] + [0.0, s])[:, None]], axis=1)
            C = G[1:-1, 1:-1 + 1]  # placeholder (unused)
            xp, xm = G[2:, 1:-1], G[:-2, 1:-1]          # i±1, j in 0..nj-1 (i in 1..ni-2)
            yp, ym = G[1:-1, 2:], G[1:-1, :-2]          # j±1
            xpyp, xpym = G[2:, 2:], G[2:, :-2]
            xmyp, xmym = G[:-2, 2:], G[:-2, :-2]
            Xi = 0.5 * (xp - xm)
            Et = 0.5 * (yp - ym)
            al = (Et ** 2).sum(-1)[..., None]
            ga = (Xi ** 2).sum(-1)[..., None]
            be = (Xi * Et).sum(-1)[..., None]
            num = al * (xp + xm) + ga * (yp + ym) - 0.5 * be * (xpyp - xpym - xmyp + xmym)
            new = num / (2.0 * (al + ga))
            Y = X[1:-1] + omega * (new - X[1:-1])       # rows i=1..ni-2, all j
            X[1:-1, 1:-1] = Y[:, 1:-1]
            # periodic rows: take x from the bottom-row update, project onto T
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

    XW = smooth(XW, 0, pr.smooth_iter)
    XE = smooth(XE, XE.shape[0] - 1, pr.smooth_iter)

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
