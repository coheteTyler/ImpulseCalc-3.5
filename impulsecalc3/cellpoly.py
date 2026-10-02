"""Real cell polygons for 2D field plots (no Delaunay hull fill).

A 2D OpenFOAM case is one cell thick in z. Each cell owns exactly one face on
the z-min plane of the ``empty`` patch; that face's point loop *is* the cell's
2D outline. Plots built from these loops show only the meshed domain: curved
periodic edges, inlet/outlet and blade wall layers look exactly as meshed.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


def _list_body(txt: str) -> str:
    """Text after the ``N\\n(`` list opener (skips the FoamFile header)."""
    txt = re.sub(r"/\*.*?\*/", "", txt, flags=re.S)
    m = re.search(r"\}\s*(?://[^\n]*\s*)*(\d+)\s*\(", txt)
    if not m:
        raise ValueError("no list opener")
    return txt[m.end():]


def _read_points(p: Path) -> np.ndarray:
    body = _list_body(p.read_text(encoding="utf-8", errors="replace"))
    vals = re.findall(r"\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)", body)
    return np.array(vals, dtype=float)


def _read_faces(p: Path) -> list[list[int]]:
    body = _list_body(p.read_text(encoding="utf-8", errors="replace"))
    return [[int(v) for v in grp.split()] for grp in re.findall(r"\d+\s*\(([\d\s]+)\)", body)]


def _read_labels(p: Path) -> np.ndarray:
    body = _list_body(p.read_text(encoding="utf-8", errors="replace"))
    body = body[: body.rfind(")")]
    return np.array(body.split(), dtype=np.int64)


@lru_cache(maxsize=4)
def _cell_polys_cached(mesh_dir: str, mtime: float) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    d = Path(mesh_dir)
    pts = _read_points(d / "points")
    faces = _read_faces(d / "faces")
    owner = _read_labels(d / "owner")
    n_cells = int(owner.max()) + 1
    zmin = float(pts[:, 2].min())
    ztol = 1e-9 + 1e-6 * float(np.ptp(pts[:, 2]) or 1.0)
    polys: list[np.ndarray | None] = [None] * n_cells
    for fi, f in enumerate(faces):
        if fi >= len(owner) or len(f) < 3:
            continue
        if np.all(np.abs(pts[f, 2] - zmin) <= ztol):
            c = int(owner[fi])
            if polys[c] is None:
                polys[c] = pts[f, :2].copy()
    if any(p is None for p in polys):
        raise ValueError("cells without a z-min face; not a 1-cell-thick 2D mesh")
    return pts[:, :2], tuple(polys)  # type: ignore[arg-type]


def cell_polygons_m(case_dir: Path) -> list[np.ndarray]:
    """Per-cell 2D outline (metres), index = OpenFOAM cell id."""
    d = Path(case_dir) / "constant" / "polyMesh"
    _, polys = _cell_polys_cached(str(d), (d / "faces").stat().st_mtime)
    return list(polys)


class CellDomain:
    """Cell polygons of one pitch stacked ×n_viz in y (mm), plus inside tests."""

    def __init__(self, case_dir: Path, pitch_mm: float, n_viz: int):
        import matplotlib.tri as mtri

        base = [p * 1000.0 for p in cell_polygons_m(case_dir)]
        self.n_cells = len(base)
        self.pitch_mm = float(pitch_mm)
        self.n_viz = max(int(n_viz or 1), 1)
        self.polys: list[np.ndarray] = []
        for k in range(self.n_viz):
            off = np.array([0.0, k * self.pitch_mm])
            self.polys.extend(p + off for p in base)
        allp = np.concatenate(self.polys)
        self.xmin, self.ymin = (float(v) for v in allp.min(axis=0))
        self.xmax, self.ymax = (float(v) for v in allp.max(axis=0))
        # One-pitch triangulation of the real cells (fan per convex cell) for
        # point location; stacked copies are tested by shifting y by k*pitch.
        pts = np.concatenate(base)
        tris = []
        off = 0
        for p in base:
            n = len(p)
            tris.extend((off, off + i, off + i + 1) for i in range(1, n - 1))
            off += n
        self._tri = mtri.Triangulation(pts[:, 0], pts[:, 1], np.array(tris))
        try:
            self._finder = self._tri.get_trifinder()
        except Exception:
            self._finder = None
        self._base = base

    def tile(self, vals) -> np.ndarray:
        v = np.asarray(vals, dtype=float)
        return np.concatenate([v] * self.n_viz) if len(v) == self.n_cells else v

    def contains(self, X, Y) -> np.ndarray:
        """Boolean mask: point lies inside some real cell (any pitch copy)."""
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        xf, yf = X.ravel(), Y.ravel()
        inside = np.zeros(xf.shape, dtype=bool)
        for k in range(self.n_viz):
            yk = yf - k * self.pitch_mm
            if self._finder is not None:
                inside |= np.asarray(self._finder(xf, yk)) >= 0
            else:
                from matplotlib.path import Path as MPath
                for p in self._base:
                    lo, hi = p.min(axis=0), p.max(axis=0)
                    sel = (~inside) & (xf >= lo[0]) & (xf <= hi[0]) & (yk >= lo[1]) & (yk <= hi[1])
                    if sel.any():
                        idx = np.where(sel)[0]
                        inside[idx[MPath(p).contains_points(np.c_[xf[idx], yk[idx]])]] = True
        return inside.reshape(X.shape)

    def clip_path(self):
        """Compound path of all cells (nonzero fill = union) for clipping lines."""
        from matplotlib.path import Path as MPath

        verts, codes = [], []
        for p in self.polys:
            verts.append(p)
            verts.append(p[:1])
            c = np.full(len(p) + 1, MPath.LINETO, dtype=np.uint8)
            c[0] = MPath.MOVETO
            c[-1] = MPath.CLOSEPOLY
            codes.append(c)
        return MPath(np.concatenate(verts), np.concatenate(codes))

    def draw(self, ax, vals, *, cmap: str, norm=None, vmin=None, vmax=None, zorder: float = 1):
        """PolyCollection of real cells coloured by cell value, no edges."""
        from matplotlib.collections import PolyCollection

        pc = PolyCollection(self.polys, array=self.tile(vals), cmap=cmap, norm=norm,
                            edgecolors="none", linewidths=0, antialiaseds=False, zorder=zorder)
        if norm is None and (vmin is not None or vmax is not None):
            pc.set_clim(vmin, vmax)
        ax.add_collection(pc)
        return pc

    def clip_artists(self, ax, artists: list[Any]) -> None:
        from matplotlib.patches import PathPatch

        patch = PathPatch(self.clip_path(), transform=ax.transData, facecolor="none", edgecolor="none")
        for a in artists:
            try:
                a.set_clip_path(patch)
            except Exception:
                pass

    def set_limits(self, ax, xlim: tuple[float, float] | None = None) -> None:
        x0, x1 = (xlim if xlim else (self.xmin, self.xmax))
        ax.set_xlim(max(x0, self.xmin), min(x1, self.xmax))
        ax.set_ylim(self.ymin, self.ymax)


_DOM_CACHE: dict[tuple, CellDomain] = {}


def try_domain(case_dir: Path | None, pitch_mm: float, n_viz: int) -> CellDomain | None:
    """Cached CellDomain, or None if the polyMesh cannot be read as 2D cells."""
    if case_dir is None:
        return None
    try:
        faces = Path(case_dir) / "constant" / "polyMesh" / "faces"
        key = (str(Path(case_dir).resolve()), round(float(pitch_mm or 0.0), 9), max(int(n_viz or 1), 1),
               faces.stat().st_mtime)
        if key not in _DOM_CACHE:
            _DOM_CACHE.clear()
            _DOM_CACHE[key] = CellDomain(Path(case_dir), pitch_mm, n_viz)
        return _DOM_CACHE[key]
    except Exception:
        return None
