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


def _wall_patches(mesh_dir: Path) -> list[tuple[str, int, int]]:
    """(name, startFace, nFaces) of every ``type wall`` patch in polyMesh/boundary."""
    txt = re.sub(r"/\*.*?\*/", "", (mesh_dir / "boundary").read_text(encoding="utf-8", errors="replace"), flags=re.S)
    out = []
    for m in re.finditer(r"(\w+)\s*\{([^{}]*)\}", txt):
        body = m.group(2)
        if re.search(r"\btype\s+wall\s*;", body):
            nf = int(re.search(r"nFaces\s+(\d+)", body).group(1))
            sf = int(re.search(r"startFace\s+(\d+)", body).group(1))
            out.append((m.group(1), sf, nf))
    return out


@lru_cache(maxsize=4)
def _wall_loops_cached(mesh_dir: str, mtime: float) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    d = Path(mesh_dir)
    pts = _read_points(d / "points")
    faces = _read_faces(d / "faces")
    zmin = float(pts[:, 2].min())
    ztol = 1e-9 + 1e-6 * float(np.ptp(pts[:, 2]) or 1.0)
    edges: list[tuple[int, int]] = []
    wall_pts: set[int] = set()
    for _name, sf, nf in _wall_patches(d):
        for f in faces[sf:sf + nf]:
            wall_pts.update(f)
            lo = [v for v in f if abs(pts[v, 2] - zmin) <= ztol]
            if len(lo) == 2:
                edges.append((lo[0], lo[1]))
    adj: dict[int, list[int]] = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    if any(len(v) != 2 for v in adj.values()):
        raise ValueError("wall patch z-min edges do not form closed loops")
    loops, seen = [], set()
    for start in adj:
        if start in seen:
            continue
        loop, prev, cur = [start], None, start
        seen.add(start)
        while True:
            nxt = adj[cur][0] if adj[cur][0] != prev else adj[cur][1]
            if nxt == start:
                break
            loop.append(nxt)
            seen.add(nxt)
            prev, cur = cur, nxt
        loops.append(pts[loop, :2].copy())
    wp = pts[sorted(wall_pts), :2] if wall_pts else np.empty((0, 2))
    return tuple(loops), wp


def wall_loops_m(case_dir: Path) -> tuple[list[np.ndarray], np.ndarray]:
    """Ordered closed loops (metres) of the wall patches' z-min boundary edges,
    taken from constant/polyMesh itself, plus all wall-patch points (2D)."""
    d = Path(case_dir) / "constant" / "polyMesh"
    loops, wp = _wall_loops_cached(str(d), (d / "faces").stat().st_mtime)
    return list(loops), wp


def cell_polygons_m(case_dir: Path) -> list[np.ndarray]:
    """Per-cell 2D outline (metres), index = owner cell id of constant/polyMesh."""
    d = Path(case_dir) / "constant" / "polyMesh"
    _, polys = _cell_polys_cached(str(d), (d / "faces").stat().st_mtime)
    return list(polys)


def polygon_centroids(polys: list[np.ndarray]) -> np.ndarray:
    """Area centroids (same definition as OpenFOAM C for a planar face)."""
    out = np.empty((len(polys), 2))
    for i, p in enumerate(polys):
        x, y = p[:, 0], p[:, 1]
        x1, y1 = np.roll(x, -1), np.roll(y, -1)
        c = x * y1 - x1 * y
        a = c.sum() / 2.0
        if abs(a) < 1e-30:
            out[i] = p.mean(axis=0)
        else:
            out[i] = (((x + x1) * c).sum() / (6 * a), ((y + y1) * c).sum() / (6 * a))
    return out


def latest_cell_centres_m(case_dir: Path) -> tuple[Path | None, np.ndarray | None]:
    """C (writeCellCentres) from the latest time dir that has one, as N×2 metres."""
    tds = []
    for d in Path(case_dir).iterdir():
        try:
            tds.append((float(d.name), d))
        except ValueError:
            continue
    for _, d in sorted(tds, reverse=True):
        f = d / "C"
        if f.is_file():
            try:
                txt = f.read_text(encoding="utf-8", errors="replace")
                m = re.search(r"internalField\s+nonuniform\s+List<vector>\s*(\d+)\s*\(", txt)
                if not m:
                    return f, None
                n = int(m.group(1))
                vals = re.findall(r"\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+[-+0-9.eE]+\s*\)", txt[m.end():])[:n]
                return f, np.array(vals, dtype=float)
            except Exception:
                return f, None
    return None, None


def align_polys_to_centres(polys: list[np.ndarray], cc_m: np.ndarray, tol_m: float = 1e-7):
    """Return polys reordered so poly i has centroid == C[i] (field numbering).

    The time-dir fields and C share one cell numbering. If constant/polyMesh was
    renumbered (renumberMesh) after those fields were written, owner ids differ
    from field ids; match by centroid so colours land on the right cell.
    """
    cen = polygon_centroids(polys)
    if len(cen) != len(cc_m):
        raise ValueError(f"cell count mismatch polys={len(cen)} C={len(cc_m)}")
    err = np.hypot(*(cen - cc_m).T)
    info: dict[str, Any] = {"direct_max_err_m": float(err.max())}
    if err.max() <= tol_m:
        info.update(reordered=False, max_err_m=float(err.max()))
        return polys, info
    idx = np.empty(len(cc_m), dtype=np.int64)
    dmin = np.empty(len(cc_m))
    for s0 in range(0, len(cc_m), 512):
        blk = cc_m[s0:s0 + 512]
        d2 = (blk[:, None, 0] - cen[None, :, 0]) ** 2 + (blk[:, None, 1] - cen[None, :, 1]) ** 2
        j = np.argmin(d2, axis=1)
        idx[s0:s0 + 512] = j
        dmin[s0:s0 + 512] = np.sqrt(d2[np.arange(len(j)), j])
    if dmin.max() > tol_m or len(np.unique(idx)) != len(idx):
        raise ValueError(f"polyMesh cells do not match C (nn max {dmin.max():.3g} m)")
    info.update(reordered=True, max_err_m=float(dmin.max()))
    return [polys[j] for j in idx], info


def outline_gate_m(loops_mm: list[np.ndarray], wall_pts_m: np.ndarray, pitch_mm: float, n_viz: int) -> float:
    """Max distance (m) of any drawn outline vertex to a polyMesh wall-patch point,
    after removing the nearest pitch-copy offset."""
    from scipy.spatial import cKDTree

    tree = cKDTree(wall_pts_m)
    worst = 0.0
    for lp in loops_mm:
        v = lp / 1000.0
        best = np.full(len(v), np.inf)
        for k in range(max(int(n_viz), 1)):
            best = np.minimum(best, tree.query(v - [0.0, k * pitch_mm / 1000.0])[0])
        worst = max(worst, float(best.max()))
    return worst


class CellDomain:
    """Cell polygons of one pitch stacked ×n_viz in y (mm), plus inside tests."""

    def __init__(self, case_dir: Path, pitch_mm: float, n_viz: int):
        import matplotlib.tri as mtri

        polys_m = cell_polygons_m(case_dir)
        self.check: dict[str, Any] = {"reordered": False, "max_err_m": None}
        c_file, cc_m = latest_cell_centres_m(case_dir)
        if cc_m is not None:
            polys_m, self.check = align_polys_to_centres(polys_m, cc_m)
            self.check["C_file"] = str(c_file)
        base = [p * 1000.0 for p in polys_m]
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
        # Metal outline from the mesh's own wall-patch faces, stacked with the
        # same pitch offsets as the cells (never from the design profile).
        self.wall_loops: list[np.ndarray] = []
        try:
            loops_m, wall_pts_m = wall_loops_m(case_dir)
            for k in range(self.n_viz):
                off = np.array([0.0, k * self.pitch_mm])
                self.wall_loops.extend(lp * 1000.0 + off for lp in loops_m)
            self.check["outline_max_dist_m"] = outline_gate_m(self.wall_loops, wall_pts_m, self.pitch_mm, self.n_viz)
        except Exception as exc:  # noqa: BLE001
            self.check["outline_error"] = f"{type(exc).__name__}: {exc}"
        self.check["centroid_max_err_m"] = float(np.hypot(*(polygon_centroids(polys_m) - cc_m).T).max()) if cc_m is not None else None

    def draw_metal(self, ax, zorder: float = 6) -> bool:
        """Grey metal fill + outline from polyMesh wall patches (all pitch copies)."""
        if not self.wall_loops:
            return False
        for lp in self.wall_loops:
            ax.fill(lp[:, 0], lp[:, 1], facecolor="#c8c8c8", edgecolor="#222", lw=0.7, zorder=zorder)
        return True

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


def _is_num(name: str) -> bool:
    try:
        float(name)
        return True
    except ValueError:
        return False


def try_domain(case_dir: Path | None, pitch_mm: float, n_viz: int) -> CellDomain | None:
    """Cached CellDomain, or None if the polyMesh cannot be read as 2D cells."""
    if case_dir is None:
        return None
    try:
        faces = Path(case_dir) / "constant" / "polyMesh" / "faces"
        c_mt = 0.0
        try:
            tdc = sorted((float(d.name), d) for d in Path(case_dir).iterdir() if _is_num(d.name) and (d / "C").is_file())
            c_mt = (tdc[-1][1] / "C").stat().st_mtime if tdc else 0.0
        except Exception:
            pass
        key = (str(Path(case_dir).resolve()), round(float(pitch_mm or 0.0), 9), max(int(n_viz or 1), 1),
               faces.stat().st_mtime, c_mt)
        if key not in _DOM_CACHE:
            _DOM_CACHE.clear()
            _DOM_CACHE[key] = CellDomain(Path(case_dir), pitch_mm, n_viz)
        return _DOM_CACHE[key]
    except Exception:
        return None
