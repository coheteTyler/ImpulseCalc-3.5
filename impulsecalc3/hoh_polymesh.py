"""Native OpenFOAM polyMesh writer for HOH.

Index convention (2-D weld then extrude):
    i_z0(k) = k          # z = 0
    i_z1(k) = k + N      # z = Z
(x, y) → k via rounding 1e-9 so TFI seams share nodes.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

Z_THICK_DEFAULT = 0.001
PATCH_ORDER = ("inlet", "outlet", "bottom", "top", "blades", "frontAndBack")


def write_foam_header(cls: str, obj: str) -> str:
    return (
        "FoamFile\n"
        "{\n"
        "    version     2.0;\n"
        "    format      ascii;\n"
        f"    class       {cls};\n"
        f"    object      {obj};\n"
        "}\n"
        "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"
    )


WELD_TOL = 1e-9


def _cross2(u, v) -> float:
    return float(u[0] * v[1] - u[1] * v[0])


def orient_quads(xy, quads):
    """CCW 2-D quads. Drop degenerates. Clockwise → (a,d,c,b)."""
    pts = np.asarray(xy, dtype=float).reshape(-1, 2)
    out = []
    for q in quads:
        ia, ib, ic, id_ = (int(v) for v in q)
        a, b, c, d = pts[ia], pts[ib], pts[ic], pts[id_]
        area = 0.5 * (_cross2(b - a, d - a) + _cross2(c - b, d - c))
        if abs(area) < 1e-16:
            continue
        if area < 0:
            out.append((ia, id_, ic, ib))
        else:
            out.append((ia, ib, ic, id_))
    return out


def cell_volumes(points_xyz, faces, owner, neighbour):
    """Signed hex volumes from face pyramids. Owner Sf points owner→neighbour."""
    pts = np.asarray(points_xyz, dtype=float)
    n_own = [int(i) for i in owner]
    n_nei = [int(i) for i in neighbour]
    if not n_own:
        return np.zeros(0)
    n_cells = 1 + max(n_own + n_nei)
    vol = np.zeros(n_cells, dtype=float)
    n_int = len(n_nei)

    def face_cf_sf(f):
        xyz = pts[[int(i) for i in f]]
        c = xyz.mean(axis=0)
        sf = np.zeros(3)
        n = len(xyz)
        for i in range(n):
            sf += np.cross(xyz[i] - c, xyz[(i + 1) % n] - c)
        sf *= 0.5
        return c, sf

    for f, o, n in zip(faces[:n_int], n_own[:n_int], n_nei):
        c, sf = face_cf_sf(f)
        d = float(np.dot(c, sf) / 3.0)
        vol[int(o)] += d
        vol[int(n)] -= d
    for f, o in zip(faces[n_int:], n_own[n_int:]):
        c, sf = face_cf_sf(f)
        vol[int(o)] += float(np.dot(c, sf) / 3.0)
    return vol


def weld_xy(points_xy, tol: float = WELD_TOL) -> tuple[np.ndarray, np.ndarray]:
    """Unique (x,y) and inverse index. Rounding 1e-9."""
    pts = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    key = np.round(pts / tol) * tol
    uniq: list[tuple[float, float]] = []
    inv = np.empty(len(pts), dtype=int)
    seen: dict[tuple[int, int], int] = {}
    scale = 1.0 / tol
    for i, (x, y) in enumerate(key):
        ik = (int(round(x * scale)), int(round(y * scale)))
        j = seen.get(ik)
        if j is None:
            j = len(uniq)
            seen[ik] = j
            uniq.append((float(pts[i, 0]), float(pts[i, 1])))
        inv[i] = j
    return np.asarray(uniq, dtype=float), inv


def _fmt_vec(p) -> str:
    return f"({p[0]:.12g} {p[1]:.12g} {p[2]:.12g})"


def _fmt_face(ids: Iterable[int]) -> str:
    ids = list(ids)
    return f"{len(ids)}(" + " ".join(str(int(i)) for i in ids) + ")"


def extrude_quads(
    points_xy,
    quads,
    z_thick: float = Z_THICK_DEFAULT,
) -> tuple[np.ndarray, list[list[int]], list[int], list[int], np.ndarray, dict]:
    """Extrude CCW 2-D quads to hexes.

    Returns points_xyz, faces (internal then later caller may append boundary),
    owner, neighbour, cell_centres, and edge_map for patch tagging.

    edge_map: frozenset({a,b}) -> {owner, oriented (a,b) first-seen}
    """
    xy, _ = weld_xy(points_xy)
    n2 = int(xy.shape[0])
    z0 = np.column_stack([xy, np.zeros(n2)])
    z1 = np.column_stack([xy, np.full(n2, float(z_thick))])
    points_xyz = np.vstack([z0, z1])

    def z0i(k: int) -> int:
        return int(k)

    def z1i(k: int) -> int:
        return int(k) + n2

    # Weld quad corners onto unique xy
    raw = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    _, inv_raw = weld_xy(raw)
    # quads index into the *input* points_xy list; map through weld of that list
    # Caller should pass already-welded indices. Support both.
    qds = [tuple(int(v) for v in q) for q in quads]

    internal: list[tuple[list[int], int, int]] = []
    edge_owner: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    # key (min,max) -> (cell, (a,b) oriented as first cell walked it)

    cell_cc = []
    for ci, (a, b, c, d) in enumerate(qds):
        # clip to unique if they passed welded ids
        a, b, c, d = int(a), int(b), int(c), int(d)
        pts2 = xy[[a, b, c, d]]
        cx = float(pts2[:, 0].mean())
        cy = float(pts2[:, 1].mean())
        cell_cc.append((cx, cy, 0.5 * float(z_thick)))
        for e0, e1 in ((a, b), (b, c), (c, d), (d, a)):
            key = (min(e0, e1), max(e0, e1))
            if key in edge_owner:
                cj, (f0, f1) = edge_owner.pop(key)
                # first cell owns oriented (f0,f1) → face (f0,f1,f1+N,f0+N)
                face = [z0i(f0), z0i(f1), z1i(f1), z1i(f0)]
                own, nei = cj, ci
                if own > nei:
                    own, nei = nei, own
                    face = [face[0], face[3], face[2], face[1]]
                internal.append((face, own, nei))
            else:
                edge_owner[key] = (ci, (e0, e1))

    faces: list[list[int]] = [f for f, _, _ in internal]
    owner = [o for _, o, _ in internal]
    neighbour = [n for _, _, n in internal]
    boundary_edges = {k: v for k, v in edge_owner.items()}
    return (
        points_xyz,
        faces,
        owner,
        neighbour,
        np.asarray(cell_cc, dtype=float),
        {"n2": n2, "boundary_edges": boundary_edges, "n_cells": len(qds), "xy": xy},
    )


def _side_face(a: int, b: int, n2: int) -> list[int]:
    return [a, b, b + n2, a + n2]


def extrude_and_classify(
    points_xy,
    quads,
    patches_2d: dict[str, list[tuple[int, int]]],
    z_thick: float = Z_THICK_DEFAULT,
) -> dict[str, Any]:
    """Full extrusion with named 2-D boundary edges → 3-D patches + empty front/back."""
    xy, inv = weld_xy(points_xy)
    # If quads index the input array:
    n_in = len(np.asarray(points_xy).reshape(-1, 2))
    if n_in != len(xy):
        qds = []
        for q in quads:
            qds.append(tuple(int(inv[int(i)]) for i in q))
        mapped_patches = {}
        for name, edges in patches_2d.items():
            mapped_patches[name] = [(int(inv[int(a)]), int(inv[int(b)])) for a, b in edges]
        patches_2d = mapped_patches
        quads = qds
        points_xy = xy

    pts, faces, owner, neighbour, ccs, meta = extrude_quads(points_xy, quads, z_thick)
    n2 = meta["n2"]
    n_cells = meta["n_cells"]
    boundary_edges = meta["boundary_edges"]

    # Tag remaining 2-D edges
    edge_to_patch: dict[tuple[int, int], str] = {}
    for name, edges in patches_2d.items():
        for a, b in edges:
            edge_to_patch[(min(int(a), int(b)), max(int(a), int(b)))] = name

    patch_faces: dict[str, list[list[int]]] = {n: [] for n in PATCH_ORDER}
    leftover = []
    for key, (ci, (e0, e1)) in boundary_edges.items():
        name = edge_to_patch.get(key)
        if name is None:
            leftover.append("skip")
            continue
        face = _side_face(e0, e1, n2)
        if name not in patch_faces:
            patch_faces[name] = []
        patch_faces[name].append(face)
        leftover.append(name)

    # frontAndBack: every cell
    xy_arr = meta["xy"]
    for ci, q in enumerate(quads):
        a, b, c, d = (int(v) for v in q)
        # back z=0 normal -z: (a,d,c,b)
        patch_faces["frontAndBack"].append([a, d, c, b])
        # front z=Z normal +z
        patch_faces["frontAndBack"].append([a + n2, b + n2, c + n2, d + n2])

    # Append boundary faces in PATCH_ORDER
    start = {}
    nfaces_p = {}
    for name in PATCH_ORDER:
        fl = patch_faces.get(name) or []
        start[name] = len(faces)
        nfaces_p[name] = len(fl)
        for f in fl:
            faces.append(f)
            owner.append(0)  # filled below from cell via... we need owner cell
    # Fix owners for boundary: recompute from classification
    # Rebuild owner for boundary portion
    owner = owner[: len(neighbour)]  # internals only so far — wait we already appended
    # Redo cleanly
    return _assemble_lists(pts, quads, n2, n_cells, ccs, meta, patches_2d, z_thick)


def _assemble_lists(pts, quads, n2, n_cells, ccs, meta, patches_2d, z_thick):
    """Second pass: internals then patches in order, correct owners."""
    xy = meta["xy"]
    # rebuild internals + boundary with owners
    edge_owner = {}
    internals = []
    for ci, (a, b, c, d) in enumerate(quads):
        a, b, c, d = int(a), int(b), int(c), int(d)
        for e0, e1 in ((a, b), (b, c), (c, d), (d, a)):
            key = (min(e0, e1), max(e0, e1))
            if key in edge_owner:
                cj, (f0, f1) = edge_owner.pop(key)
                face = [f0, f1, f1 + n2, f0 + n2]
                own, nei = cj, ci
                if own > nei:
                    own, nei = nei, own
                    face = [face[0], face[3], face[2], face[1]]
                internals.append((face, own, nei))
            else:
                edge_owner[key] = (ci, (e0, e1))

    edge_to_patch = {}
    for name, edges in patches_2d.items():
        for a, b in edges:
            edge_to_patch[(min(int(a), int(b)), max(int(a), int(b)))] = name

    patch_items: dict[str, list[tuple[list[int], int]]] = {n: [] for n in PATCH_ORDER}
    n_leftover = 0
    for key, (ci, (e0, e1)) in edge_owner.items():
        name = edge_to_patch.get(key)
        if name is None:
            n_leftover += 1
            continue
        if name not in patch_items:
            patch_items[name] = []
        face = [e0, e1, e1 + n2, e0 + n2]
        patch_items[name].append((face, ci))
    meta["n_leftover_untagged"] = n_leftover

    for ci, q in enumerate(quads):
        a, b, c, d = (int(v) for v in q)
        patch_items["frontAndBack"].append(([a, d, c, b], ci))
        patch_items["frontAndBack"].append(([a + n2, b + n2, c + n2, d + n2], ci))

    faces: list[list[int]] = []
    owner: list[int] = []
    neighbour: list[int] = []
    for face, own, nei in internals:
        faces.append(face)
        owner.append(own)
        neighbour.append(nei)
    n_int = len(neighbour)
    patch_meta = {}
    for name in list(PATCH_ORDER) + [k for k in patch_items if k not in PATCH_ORDER]:
        items = patch_items.get(name) or []
        patch_meta[name] = {"start": len(faces), "nFaces": len(items)}
        for face, ci in items:
            faces.append(face)
            owner.append(ci)
    return {
        "points_xyz": pts if pts is not None else None,
        "faces": faces,
        "owner": owner,
        "neighbour": neighbour,
        "cell_centres": ccs,
        "n_internal": n_int,
        "n_cells": n_cells,
        "n2": n2,
        "patches": patch_meta,
        "xy": xy,
    }


def write_polymesh(
    points_xy=None,
    quads=None,
    patches=None,
    z_thick: float = Z_THICK_DEFAULT,
    out_dir=None,
    cyclic_sep_y: float = 0.0,
    *,
    points_xyz=None,
    faces=None,
    owner=None,
    neighbour=None,
    patches_dict=None,
) -> Path:
    """Write constant/polyMesh. 2-D quad path or pre-extruded lists."""
    out = Path(out_dir)
    mesh = out / "constant" / "polyMesh"

    if points_xyz is None:
        xy, inv = weld_xy(points_xy)
        n_in = len(np.asarray(points_xy).reshape(-1, 2))
        qds = list(quads)
        p2d = dict(patches or {})
        if n_in != len(xy):
            qds = [tuple(int(inv[int(i)]) for i in q) for q in quads]
            p2d = {n: [(int(inv[int(a)]), int(inv[int(b)])) for a, b in ed] for n, ed in p2d.items()}
        qds = orient_quads(xy, qds)
        if not qds:
            raise RuntimeError("HOH: all quads degenerate after orient")
        pts, _, _, _, ccs, meta = extrude_quads(xy, qds, z_thick)
        built = _assemble_lists(pts, qds, meta["n2"], meta["n_cells"], ccs, meta, p2d, z_thick)
        points_xyz = built["points_xyz"] if built["points_xyz"] is not None else pts
        faces = built["faces"]
        owner = built["owner"]
        neighbour = built["neighbour"]
        patches_dict = built["patches"]
        n_cells = len(qds)
        built["n_cells"] = n_cells
        vols = cell_volumes(points_xyz, faces, owner, neighbour)
        vmin = float(np.min(vols)) if len(vols) else -1.0
        if vmin <= 0.0:
            raise RuntimeError(
                f"HOH hex volume <= 0 (min={vmin:.3e} n={len(vols)} leftover={meta.get('n_leftover_untagged')}). "
                "Refusing to write polyMesh."
            )
    else:
        n_cells = 1 + max(owner)

    mesh.mkdir(parents=True, exist_ok=True)
    points_xyz = np.asarray(points_xyz, dtype=float)
    n_pts = len(points_xyz)
    n_faces = len(faces)
    n_int = len(neighbour)

    def dump_list(path: Path, cls: str, obj: str, n: int, lines: list[str]) -> None:
        body = write_foam_header(cls, obj) + f"\n{n}\n(\n" + "\n".join(lines) + "\n)\n"
        path.write_text(body)

    dump_list(
        mesh / "points",
        "vectorField",
        "points",
        n_pts,
        [_fmt_vec(p) for p in points_xyz],
    )
    dump_list(
        mesh / "faces",
        "faceList",
        "faces",
        n_faces,
        [_fmt_face(f) for f in faces],
    )
    dump_list(
        mesh / "owner",
        "labelList",
        "owner",
        n_faces,
        [str(int(i)) for i in owner],
    )
    dump_list(
        mesh / "neighbour",
        "labelList",
        "neighbour",
        n_int,
        [str(int(i)) for i in neighbour],
    )

    # boundary
    y = float(cyclic_sep_y)
    lines = [write_foam_header("polyBoundaryMesh", "boundary"), f"{len(PATCH_ORDER)}\n(\n"]
    for name in PATCH_ORDER:
        info = (patches_dict or {}).get(name) or {"start": 0, "nFaces": 0}
        start = int(info["start"])
        nf = int(info["nFaces"])
        if name == "bottom":
            typ = (
                f"    {name}\n    {{\n"
                f"        type            cyclic;\n"
                f"        neighbourPatch  top;\n"
                f"        transform       translational;\n"
                f"        separationVector (0 {y:.12g} 0);\n"
                f"        nFaces          {nf};\n"
                f"        startFace       {start};\n"
                f"    }}\n"
            )
        elif name == "top":
            typ = (
                f"    {name}\n    {{\n"
                f"        type            cyclic;\n"
                f"        neighbourPatch  bottom;\n"
                f"        transform       translational;\n"
                f"        separationVector (0 {-y:.12g} 0);\n"
                f"        nFaces          {nf};\n"
                f"        startFace       {start};\n"
                f"    }}\n"
            )
        elif name == "blades":
            typ = (
                f"    {name}\n    {{\n"
                f"        type            wall;\n"
                f"        nFaces          {nf};\n"
                f"        startFace       {start};\n"
                f"    }}\n"
            )
        elif name == "frontAndBack":
            typ = (
                f"    {name}\n    {{\n"
                f"        type            empty;\n"
                f"        nFaces          {nf};\n"
                f"        startFace       {start};\n"
                f"    }}\n"
            )
        else:
            typ = (
                f"    {name}\n    {{\n"
                f"        type            patch;\n"
                f"        nFaces          {nf};\n"
                f"        startFace       {start};\n"
                f"    }}\n"
            )
        lines.append(typ)
    lines.append(")\n")
    (mesh / "boundary").write_text("".join(lines))
    return mesh
