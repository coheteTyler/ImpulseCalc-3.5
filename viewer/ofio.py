"""Read OpenFOAM ascii polyMesh + volFields. Honest parsers, no interpolating T as p."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np


def time_dirs(case_dir: Path) -> list[tuple[float, Path]]:
    out = []
    for p in case_dir.iterdir():
        if not p.is_dir():
            continue
        try:
            t = float(p.name)
        except ValueError:
            continue
        if (p / "p").is_file() or (p / "U").is_file():
            out.append((t, p))
    out.sort(key=lambda kv: kv[0])
    return out


def _strip_header(text: str) -> str:
    # drop FoamFile { ... }
    m = re.search(r"FoamFile\s*\{.*?\}\s*", text, flags=re.S)
    if m:
        text = text[m.end() :]
    return text


def read_points(mesh_dir: Path) -> np.ndarray:
    text = _strip_header((mesh_dir / "points").read_text(encoding="utf-8", errors="replace"))
    m = re.search(r"(\d+)\s*\(", text)
    if not m:
        raise RuntimeError("points: no count")
    n = int(m.group(1))
    body = text[m.end() :]
    pts = []
    for m2 in re.finditer(r"\(([^)]+)\)", body):
        nums = m2.group(1).split()
        if len(nums) >= 3:
            pts.append((float(nums[0]), float(nums[1]), float(nums[2])))
        if len(pts) >= n:
            break
    return np.array(pts, dtype=float)


def read_faces(mesh_dir: Path) -> list[list[int]]:
    text = _strip_header((mesh_dir / "faces").read_text(encoding="utf-8", errors="replace"))
    m = re.search(r"(\d+)\s*\(", text)
    if not m:
        raise RuntimeError("faces: no count")
    n = int(m.group(1))
    body = text[m.end() :]
    faces = []
    for m2 in re.finditer(r"(\d+)\(([^)]+)\)", body):
        faces.append([int(x) for x in m2.group(2).split()])
        if len(faces) >= n:
            break
    return faces


def read_label_list(path: Path) -> list[int]:
    text = _strip_header(path.read_text(encoding="utf-8", errors="replace"))
    m = re.search(r"(\d+)\s*\(", text)
    if not m:
        return []
    n = int(m.group(1))
    body = text[m.end() :]
    nums = [int(x) for x in body.replace(")", " ").split() if re.fullmatch(r"-?\d+", x)]
    return nums[:n]


def cell_centres(case_dir: Path) -> np.ndarray:
    mesh = case_dir / "constant" / "polyMesh"
    points = read_points(mesh)
    faces = read_faces(mesh)
    owner = read_label_list(mesh / "owner")
    n_cells = max(owner) + 1 if owner else 0
    acc = np.zeros((n_cells, 3), dtype=float)
    w = np.zeros(n_cells, dtype=float)
    for fi, fv in enumerate(faces):
        if fi >= len(owner):
            break
        fc = points[fv].mean(axis=0)
        acc[owner[fi]] += fc
        w[owner[fi]] += 1.0
    neigh_path = mesh / "neighbour"
    if neigh_path.is_file():
        neigh = read_label_list(neigh_path)
        for fi, nb in enumerate(neigh):
            fv = faces[fi]
            fc = points[fv].mean(axis=0)
            acc[nb] += fc
            w[nb] += 1.0
    w = np.clip(w, 1.0, None)
    return acc / w[:, None]


def _parse_nonuniform_scalar(text: str, n_expect: int | None = None) -> list[float] | None:
    m = re.search(r"internalField\s+nonuniform\s+List<scalar>\s*(\d+)\s*\(", text)
    if not m:
        m2 = re.search(r"internalField\s+uniform\s+([^\s;]+)", text)
        if m2:
            v = float(m2.group(1))
            n = n_expect or 1
            return [v] * n
        return None
    n = int(m.group(1))
    body = text[m.end() :]
    vals = []
    for tok in body.replace(")", " ").split():
        try:
            vals.append(float(tok))
        except ValueError:
            if vals:
                break
        if len(vals) >= n:
            break
    return vals


def _parse_nonuniform_vector(text: str, n_expect: int | None = None) -> list[tuple[float, float, float]] | None:
    m = re.search(r"internalField\s+nonuniform\s+List<vector>\s*(\d+)\s*\(", text)
    if not m:
        m2 = re.search(r"internalField\s+uniform\s+\(([^)]+)\)", text)
        if m2:
            nums = [float(x) for x in m2.group(1).split()]
            n = n_expect or 1
            tup = (nums[0], nums[1], nums[2] if len(nums) > 2 else 0.0)
            return [tup] * n
        return None
    n = int(m.group(1))
    body = text[m.end() :]
    out = []
    for mm in re.finditer(r"\(([^)]+)\)", body):
        nums = [float(x) for x in mm.group(1).split()]
        if len(nums) >= 3:
            out.append((nums[0], nums[1], nums[2]))
        if len(out) >= n:
            break
    return out


def read_scalar_field(path: Path, n_cells: int | None = None) -> list[float] | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    return _parse_nonuniform_scalar(text, n_cells)


def read_vector_field(path: Path, n_cells: int | None = None) -> list[tuple[float, float, float]] | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    return _parse_nonuniform_vector(text, n_cells)


def patch_mean_from_field(path: Path, patch: str) -> float | None:
    """Mean of a uniform/nonuniform boundaryField patch (p)."""
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(rf"\b{re.escape(patch)}\s*\{{(.*?)\n\s*\}}", text, flags=re.S)
    if not m:
        return None
    block = m.group(1)
    mu = re.search(r"value\s+uniform\s+([^\s;]+)", block)
    if mu:
        try:
            return float(mu.group(1))
        except ValueError:
            return None
    mn = re.search(r"value\s+nonuniform\s+List<scalar>\s*(\d+)\s*\(", block)
    if not mn:
        return None
    n = int(mn.group(1))
    body = block[mn.end() :]
    vals = []
    for tok in body.replace(")", " ").split():
        try:
            vals.append(float(tok))
        except ValueError:
            if vals:
                break
        if len(vals) >= n:
            break
    if not vals:
        return None
    return sum(vals) / len(vals)


def parse_yplus(case_dir: Path, log_text: str = "") -> dict[str, Any]:
    """y+ after the solve: LAST time, never t=0, never a zeroed postProcess dump.

    The in-run yPlus FO writes wall y+ at writeTime into postProcessing/yPlus1.
    `postProcess -func yPlus` without the solver turbulence model overwrites the
    last-time volField (and a sibling yPlus/ dat) with zeros — that is not
    after-solve y+. Prefer the in-run FO table at the last time with max>0.
    """
    out: dict[str, Any] = {"source": None, "min": None, "max": None, "average": None, "time_s": None}
    tds = time_dirs(case_dir)
    last_t = tds[-1][0] if tds else None
    last_dir = tds[-1][1] if tds else None

    def _nonzero(mn, mx, avg) -> bool:
        try:
            return max(abs(float(mn)), abs(float(mx)), abs(float(avg))) > 0.0
        except (TypeError, ValueError):
            return False

    # 1. postProcessing yPlus*.dat — highest time with non-zero wall y+
    root = case_dir / "postProcessing"
    rows_all: list[tuple[float, float, float, float, str]] = []
    if root.is_dir():
        for pth in root.glob("yPlus*/**/*"):
            if not pth.is_file():
                continue
            if "yplus" not in pth.name.lower():
                continue
            try:
                txt = pth.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            for line in txt.splitlines():
                if line.strip().startswith("#") or not line.strip():
                    continue
                nums: list[float] = []
                for tok in line.replace("(", " ").replace(")", " ").split():
                    try:
                        nums.append(float(tok))
                    except ValueError:
                        pass
                if len(nums) >= 4:
                    t_row, mn, mx, avg = nums[0], nums[-3], nums[-2], nums[-1]
                    if t_row > 0 and _nonzero(mn, mx, avg):
                        rows_all.append((t_row, mn, mx, avg, str(pth)))
    if rows_all:
        tmax = max(r[0] for r in rows_all)
        at = [r for r in rows_all if abs(r[0] - tmax) / max(tmax, 1e-30) < 1e-8]
        at_pref = [r for r in at if "yPlus1" in r[4]] or at
        out.update(
            source=at_pref[0][4],
            min=float(min(r[1] for r in at_pref)),
            max=float(max(r[2] for r in at_pref)),
            average=float(sum(r[3] for r in at_pref) / len(at_pref)),
            time_s=float(tmax),
        )
        return out

    # 2. solver FO log last non-zero wall patch line (skip postProcess zeros)
    patch_hits = list(
        re.finditer(
            r"patch\s+blade\d+\s+y\+\s*:\s*min\s*=\s*([0-9.eE+-]+).*?max\s*=\s*([0-9.eE+-]+).*?average\s*=\s*([0-9.eE+-]+)",
            log_text,
            flags=re.I,
        )
    )
    good = [h for h in patch_hits if _nonzero(h.group(1), h.group(2), h.group(3))]
    if good:
        n_take = min(3, len(good))
        grp = good[-n_take:]
        out.update(
            source="functionObject_log",
            min=min(float(h.group(1)) for h in grp),
            max=max(float(h.group(2)) for h in grp),
            average=sum(float(h.group(3)) for h in grp) / len(grp),
            time_s=float(last_t) if last_t else None,
        )
        return out

    # 3. last-time volField only if max>0 (internalField is usually 0; wall on patches)
    if last_dir is not None:
        yp_path = last_dir / "yPlus"
        yp = read_scalar_field(yp_path)
        if yp:
            arr = np.array([v for v in yp if v == v], dtype=float)
            if arr.size and float(np.abs(arr).max()) > 0:
                out.update(
                    source=str(yp_path),
                    min=float(arr.min()),
                    max=float(arr.max()),
                    average=float(arr.mean()),
                    time_s=float(last_t),
                )
                return out
        if yp_path.is_file():
            text_f = yp_path.read_text(encoding="utf-8", errors="replace")
            mins, maxs, avgs = [], [], []
            for m in re.finditer(r"blade\d+\s*\{(.*?)\}", text_f, flags=re.S):
                block = m.group(1)
                mn = re.search(r"value\s+nonuniform\s+List<scalar>\s*(\d+)\s*\(", block)
                if not mn:
                    continue
                n = int(mn.group(1))
                vals: list[float] = []
                for tok in block[mn.end() :].replace(")", " ").split():
                    try:
                        vals.append(float(tok))
                    except ValueError:
                        if vals:
                            break
                    if len(vals) >= n:
                        break
                if vals and max(abs(v) for v in vals) > 0:
                    mins.append(min(vals))
                    maxs.append(max(vals))
                    avgs.append(sum(vals) / len(vals))
            if avgs:
                out.update(
                    source=str(yp_path),
                    min=float(min(mins)),
                    max=float(max(maxs)),
                    average=float(sum(avgs) / len(avgs)),
                    time_s=float(last_t),
                )
                return out
    return out


def parse_checkmesh(log: str) -> dict[str, Any]:
    failed = "Mesh OK." not in log and "Mesh OK" not in log
    n_fail = None
    m = re.search(r"Failed\s+(\d+)\s+mesh checks", log)
    if m:
        n_fail = int(m.group(1))
        failed = True
    if "Mesh OK." in log or re.search(r"\nMesh OK\.\s*$", log):
        failed = False
    neg_vol = "zero or negative pyramid volume" in log.lower() or "Min volume =" in log and "- " in (
        re.search(r"Min volume = ([^\.\s]+)", log).group(1) if re.search(r"Min volume = ([^\.\s]+)", log) else ""
    )
    num = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    mv = re.search(r"Min volume =\s*" + num, log)
    xv = re.search(r"Max volume =\s*" + num, log)
    no = re.search(r"Mesh non-orthogonality Max:\s*" + num + r"\s*average:\s*" + num, log)
    sk = re.search(r"Max skewness =\s*" + num, log)
    open_c = "Open cells found" in log
    wrong = re.search(r"(\d+) faces are incorrectly oriented", log)
    n_cells = re.search(r"cells:\s+(\d+)", log)
    hexes = re.search(r"hexahedra:\s+(\d+)", log)
    return {
        "mesh_ok_strict": (not failed) and (not open_c),
        "failed_checks": n_fail,
        "open_cells": open_c,
        "wrong_oriented_faces": int(wrong.group(1)) if wrong else 0,
        "min_volume": float(mv.group(1)) if mv else None,
        "max_volume": float(xv.group(1)) if xv else None,
        "nonortho_max": float(no.group(1)) if no else None,
        "nonortho_avg": float(no.group(2)) if no else None,
        "skew_max": float(sk.group(1)) if sk else None,
        "n_cells": int(n_cells.group(1)) if n_cells else None,
        "n_hex": int(hexes.group(1)) if hexes else None,
        "negative_volume": bool(mv and float(mv.group(1)) <= 0),
        "log_tail": "\n".join(log.strip().splitlines()[-40:]),
    }
