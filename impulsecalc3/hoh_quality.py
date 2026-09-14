"""HOH mesh gates. Rectangle outer is a fail."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .hoh_stop import (
    MAX_NONORTHO_DEG,
    N_CELLS_CAP,
    N_CELLS_MIN,
    RECTANGLE_YSTDEV_OVER_PITCH,
    HohStop,
    assert_not_rectangle,
)


def face_normal(face_pts) -> np.ndarray:
    p = np.asarray(face_pts, dtype=float)
    n = np.cross(p[1] - p[0], p[2] - p[0])
    L = float(np.linalg.norm(n))
    if L < 1e-18:
        return np.zeros(3)
    return n / L


def nonortho_deg(owner_cc, neighbour_cc, face_n) -> float:
    d = np.asarray(neighbour_cc, dtype=float) - np.asarray(owner_cc, dtype=float)
    n = np.asarray(face_n, dtype=float)
    dn = float(np.linalg.norm(d) * np.linalg.norm(n))
    if dn < 1e-18:
        return 90.0
    c = abs(float(np.dot(n, d))) / dn
    c = min(1.0, max(0.0, c))
    return float(math.degrees(math.acos(c)))


def skewness(owner_cc, neighbour_cc, face_centre, face_n) -> float:
    """Screening skew: how far fc-to-cc misses the face normal."""
    n = np.asarray(face_n, dtype=float)
    ln = float(np.linalg.norm(n))
    if ln < 1e-18:
        return 99.0
    n = n / ln
    fc = np.asarray(face_centre, dtype=float)
    d0 = np.asarray(owner_cc, dtype=float) - fc
    d1 = np.asarray(neighbour_cc, dtype=float) - fc
    # OpenFOAM-ish: intersection miss / face-to-cc
    t0 = abs(float(np.dot(d0, n)))
    t1 = abs(float(np.dot(d1, n)))
    miss0 = float(np.linalg.norm(d0 - n * np.dot(d0, n)))
    miss1 = float(np.linalg.norm(d1 - n * np.dot(d1, n)))
    denom = max(t0 + t1, 1e-18)
    return float((miss0 + miss1) / denom)


def cyclic_translate_error(bottom_xyz, top_xyz, Y: float) -> float:
    b = np.asarray(bottom_xyz, dtype=float)
    t = np.asarray(top_xyz, dtype=float)
    if b.shape != t.shape:
        return 1.0
    delta = t - b
    if b.ndim == 2 and b.shape[1] == 2:
        err = np.abs(delta - np.array([0.0, float(Y)]))
    else:
        pad = np.zeros(b.shape[-1], dtype=float)
        pad[1] = float(Y)
        err = np.abs(delta - pad)
    return float(err.max())


def is_rectangle_outer(P_bot, pitch: float) -> bool:
    y = np.asarray(P_bot, dtype=float)[:, 1]
    if len(y) < 2:
        return True
    return float(np.std(y)) <= RECTANGLE_YSTDEV_OVER_PITCH * float(pitch)


def cyclic_ystdev_over_pitch(P_bot, pitch: float) -> float:
    y = np.asarray(P_bot, dtype=float)[:, 1]
    return float(np.std(y) / max(float(pitch), 1e-18))


def gates(scratch: dict[str, Any]) -> dict[str, Any]:
    """pass/fail dict. Does not raise maxNonOrtho. Does not AMI."""
    n_blades = int(scratch.get("n_blades") or 0)
    n_cells = int(scratch.get("n_cells") or 0)
    max_no = float(scratch.get("max_nonortho_deg") or 99.0)
    max_sk = float(scratch.get("max_skew") or 99.0)
    ystd = float(scratch.get("cyclic_ystdev_over_pitch") or 0.0)
    n_closed = int(scratch.get("n_closed_blade_loops") or 0)
    cyc_err = float(scratch.get("cyclic_translate_error") or 0.0)
    mesh_s = float(scratch.get("mesh_seconds") or 0.0)
    metal = float(scratch.get("metal_nonortho_deg") or 0.0)
    min_det = float(scratch.get("min_det") or 1.0)
    tagged = list(scratch.get("tagged_singularities") or [])
    nf_bot = int(scratch.get("nfaces_bottom") or 0)
    nf_top = int(scratch.get("nfaces_top") or 0)
    max_untagged = float(scratch.get("max_nonortho_untagged", max_no))
    ok_no = max_untagged <= MAX_NONORTHO_DEG and len(tagged) <= 8
    checks = {
        "n_blades_3": n_blades == 3,
        "three_closed_loops": n_closed == 3,
        "n_cells_band": N_CELLS_MIN <= n_cells <= N_CELLS_CAP,
        "max_nonortho": ok_no,
        "metal_nonortho": metal <= 25.0,
        "skew_internal": max_sk <= 4.0,
        "min_det": min_det >= 0.001,
        "cyclic_nfaces": nf_bot == nf_top and nf_bot > 0,
        "cyclic_translate": cyc_err < 1e-9,
        "curved_cyclic": ystd > RECTANGLE_YSTDEV_OVER_PITCH,
        "clock": mesh_s <= 180.0,
        "not_rectangle": not bool(scratch.get("rectangle_outer")),
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "max_nonortho_deg": max_no,
        "max_skew": max_sk,
        "n_cells": n_cells,
        "n_blades": n_blades,
        "cyclic_ystdev_over_pitch": ystd,
    }
