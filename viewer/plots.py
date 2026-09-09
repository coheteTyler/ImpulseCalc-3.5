"""2D matplotlib figures from real time directories. PNG sequence, no fake video."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _agg():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def wall_cp_png(
    path: Path,
    wall: dict[str, Any],
    *,
    p1_pa: float,
    q_dyn_pa: float,
    title: str = "Wall Cp (patch sample, not synthetic)",
) -> Path | None:
    if not wall.get("blades"):
        return None
    plt = _agg()
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=130)
    b = wall["blades"][0]
    drawn = False
    for name, col in (("ps", "#c0392b"), ("ss", "#2471a3")):
        chain = b.get(name) or []
        xs, cps = [], []
        for r in chain:
            if r.get("Cp") is None:
                continue
            xs.append(r.get("x_over_c", r["x"]))
            cps.append(r["Cp"])
        if xs:
            ax.plot(xs, cps, color=col, lw=1.4, label=name.upper())
            drawn = True
    if not drawn:
        plt.close(fig)
        return None
    ax.axhline(0.0, color="0.5", lw=0.6)
    ax.set_xlabel("x / c")
    ax.set_ylabel("Cp = (p - p1) / q1")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def force_history_png(path: Path, forces: dict[str, Any]) -> Path | None:
    hist = forces.get("history") or []
    if not hist:
        return None
    plt = _agg()
    fig, ax = plt.subplots(figsize=(8, 4.0), dpi=130)
    for h in hist:
        t = np.array(h["t"]) * 1e6
        ax.plot(t, h["Ft_N"], lw=1.3, label=f"blade{h['blade']} Ft")
    ax.set_xlabel("t [µs]")
    ax.set_ylabel("Ft [N] at span (scaled slab)")
    ax.set_title("Per-blade Ft history — PREDICTED until plateau")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def field_png(
    path: Path,
    cc: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    cbar: str,
    blade_polys: list | None = None,
    xlim_m: tuple[float, float] | None = None,
) -> Path | None:
    if cc.size == 0 or values.size == 0:
        return None
    plt = _agg()
    fig, ax = plt.subplots(figsize=(9.0, 4.6), dpi=130)
    x = cc[:, 0] * 1000.0
    y = cc[:, 1] * 1000.0
    sc = ax.scatter(x, y, c=values, s=6, cmap="coolwarm", linewidths=0)
    fig.colorbar(sc, ax=ax, label=cbar, shrink=0.85)
    if blade_polys:
        for poly in blade_polys:
            ax.plot([p[0] * 1000 for p in poly], [p[1] * 1000 for p in poly], "k", lw=0.8)
    ax.set_aspect("equal")
    ax.set_xlabel("x axial [mm]")
    ax.set_ylabel("y pitch [mm]")
    ax.set_title(title)
    if xlim_m is not None:
        ax.set_xlim(xlim_m[0] * 1000.0, xlim_m[1] * 1000.0)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def sequence_from_times(
    out_dir: Path,
    case_dir: Path,
    cc: np.ndarray,
    blade_polys: list | None,
    p1_pa: float,
) -> list[str]:
    from .ofio import read_scalar_field, read_vector_field, time_dirs

    written: list[str] = []
    seq = out_dir / "seq"
    seq.mkdir(parents=True, exist_ok=True)
    n = cc.shape[0]
    for t, tdir in time_dirs(case_dir):
        p = read_scalar_field(tdir / "p", n)
        u = read_vector_field(tdir / "U", n)
        tag = f"{t:.8g}".replace("-", "m").replace(".", "p")
        if p:
            pv = np.array(p, dtype=float)
            png = field_png(
                seq / f"p_{tag}.png",
                cc,
                pv,
                title=f"p  t={t:.4g} s  (real time dir)",
                cbar="p [Pa]",
                blade_polys=blade_polys,
            )
            if png:
                written.append(str(png))
        if u:
            mag = np.array([math_hyp(v) for v in u], dtype=float)
            png = field_png(
                seq / f"Umag_{tag}.png",
                cc,
                mag,
                title=f"|U|  t={t:.4g} s  (real time dir)",
                cbar="|U| [m/s]",
                blade_polys=blade_polys,
            )
            if png:
                written.append(str(png))
    return written


def math_hyp(v: tuple[float, float, float]) -> float:
    return float((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) ** 0.5)


def triangle_png(path: Path, ml: Any, *, title: str = "Velocity triangles — SCOPING / PREDICTED") -> Path | None:
    """Inlet/outlet C, W, U from meanline. Not OpenFOAM."""
    plt = _agg()
    fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=130)
    u = float(ml.u_m_s)
    # left inlet origin (0,0); right outlet origin shifted by 1.15*max(C,W)
    scale_shift = 1.15 * max(ml.c1_m_s, ml.c2_m_s, ml.w1_m_s) 
    ox = scale_shift

    def arrow(x0, y0, dx, dy, color, label):
        ax.annotate(
            "",
            xy=(x0 + dx, y0 + dy),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.6),
        )
        ax.plot([], [], color=color, lw=1.6, label=label)

    arrow(0, 0, ml.cx1, ml.cy1, "#1e8449", "C1")
    arrow(0, 0, ml.wx1, ml.wy1, "#c0392b", "W1")
    arrow(ml.wx1, ml.wy1, 0.0, u, "#2471a3", "U")
    arrow(ox, 0, ml.cx2, ml.cy2, "#196f3d", "C2")
    arrow(ox, 0, ml.wx2, ml.wy2, "#922b21", "W2")
    arrow(ml.wx2 + ox, ml.wy2, 0.0, u, "#1a5276", "U")
    ax.set_aspect("equal")
    span=max(abs(ml.c1_m_s), abs(ml.c2_m_s), abs(ml.w1_m_s), abs(u), 1.0)
    ax.set_xlim(-0.25*span, ox+abs(ml.cx2)+0.25*span)
    ax.set_ylim(min(0.0, ml.cy2, ml.wy2, ml.wy2+u)-0.2*span, max(0.0, ml.cy1, ml.wy1+u)+0.2*span)
    ax.set_xlabel("axial component [m/s]")
    ax.set_ylabel("tangential component [m/s]")
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    by = dict(zip(labels, handles))
    ax.legend(by.values(), by.keys(), fontsize=8, loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
