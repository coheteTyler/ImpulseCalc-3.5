"""VTK + matplotlib 2D p and |U| contours + wall Cp. Real time directories only."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .ofenv import foam_env, run_foam


def _crop_cascade_ax(ax, job: dict[str, Any] | None):
    """Cascade plots crop to x in [-1.5c, 2c]; dump to 6c is computational, not the view."""
    if not job:
        return None
    try:
        from .job import cascade_view_xlim_m
        x0, x1 = cascade_view_xlim_m(job)
    except Exception:
        return None
    lim = (x0 * 1000.0, x1 * 1000.0)
    ax.set_xlim(*lim)
    return lim


def foam_time_dirs(case_dir: Path) -> list[str]:
    out = []
    for p in case_dir.iterdir():
        if not p.is_dir():
            continue
        try:
            float(p.name)
        except ValueError:
            continue
        if (p / "p").is_file() or (p / "p.gz").is_file():
            out.append(p.name)
    return sorted(out, key=lambda s: float(s))


def _boundary_start_nfaces(case_dir: Path) -> dict[str, tuple[int, int]]:
    """polyMesh/boundary name -> (startFace, nFaces)."""
    import re

    path = Path(case_dir) / "constant" / "polyMesh" / "boundary"
    out: dict[str, tuple[int, int]] = {}
    if not path.is_file():
        return out
    txt = path.read_text(encoding="utf-8", errors="replace")
    for m in re.finditer(
        r"(?m)^\s*([A-Za-z_][\w]*)\s*\n\s*\{(.*?)\n\s*\}",
        txt,
        flags=re.S,
    ):
        name, block = m.group(1), m.group(2)
        nf = re.search(r"nFaces\s+(\d+)", block)
        sf = re.search(r"startFace\s+(\d+)", block)
        if nf and sf:
            out[name] = (int(sf.group(1)), int(nf.group(1)))
    return out


def _median(vals: list[float]) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    if n % 2:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


_OF_STATION_CACHE: dict[str, Any] = {"key": None, "data": None}


def extract_of_stations(case_dir: Path, job: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Latest-time inlet/outlet |U| and |U|/a. Not η. Does not call the solver.

    Probe: owner cells of inlet/outlet patches (interior, not the inlet fixedValue BC).
    Fallback: median over cells in the first/last 12% of x-span. Relative-frame cascade
    so |U| ≈ |W|. PREDICTED until STAND.
    """
    case_dir = Path(case_dir)
    if not case_dir.is_dir():
        return None
    from viewer.ofio import cell_centres, read_label_list, read_scalar_field, read_vector_field, time_dirs

    tds = time_dirs(case_dir)
    tds = [(t, d) for t, d in tds if t > 0]
    if not tds:
        return None
    t_latest, tdir = tds[-1]
    u_path = tdir / "U"
    if not u_path.is_file():
        return None
    key = (str(case_dir.resolve()), str(t_latest), u_path.stat().st_mtime)
    if _OF_STATION_CACHE["key"] == key and _OF_STATION_CACHE["data"] is not None:
        return _OF_STATION_CACHE["data"]
    try:
        cc = cell_centres(case_dir)
    except Exception:
        cc = None
        # existing C file from writeCellCentres
        cvals = _parse_of_vector(tdir / "C")
        if cvals:
            cc = __import__("numpy").array(cvals, dtype=float)
    if cc is None or len(cc) == 0:
        return None
    n = len(cc)
    uvals = read_vector_field(tdir / "U", n)
    if not uvals or len(uvals) != n:
        uvals = _parse_of_vector(tdir / "U")
    if not uvals or len(uvals) != n:
        return None
    pvals = read_scalar_field(tdir / "p", n) or _parse_of_scalar(tdir / "p")
    rhovals = read_scalar_field(tdir / "rho", n) or _parse_of_scalar(tdir / "rho")
    tvals = read_scalar_field(tdir / "T", n) or _parse_of_scalar(tdir / "T")
    gamma = 1.33
    rspec = 287.0
    if job and isinstance(job.get("gas"), dict):
        try:
            gamma = float(job["gas"].get("gamma") or gamma)
        except (TypeError, ValueError):
            pass
        try:
            rspec = float(job["gas"].get("r_specific_j_kg_k") or rspec)
        except (TypeError, ValueError):
            pass
    mag = []
    a_loc = []
    for i, uv in enumerate(uvals):
        m = math.hypot(float(uv[0]), float(uv[1]))
        mag.append(m)
        a = None
        if rhovals and pvals and i < len(rhovals) and i < len(pvals):
            rho = max(float(rhovals[i]), 1e-12)
            a = math.sqrt(max(gamma * float(pvals[i]) / rho, 0.0))
        elif tvals and i < len(tvals):
            a = math.sqrt(max(gamma * rspec * float(tvals[i]), 0.0))
        a_loc.append(a)
    owner_path = case_dir / "constant" / "polyMesh" / "owner"
    owner = read_label_list(owner_path) if owner_path.is_file() else []
    patches = _boundary_start_nfaces(case_dir)
    probe = (
        "owner cells of inlet/outlet patches at latest time "
        f"t={t_latest} s (interior; not inlet fixedValue BC). Relative-frame |U|≈|W|."
    )

    def _ids_for(patch: str, *, inlet: bool) -> list[int]:
        ids: list[int] = []
        if patch in patches and owner:
            start, nf = patches[patch]
            for fi in range(start, start + nf):
                if fi < len(owner):
                    cid = owner[fi]
                    if 0 <= cid < n:
                        ids.append(cid)
            # unique, preserve order
            seen: set[int] = set()
            uniq = []
            for i in ids:
                if i not in seen:
                    seen.add(i)
                    uniq.append(i)
            if uniq:
                return uniq
        xs = [float(cc[i][0]) for i in range(n)]
        xmin, xmax = min(xs), max(xs)
        span = max(xmax - xmin, 1e-16)
        frac = 0.12
        out = []
        for i, x in enumerate(xs):
            if inlet and x <= xmin + frac * span:
                out.append(i)
            if (not inlet) and x >= xmax - frac * span:
                out.append(i)
        return out

    in_ids = _ids_for("inlet", inlet=True)
    out_ids = _ids_for("outlet", inlet=False)
    used_patch = bool(patches.get("inlet") and patches.get("outlet") and owner)
    if not used_patch:
        probe = (
            f"median |U| over cells in first/last 12% of x-span at t={t_latest} s "
            "(patch owner list missing). Relative-frame |U|≈|W|."
        )

    def _station(ids: list[int]) -> dict[str, Any]:
        wm = _median([mag[i] for i in ids]) if ids else None
        mw = None
        machs = []
        for i in ids:
            if a_loc[i] and a_loc[i] > 1e-9:
                machs.append(mag[i] / a_loc[i])
        mw = _median(machs)
        return {"n": len(ids), "w_m_s": wm, "Mw": mw}

    st_in = _station(in_ids)
    st_out = _station(out_ids)
    w1, w2 = st_in.get("w_m_s"), st_out.get("w_m_s")
    ratio = None
    if w1 and abs(float(w1)) > 1e-12 and w2 is not None:
        ratio = float(w2) / float(w1)
    data: dict[str, Any] = {
        "time_s": float(t_latest),
        "probe": probe,
        "n_inlet": st_in["n"],
        "n_outlet": st_out["n"],
        "w_inlet_m_s": st_in["w_m_s"],
        "w_outlet_m_s": st_out["w_m_s"],
        "w2_over_w1": ratio,
        "Mw1_OF": st_in["Mw"],
        "Mw_outlet_OF": st_out["Mw"],
        "predicted": True,
        "eta_from_cfd": None,
        "note": "OpenFOAM station extract. Not stage η. Not a KO shock term.",
    }
    _OF_STATION_CACHE["key"] = key
    _OF_STATION_CACHE["data"] = data
    return data


def write_vtk(case_dir: Path) -> dict[str, Any]:
    env = foam_env()
    rc = run_foam(["foamToVTK", "-latestTime"], case_dir, "log.foamToVTK", env)
    vtk_dir = case_dir / "VTK"
    n = len(list(vtk_dir.glob("**/*"))) if vtk_dir.is_dir() else 0
    return {"rc": rc, "vtk_dir": str(vtk_dir) if vtk_dir.is_dir() else None, "n_files": n}


def _read_internal_field_xy(case_dir: Path, time_name: str) -> dict[str, Any] | None:
    """Very small 2D cell-centre dump from OF cellCentres + p,U if present.

    Falls back to None; matplotlib then skips the contour and still writes wall Cp.
    """
    return None


def write_plots(
    out_dir: Path,
    case_dir: Path,
    job: dict[str, Any],
    forces: dict[str, Any],
    sample: dict[str, Any],
    ml: dict[str, Any],
) -> dict[str, str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        return {"error": f"matplotlib missing: {e}"}

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    gas = job["gas"]
    p1 = float(gas["p1_pa"])
    w1 = float(gas["w1_m_s"])
    rho = float(ml.get("rho1_kg_m3") or p1 / (gas["r_specific_j_kg_k"] * gas["t1_k"]))
    q = 0.5 * rho * w1 * w1

    # Force history
    if forces.get("history"):
        fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
        for h in forces["history"]:
            ax.plot(h["t"], h["Ft_N"], label=f"blade{h['blade']} Ft")
            ax.plot(h["t"], h["Fd_N"], ls="--", label=f"blade{h['blade']} Fd")
        ax.set_xlabel("t [s]")
        ax.set_ylabel("N at span")
        ax.set_title(
            f"Ft/Fd  predicted={forces.get('predicted')}  climbing={forces.get('climbing')}"
        )
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fp = out_dir / "force_history.png"
        fig.savefig(fp)
        plt.close(fig)
        paths["force_history"] = str(fp)

    # Wall Cp
    if sample.get("on_wall") and sample.get("blades"):
        fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
        b0 = sample["blades"][0]
        for name, chain, col in (("PS", b0.get("ps") or [], "#c0392b"), ("SS", b0.get("ss") or [], "#2471a3")):
            if not chain:
                continue
            xc = [r.get("x_over_c", r["x"] / job["geometry"]["chord_m"]) for r in chain]
            cp = [(r["p"] - p1) / q for r in chain]
            ax.plot(xc, cp, color=col, lw=1.2, label=name)
        ax.set_xlabel("x/c")
        ax.set_ylabel("Cp = (p-p1)/q_ref")
        ax.set_title("blade0 wall Cp  (p from wall patch, not a chord-line cut)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fp = out_dir / "wall_cp_blade0.png"
        fig.savefig(fp)
        plt.close(fig)
        paths["wall_cp"] = str(fp)

        # p(s) CSV
        csv = out_dir / "wall_p_blade0.csv"
        with csv.open("w", encoding="utf-8") as fh:
            fh.write("side,s_m,x_m,y_m,x_over_c,p_pa,Cp\n")
            for name, chain in (("PS", b0.get("ps") or []), ("SS", b0.get("ss") or [])):
                for r in chain:
                    xc = r.get("x_over_c", r["x"] / job["geometry"]["chord_m"])
                    cp = (r["p"] - p1) / q
                    fh.write(
                        f"{name},{r.get('s_m', 0)},{r['x']},{r['y']},{xc},{r['p']},{cp}\n"
                    )
        paths["wall_p_csv"] = str(csv)

    # Contours from reconstructed cell centres via postProcess? Try foamToVTK already done.
    # Read VTK is heavy; instead sample a reconstructed internal field with a Python
    # walk of the latest time using `postProcess -func writeCellCentres` if present.
    # Minimal 2D contour: interpolate scattered wall+inlet is not a field.
    # Use reconstructed points from constant/polyMesh + latest p if we can parse.
    paths.update(_try_field_contours(out_dir, case_dir, p1, w1, job=job) or {})
    return paths


def _internal_block(txt: str, kind: str) -> str | None:
    """Slice OpenFOAM internalField list. Do not stop at the first vector ')'."""
    import re

    m = re.search(
        r"internalField\s+nonuniform\s+List<" + kind + r">\s+\d+\s+\(",
        txt,
    )
    if not m:
        return None
    rest = txt[m.end() :]
    end = rest.find("\n)")
    if end < 0:
        return None
    return rest[:end]


def _parse_of_scalar(path: Path) -> list[float] | None:
    if not path.is_file():
        return None
    txt = path.read_text(encoding="utf-8", errors="replace")
    if "nonuniform" not in txt:
        return None
    block = _internal_block(txt, "scalar")
    if block is None:
        return None
    vals = []
    for tok in block.split():
        try:
            vals.append(float(tok))
        except ValueError:
            continue
    return vals or None


def _parse_of_vector(path: Path) -> list[tuple[float, float, float]] | None:
    if not path.is_file():
        return None
    txt = path.read_text(encoding="utf-8", errors="replace")
    import re

    block = _internal_block(txt, "vector")
    if block is None:
        return None
    vals = []
    for a, b, c in re.findall(
        r"\(([-+eE0-9.]+)\s+([-+eE0-9.]+)\s+([-+eE0-9.]+)\)", block
    ):
        vals.append((float(a), float(b), float(c)))
    return vals or None


def _parse_points(path: Path) -> list[tuple[float, float, float]] | None:
    if not path.is_file():
        return None
    txt = path.read_text(encoding="utf-8", errors="replace")
    # skip header, find N\n(
    import re

    m = re.search(r"\n(\d+)\s*\n\(", txt)
    if not m:
        return None
    rest = txt[m.end() :]
    pts = []
    for line in rest.splitlines():
        line = line.strip()
        if line.startswith("(") and ")" in line:
            inner = line[1 : line.index(")")]
            a, b, c = inner.split()
            pts.append((float(a), float(b), float(c)))
    return pts or None


def _try_field_contours(out_dir: Path, case_dir: Path, p1: float, w1: float, job: dict[str, Any] | None = None) -> dict[str, str]:
    times = foam_time_dirs(case_dir)
    if not times:
        return {}
    latest = times[-1]
    tdir = case_dir / latest
    pvals = _parse_of_scalar(tdir / "p")
    uvals = _parse_of_vector(tdir / "U")
    tvals = _parse_of_scalar(tdir / "T")
    rhovals = _parse_of_scalar(tdir / "rho")
    # cell centres: write if missing
    cc_path = tdir / "C"
    if not cc_path.is_file():
        run_foam(["postProcess", "-func", "writeCellCentres", "-time", latest], case_dir, "log.writeCellCentres")
    cc = _parse_of_vector(tdir / "C")
    if not cc or not pvals or len(cc) != len(pvals):
        return {}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return {}
    x = np.array([c[0] for c in cc]) * 1000
    y = np.array([c[1] for c in cc]) * 1000
    p = np.array(pvals)
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=120)
    sc = ax.scatter(x, y, c=p / 1e5, s=6, cmap="coolwarm", linewidths=0)
    ax.set_aspect("equal")
    _crop_cascade_ax(ax, job)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"p [bar]  t={latest} s  (cell centres, real solve)")
    fig.colorbar(sc, ax=ax, label="p [bar]")
    if uvals and len(uvals) == len(cc):
        ux = np.array([u[0] for u in uvals])
        uy = np.array([u[1] for u in uvals])
        xmin = float(x.min())
        xspan = max(float(x.max() - xmin), 1e-9)
        inlet = x <= (xmin + 0.10 * xspan)
        if np.count_nonzero(inlet) >= 4:
            ys = np.linspace(float(y[inlet].min()), float(y[inlet].max()), 5)
            qx, qy, qu, qv = [], [], [], []
            for yi in ys:
                sel = np.where(inlet)[0]
                j = int(sel[np.argmin(np.abs(y[sel] - yi))])
                mag = math.hypot(float(ux[j]), float(uy[j]))
                if mag < 1e-6:
                    continue
                qx.append(float(x[j]))
                qy.append(float(y[j]))
                qu.append(float(ux[j]) / mag)
                qv.append(float(uy[j]) / mag)
            if qx:
                ax.quiver(
                    qx, qy, qu, qv,
                    color="#111",
                    angles="xy",
                    scale_units="xy",
                    scale=0.12,
                    width=0.008,
                    zorder=5,
                )
                ax.text(
                    xmin + 0.02 * xspan,
                    float(np.max(qy)) if qy else float(y.max()),
                    "inlet U (OF)",
                    fontsize=8,
                    color="#111",
                )
    fig.tight_layout()
    fp = out_dir / "contour_p.png"
    fig.savefig(fp)
    plt.close(fig)
    paths = {"contour_p": str(fp)}
    if uvals and len(uvals) == len(cc):
        ux = np.array([u[0] for u in uvals])
        uy = np.array([u[1] for u in uvals])
        umag = np.hypot(ux, uy)
        fig, ax = plt.subplots(figsize=(8, 4.2), dpi=120)
        sc = ax.scatter(x, y, c=umag, s=6, cmap="viridis", linewidths=0)
        ax.set_aspect("equal")
        _crop_cascade_ax(ax, job)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y pitch [mm]")
        ax.set_title(f"U field  |U| color + arrows  t={latest} s  (OF cell centres)")
        fig.colorbar(sc, ax=ax, label="|U| [m/s]")
        n = len(x)
        step = max(1, n // 140)
        mag = np.maximum(umag, 1e-9)
        ax.quiver(
            x[::step], y[::step], ux[::step] / mag[::step], uy[::step] / mag[::step],
            color="#111", angles="xy", scale_units="xy", scale=0.18, width=0.004, zorder=5,
        )
        fig.tight_layout()
        fu = out_dir / "contour_U.png"
        fig.savefig(fu)
        plt.close(fig)
        paths["contour_U"] = str(fu)
        import matplotlib.tri as mtri
        triang = mtri.Triangulation(x, y)
        xlo, xhi = float(x.min()), float(x.max())
        if job:
            try:
                from .job import cascade_view_xlim_m
                c0, c1 = cascade_view_xlim_m(job)
                xlo, xhi = c0 * 1000.0, c1 * 1000.0
            except Exception:
                pass
        xi = np.linspace(xlo, xhi, 180)
        yi = np.linspace(float(y.min()), float(y.max()), 140)
        X, Y = np.meshgrid(xi, yi)
        Ui = np.ma.filled(mtri.LinearTriInterpolator(triang, ux)(X, Y), np.nan)
        Vi = np.ma.filled(mtri.LinearTriInterpolator(triang, uy)(X, Y), np.nan)
        wx1 = float(np.median(ux[x <= (float(x.min()) + 0.12 * (float(x.max()) - float(x.min())))]))
        wy1 = float(np.median(uy[x <= (float(x.min()) + 0.12 * (float(x.max()) - float(x.min())))]))
        w1 = max(math.hypot(wx1, wy1), 1.0)
        departed = np.hypot(Ui - wx1, Vi - wy1) > 0.08 * w1
        inlet_strip = X <= (float(x.min()) + 0.18 * (float(x.max()) - float(x.min())))
        # Do not paint leftover IC / cloned inlet W downstream of the inlet strip.
        show = inlet_strip | departed
        Ui_s = np.where(show, Ui, np.nan)
        Vi_s = np.where(show, Vi, np.nan)
        speed = np.hypot(np.nan_to_num(Ui_s, nan=0.0), np.nan_to_num(Vi_s, nan=0.0))
        fig, ax = plt.subplots(figsize=(8, 4.2), dpi=140)
        cf = ax.contourf(X, Y, np.ma.masked_invalid(np.hypot(Ui_s, Vi_s)), levels=24, cmap="turbo")
        ax.streamplot(
            xi, yi,
            np.nan_to_num(Ui_s, nan=0.0), np.nan_to_num(Vi_s, nan=0.0),
            color=np.where(show, speed, 0.0), cmap="turbo", density=1.4, linewidth=0.85, arrowsize=0.85,
        )
        if job:
            try:
                from .geometry import center_in_pitch, profile_from_job, spec_from_job
                from .job import pitch_m as _pitch_m
                spec = spec_from_job(job)
                poly = profile_from_job(job, spec)
                pitch = _pitch_m(job)
                try:
                    poly0, _ = center_in_pitch(poly, pitch)
                except Exception:
                    poly0 = poly
                n_b = int(job["geometry"].get("n_blades_cascade") or 3)
                for k in range(n_b):
                    bx = [pt[0] * 1000.0 for pt in poly0]
                    by = [(pt[1] + k * pitch) * 1000.0 for pt in poly0]
                    ax.fill(bx, by, facecolor="#c8c8c8", edgecolor="#222", lw=0.6, zorder=6)
            except Exception:
                pass
        ax.set_aspect("equal")
        _crop_cascade_ax(ax, job)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y pitch [mm]")
        ax.set_title(f"2D streamlines  |U| color  t={latest} s  (OF U, slider mesh)")
        fig.colorbar(cf, ax=ax, label="|U| [m/s]")
        fig.tight_layout()
        fs = out_dir / "contour_stream.png"
        fig.savefig(fs)
        plt.close(fig)
        paths["contour_stream"] = str(fs)
        gamma = 1.33
        if job and isinstance(job.get("gas"), dict):
            try:
                gamma = float(job["gas"].get("gamma") or gamma)
            except (TypeError, ValueError):
                pass
        a_loc = None
        if rhovals and len(rhovals) == len(cc) and pvals and len(pvals) == len(cc):
            rho_a = np.maximum(np.array(rhovals), 1e-12)
            a_loc = np.sqrt(gamma * np.array(pvals) / rho_a)
        elif tvals and len(tvals) == len(cc) and job:
            rspec = float((job.get("gas") or {}).get("r_specific_j_kg_k") or 287.0)
            a_loc = np.sqrt(gamma * rspec * np.array(tvals))
        def _fill_blades(ax_):
            if not job:
                return
            try:
                from .geometry import center_in_pitch, profile_from_job, spec_from_job
                from .job import pitch_m as _pitch_m
                spec = spec_from_job(job)
                poly = profile_from_job(job, spec)
                pitch = _pitch_m(job)
                try:
                    poly0, _ = center_in_pitch(poly, pitch)
                except Exception:
                    poly0 = poly
                n_b = int(job["geometry"].get("n_blades_cascade") or 3)
                for k in range(n_b):
                    bx = [pt[0] * 1000.0 for pt in poly0]
                    by = [(pt[1] + k * pitch) * 1000.0 for pt in poly0]
                    ax_.fill(bx, by, facecolor="#c8c8c8", edgecolor="#222", lw=0.6, zorder=6)
            except Exception:
                return
        if a_loc is not None:
            mach = umag / np.maximum(a_loc, 1.0)
            Mi = np.ma.filled(mtri.LinearTriInterpolator(triang, mach)(X, Y), np.nan)
            fig, ax = plt.subplots(figsize=(8, 4.2), dpi=140)
            cf = ax.contourf(X, Y, np.ma.masked_invalid(Mi), levels=24, cmap="turbo")
            ax.streamplot(
                xi, yi,
                np.nan_to_num(Ui_s, nan=0.0), np.nan_to_num(Vi_s, nan=0.0),
                color="k", density=0.9, linewidth=0.5, arrowsize=0.7,
            )
            _fill_blades(ax)
            ax.set_aspect("equal")
            _crop_cascade_ax(ax, job)
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y pitch [mm]")
            ax.set_title(f"Mach  t={latest} s  ( |U|/a from OF p,rho )  PREDICTED")
            fig.colorbar(cf, ax=ax, label="Mach")
            fig.tight_layout()
            fm = out_dir / "contour_M.png"
            fig.savefig(fm)
            plt.close(fig)
            paths["contour_M"] = str(fm)
        Pi = np.ma.filled(mtri.LinearTriInterpolator(triang, p)(X, Y), np.nan)
        # yi is rows, xi is cols. np.gradient(f, *spacing) with f[row,col]
        dpy, dpx = np.gradient(np.nan_to_num(Pi, nan=0.0), yi, xi)
        sch = np.hypot(dpx, dpy)
        # Inlet/outlet patch lines dominate linear max. Log color + many hues
        # so weaker passage jumps still get a distinct band. Not KO.
        from matplotlib.colors import LogNorm
        dx_mm = 1.5
        dy_mm = 0.5
        interior = (
            (X > float(np.nanmin(X)) + dx_mm)
            & (X < float(np.nanmax(X)) - dx_mm)
            & (Y > float(np.nanmin(Y)) + dy_mm)
            & (Y < float(np.nanmax(Y)) - dy_mm)
        )
        finite = np.isfinite(sch) & interior & (sch > 0)
        pos = sch[finite]
        if pos.size > 50:
            vmin = float(np.nanpercentile(pos, 8))
            vmax = float(np.nanpercentile(pos, 92))
        else:
            pos2 = sch[np.isfinite(sch) & (sch > 0)]
            vmin = float(np.nanmin(pos2)) if pos2.size else 1.0
            vmax = float(np.nanmax(pos2)) if pos2.size else 10.0
        vmin = max(vmin, 1e-6)
        if vmax <= vmin:
            vmax = vmin * 10.0
        levels = np.logspace(np.log10(vmin), np.log10(vmax), 64)
        fig, ax = plt.subplots(figsize=(8, 4.2), dpi=140)
        cf = ax.contourf(
            X, Y, np.ma.masked_less_equal(np.ma.masked_invalid(sch), 0),
            levels=levels,
            cmap="turbo",
            norm=LogNorm(vmin=vmin, vmax=vmax),
            extend="both",
        )
        ax.contour(
            X, Y, np.nan_to_num(sch, nan=0.0),
            levels=np.logspace(np.log10(vmin), np.log10(vmax), 10),
            colors="k",
            linewidths=0.25,
            alpha=0.45,
        )
        _fill_blades(ax)
        ax.set_aspect("equal")
        _crop_cascade_ax(ax, job)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y pitch [mm]")
        ax.set_title(f"|grad p|  t={latest} s  (log turbo, 64 bands, not KO)  PREDICTED")
        fig.colorbar(cf, ax=ax, label="|grad p| [Pa/mm]  log")
        fig.tight_layout()
        fg = out_dir / "contour_shock.png"
        fig.savefig(fg)
        plt.close(fig)
        paths["contour_shock"] = str(fg)
        if tvals and len(tvals) == len(cc):
            Tv = np.array(tvals, dtype=float)
            Ti = np.ma.filled(mtri.LinearTriInterpolator(triang, Tv)(X, Y), np.nan)
            fig, ax = plt.subplots(figsize=(8, 4.2), dpi=140)
            cf = ax.contourf(X, Y, np.ma.masked_invalid(Ti), levels=24, cmap="inferno")
            _fill_blades(ax)
            ax.set_aspect("equal")
            _crop_cascade_ax(ax, job)
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y pitch [mm]")
            ax.set_title(f"T [K]  t={latest} s  (OF time-dir T)  PREDICTED")
            fig.colorbar(cf, ax=ax, label="T [K]")
            fig.tight_layout()
            ft = out_dir / "contour_T.png"
            fig.savefig(ft)
            plt.close(fig)
            paths["contour_T"] = str(ft)
    # PNG sequence from real time dirs only
    seq_dir = out_dir / "p_sequence"
    seq_dir.mkdir(exist_ok=True)
    frames = []
    for tname in times:
        pv = _parse_of_scalar(case_dir / tname / "p")
        if not pv or len(pv) != len(cc):
            continue
        fig, ax = plt.subplots(figsize=(7, 3.6), dpi=90)
        sc = ax.scatter(x, y, c=np.array(pv) / 1e5, s=5, cmap="coolwarm", linewidths=0, vmin=p.min()/1e5, vmax=p.max()/1e5)
        ax.set_aspect("equal")
        _crop_cascade_ax(ax, job)
        ax.set_title(f"p [bar] t={tname}")
        fig.colorbar(sc, ax=ax)
        fig.tight_layout()
        fn = seq_dir / f"p_{tname}.png"
        fig.savefig(fn)
        plt.close(fig)
        frames.append(fn)
    if frames:
        paths["p_sequence_dir"] = str(seq_dir)
        # mp4 if ffmpeg exists
        import shutil
        import subprocess

        if shutil.which("ffmpeg") and len(frames) >= 2:
            mp4 = out_dir / "p_field.mp4"
            # concat list
            lst = out_dir / "p_seq.txt"
            lst.write_text("".join(f"file '{f}'\n" for f in frames), encoding="utf-8")
            rc = subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-pix_fmt", "yuv420p", str(mp4)],
                capture_output=True,
            )
            if rc.returncode == 0 and mp4.is_file():
                paths["p_field_mp4"] = str(mp4)
    return paths


def estimate_yplus(
    case_dir: Path,
    *,
    first_cell_m: float,
    rho: float,
    mu: float,
    forces: dict[str, Any],
    z_thick_m: float,
) -> dict[str, Any]:
    """Crude y+ from viscous force / wall area and first-cell height."""
    notes = ["y+ is crude (first-cell height × uτ/ν from FO viscous). Laminar, no wall function."]
    yplus_file = list((case_dir / "postProcessing").glob("yPlus1/**/yPlus.dat")) if (case_dir / "postProcessing").is_dir() else []
    of_yplus = None
    if yplus_file:
        notes.append(f"yPlus FO wrote {yplus_file[-1]}")
    blades = forces.get("blades") or []
    out = []
    for b in blades:
        # τ ≈ |Fvisc| / A ; A unknown — use |Fvisc_N/m| / chord as order-of-magnitude
        fv = math.hypot(b.get("Ft_viscous_N", 0), b.get("Fd_viscous_N", 0))
        # N at span → N/m already; τ ~ F/A, A ~ chord * span, F_N/m ~ τ * chord
        chord = 0.01
        tau = abs(fv) / max(b.get("span_m", 0.005) * chord, 1e-16)
        utau = math.sqrt(max(tau, 0.0) / max(rho, 1e-16))
        nu = mu / max(rho, 1e-16)
        yp = first_cell_m * utau / max(nu, 1e-16)
        out.append({"blade": b["blade_index"], "tau_Pa_est": tau, "u_tau": utau, "y_plus_first_cell": yp})
    return {
        "predicted": True,
        "first_cell_m": first_cell_m,
        "blades": out,
        "notes": notes,
        "of_yplus_present": bool(yplus_file),
    }
