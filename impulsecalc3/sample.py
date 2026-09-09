"""Wall pressure sample: parse a p column, never T. Assert sample is on the metal."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any


def _split_tokens(line: str) -> list[str]:
    return line.replace("#", " ").replace(",", " ").split()


def parse_raw_surfaces(path: Path) -> list[dict[str, float]]:
    """Parse OpenFOAM surfaces raw: columns include x y z p (header names).

    Never assume column 1 is p. If the header has T before p, still pick p.
    """
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header: list[str] = []
    rows: list[dict[str, float]] = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            toks = _split_tokens(s)
            # keep the last comment that looks like a header
            names = [t.lower() for t in toks]
            if any(n in ("p", "x", "y", "z") for n in names):
                header = names
            continue
        parts = _split_tokens(s)
        try:
            nums = [float(p) for p in parts]
        except ValueError:
            continue
        rec: dict[str, float] = {}
        if header and len(header) >= 3:
            for name, val in zip(header, nums):
                rec[name] = val
            # some raw files: x y z p
            if "p" not in rec:
                # try last numeric as p only if header said so
                pass
        else:
            # fallback: x y z p  (still named, not T-as-p)
            if len(nums) >= 4:
                rec = {"x": nums[0], "y": nums[1], "z": nums[2], "p": nums[-1]}
            elif len(nums) >= 3:
                rec = {"x": nums[0], "y": nums[1], "z": nums[2]}
        if "p" in rec and "x" in rec:
            rows.append(rec)
    return rows


def _time_from_path(path: Path) -> float:
    """Numeric time = parent directory of the raw file (OF writeTime). Not lexicographic."""
    try:
        return float(path.parent.name)
    except ValueError:
        return 0.0


def _raw_hits_by_time(case_dir: Path, blade: int) -> list[tuple[float, Path]]:
    root = case_dir / "postProcessing"
    if not root.is_dir():
        return []
    hits = list(root.glob(f"surfaces_blade{blade}/**/*blade{blade}Wall*.raw"))
    if not hits:
        hits = list(root.glob(f"surfaces_blade{blade}/**/*.raw"))
    out = [(_time_from_path(p), p) for p in hits]
    out.sort(key=lambda kv: kv[0])
    return out


def find_wall_raw(case_dir: Path, blade: int) -> Path | None:
    hits = _raw_hits_by_time(case_dir, blade)
    return hits[-1][1] if hits else None


def find_t0_wall_raw(case_dir: Path, blade: int) -> Path | None:
    hits = _raw_hits_by_time(case_dir, blade)
    return hits[0][1] if hits else None


def _internal_p0(case_dir: Path) -> float | None:
    """Initialized 0/p (uniform IC). This is t=0, not the first writeTime dump."""
    p0 = case_dir / "0" / "p"
    if not p0.is_file():
        return None
    import re
    text = p0.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"internalField\s+uniform\s+([^\s;]+)", text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def split_ps_ss_by_y(
    rows: list[dict[str, float]],
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Split a blade-0-ish cloud into SS (higher y) and PS (lower y) via a mid-y cut
    along local x. Crude but on-wall.
    """
    if not rows:
        return [], []
    # Camber-ish midline: for each x-bin, median y; SS above, PS below.
    xs = [r["x"] for r in rows]
    xmin, xmax = min(xs), max(xs)
    nbin = 24
    bins: list[list[float]] = [[] for _ in range(nbin)]
    for r in rows:
        b = int((r["x"] - xmin) / max(xmax - xmin, 1e-16) * (nbin - 1e-9))
        b = min(max(b, 0), nbin - 1)
        bins[b].append(r["y"])
    mid = []
    for ys in bins:
        ys = sorted(ys)
        mid.append(ys[len(ys) // 2] if ys else 0.0)

    def mid_y(x: float) -> float:
        t = (x - xmin) / max(xmax - xmin, 1e-16)
        f = t * (nbin - 1)
        i = min(int(f), nbin - 2)
        a = f - i
        return mid[i] * (1 - a) + mid[i + 1] * a

    ps, ss = [], []
    for r in rows:
        if r["y"] >= mid_y(r["x"]):
            ss.append(r)
        else:
            ps.append(r)
    ps.sort(key=lambda r: r["x"])
    ss.sort(key=lambda r: r["x"])
    return ps, ss


def _s_coord(chain: list[dict[str, float]]) -> list[dict[str, float]]:
    s = 0.0
    out = []
    prev = None
    for r in chain:
        if prev is not None:
            s += math.hypot(r["x"] - prev["x"], r["y"] - prev["y"])
        rec = dict(r)
        rec["s_m"] = s
        out.append(rec)
        prev = r
    return out


def load_wall_pressure(
    case_dir: Path,
    *,
    p1_pa: float,
    chord_m: float,
    q_dyn_pa: float | None = None,
    blade_polys: list | None = None,
    first_cell_m: float | None = None,
) -> dict[str, Any]:
    notes: list[str] = []
    blades = []
    fail = None
    n_blades = len(blade_polys) if blade_polys else 3
    for k in range(n_blades):
        raw = find_wall_raw(case_dir, k)
        if raw is None:
            fail = f"no wall raw sample for blade{k}"
            break
        rows = parse_raw_surfaces(raw)
        if not rows or "p" not in rows[0]:
            fail = f"wall sample for blade{k} has no p column (refusing T-as-p)"
            break
        ps, ss = split_ps_ss_by_y(rows)
        ps, ss = _s_coord(ps), _s_coord(ss)
        for rec in ps + ss:
            rec["x_over_c"] = rec["x"] / max(chord_m, 1e-16)
            rec["Cp"] = (rec["p"] - p1_pa) / q_dyn_pa if (q_dyn_pa and q_dyn_pa > 0) else None
        mean_p = sum(r["p"] for r in rows) / len(rows)
        blades.append(
            {
                "blade": k,
                "n_points": len(rows),
                "mean_p": mean_p,
                "ps": ps,
                "ss": ss,
                "path": str(raw),
            }
        )
        notes.append(f"blade{k}: {len(rows)} wall points from {raw.name}, mean p={mean_p:.6g} Pa")

    if fail:
        return {
            "success": False,
            "on_wall": False,
            "error": fail,
            "notes": notes + ["FAIL LOUD: sample not on wall / no p column"],
        }

    # t=0+ assert: initialized 0/p, NOT the first writeTime surfaces dump.
    # First writeTime is ~0.6 chord; wall p there can be 1.5 p1 and still on the metal.
    t0_mean = _internal_p0(case_dir)
    t0_hits = _raw_hits_by_time(case_dir, 0)
    if t0_mean is None and t0_hits and t0_hits[0][0] < 1e-8:
        t0_rows = parse_raw_surfaces(t0_hits[0][1])
        t0_mean = sum(r["p"] for r in t0_rows) / len(t0_rows) if t0_rows else None
    on_wall = True
    reasons = []
    last_mean = blades[0]["mean_p"]
    if blade_polys:
        import math as _math
        def _dmin(x, y, poly):
            dmin = 1e99
            pts = poly[:-1] if poly and poly[0] == poly[-1] else poly
            for a, b in zip(pts, pts[1:] + pts[:1]):
                vx, vy = b[0] - a[0], b[1] - a[1]
                wx, wy = x - a[0], y - a[1]
                den = vx * vx + vy * vy
                tt = 0.0 if den < 1e-30 else max(0.0, min(1.0, (wx * vx + wy * vy) / den))
                dmin = min(dmin, _math.hypot(x - (a[0] + tt * vx), y - (a[1] + tt * vy)))
            return dmin
        tol = 4.0 * (first_cell_m or 1e-4)
        for b in blades:
            idx = int(b["blade"])
            if idx >= len(blade_polys):
                on_wall = False
                reasons.append(f"blade{idx} sample has no metal polygon (passage has {len(blade_polys)} walls)")
                continue
            poly = blade_polys[idx]
            rows = b["ps"] + b["ss"]
            mean_d = sum(_dmin(r["x"], r["y"], poly) for r in rows) / max(len(rows), 1)
            b["mean_distance_to_metal_m"] = mean_d
            if mean_d > tol:
                on_wall = False
                reasons.append(
                    f"blade{b['blade']} sample mean distance {mean_d:.3g} m > {tol:.3g} m (not on metal)"
                )
    if t0_mean is not None and abs(t0_mean - p1_pa) / p1_pa > 0.20:
        on_wall = False
        reasons.append(
            f"t=0+ mean wall p={t0_mean:.4g} is not within 20% of p1={p1_pa:.4g}"
        )
    if last_mean < 0.2 * p1_pa or last_mean > 5 * p1_pa:
        on_wall = False
        reasons.append(f"wall p={last_mean:.4g} is not O(p1={p1_pa:.4g})")
    # classic fail: O(1 kPa) vs 5.5 bar
    if last_mean < 2e4 and p1_pa > 1e5:
        on_wall = False
        reasons.append(f"wall p is O({last_mean:.3g} Pa) while p1 is {p1_pa:.3g} Pa — not on the metal")

    return {
        "success": on_wall,
        "on_wall": on_wall,
        "t0_mean_p": t0_mean,
        "p1_pa": p1_pa,
        "blades": blades,
        "error": None if on_wall else "; ".join(reasons),
        "notes": notes + reasons,
        "parser": "header-aware p column, never T",
    }
