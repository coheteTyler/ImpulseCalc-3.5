"""2D cascade metal: circular-arc foil, dual-arc impulse bucket, or points.

Live 8766 / knobs metal: `impulse_bucket` — two circular arcs (outer/inner sagittas)
plus circular LE/TE fillets. Smooth round ends. Closed CCW, no self-intersect.
TN D-4421 MOC lives in goldman.py for later; this module does not call it.

Marlin JSON (no family): circular-arc camber foil. `profile_points` is a closed
(x,y) list. Open / self-intersecting / zero-area fail loud.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

R_GAS = 8.314462618  # J/mol/K — unused here, documented


def _closed(poly: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pts = [(float(x), float(y)) for x, y in poly]
    if len(pts) < 3:
        return pts
    if abs(pts[0][0] - pts[-1][0]) > 1e-14 or abs(pts[0][1] - pts[-1][1]) > 1e-14:
        pts.append(pts[0])
    return pts


def polygon_signed_area(poly: list[tuple[float, float]]) -> float:
    pts = poly[:-1] if poly and poly[0] == poly[-1] else poly
    a = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return 0.5 * a


def polygon_centroid(poly: list[tuple[float, float]]) -> tuple[float, float]:
    pts = poly[:-1] if poly and poly[0] == poly[-1] else list(poly)
    a = polygon_signed_area(pts)
    if abs(a) < 1e-16:
        return (
            sum(p[0] for p in pts) / max(len(pts), 1),
            sum(p[1] for p in pts) / max(len(pts), 1),
        )
    cx = cy = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        cross = x1 * y2 - x2 * y1
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    return (cx / (6.0 * a), cy / (6.0 * a))


@dataclass
class BladeSpec:
    chord_m: float = 0.01
    beta1_metal_deg: float = 72.0
    beta2_metal_deg: float = -72.0
    thickness_c: float = 0.12
    le_radius_c: float = 0.03
    te_radius_c: float = 0.012
    n_points: int = 160

    @property
    def stagger_deg(self) -> float:
        return 0.5 * (self.beta1_metal_deg + self.beta2_metal_deg)

    @property
    def turning_deg(self) -> float:
        return self.beta1_metal_deg - self.beta2_metal_deg

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["stagger_deg"] = self.stagger_deg
        d["turning_deg"] = self.turning_deg
        d["family"] = "circular_arc_camber_metal_angles"
        return d


def circular_arc_camber(
    spec: BladeSpec,
    n: int | None = None,
) -> list[tuple[float, float]]:
    """Camber line in cascade (x=axial, y=tangential).

    Built in the chord frame then rotated by stagger so the chord sits at
    γ = ½(β1*+β2*) from axial. Tangent at LE is β1*, at TE is β2*.
    """
    n = int(n or max(spec.n_points // 2, 40))
    c = float(spec.chord_m)
    b1 = math.radians(spec.beta1_metal_deg)
    b2 = math.radians(spec.beta2_metal_deg)
    stagger = 0.5 * (b1 + b2)
    turning = b1 - b2
    if abs(turning) < 1e-6:
        # Flat plate at stagger
        pts = []
        for i in range(n):
            t = i / (n - 1)
            xi = t * c
            pts.append((xi * math.cos(stagger), xi * math.sin(stagger)))
        return pts

    # Chord-frame: LE at (0,0), TE at (c,0). Relative metal angles ±turning/2.
    half = 0.5 * turning
    R = c / (2.0 * math.sin(abs(half)))
    # Camber bulges to +η when LE relative angle is positive (β1* > stagger).
    sign = 1.0 if half > 0 else -1.0
    cx_c, cy_c = 0.5 * c, -sign * R * math.cos(abs(half))

    # Radius angle at LE (centre → LE). Sweep of −turning (radius opposite the
    # tangent) lands on TE at (c, 0) with included angle |turning|.
    a_le = math.atan2(0.0 - cy_c, 0.0 - cx_c)
    sweep = -turning
    pts_c = []
    for i in range(n):
        t = i / (n - 1)
        ang = a_le + t * sweep
        pts_c.append((cx_c + R * math.cos(ang), cy_c + R * math.sin(ang)))

    # Rotate by stagger into cascade axes; LE stays at origin.
    cg, sg = math.cos(stagger), math.sin(stagger)
    out = []
    for x, y in pts_c:
        out.append((x * cg - y * sg, x * sg + y * cg))
    return out


def _naca_half_thickness(xoc: float, t_c: float) -> float:
    """Classic four-digit half-thickness (closed TE modified). xoc in [0,1]."""
    x = min(max(xoc, 0.0), 1.0)
    # -0.1015 x^4 closes with a sharp TE; we keep a small finite TE.
    yt = (
        5.0
        * t_c
        * (
            0.2969 * math.sqrt(x)
            - 0.1260 * x
            - 0.3516 * x * x
            + 0.2843 * x * x * x
            - 0.1036 * x * x * x * x
        )
    )
    return max(yt, 0.0)


def closed_profile(spec: BladeSpec) -> list[tuple[float, float]]:
    """Closed CCW polygon (fluid outside, metal inside). Metres."""
    n_cam = max(spec.n_points, 80)
    cam = circular_arc_camber(spec, n=n_cam)
    c = spec.chord_m
    t_c = float(spec.thickness_c)
    r_le = float(spec.le_radius_c) * c
    r_te = float(spec.te_radius_c) * c

    # Tangents / normals along camber
    def _tan_norm(i: int) -> tuple[float, float, float, float]:
        if i == 0:
            dx = cam[1][0] - cam[0][0]
            dy = cam[1][1] - cam[0][1]
        elif i == len(cam) - 1:
            dx = cam[-1][0] - cam[-2][0]
            dy = cam[-1][1] - cam[-2][1]
        else:
            dx = cam[i + 1][0] - cam[i - 1][0]
            dy = cam[i + 1][1] - cam[i - 1][1]
        L = math.hypot(dx, dy) or 1e-16
        tx, ty = dx / L, dy / L
        # Left normal (toward +η if travelling LE→TE along a +η camber)
        return tx, ty, -ty, tx

    ss: list[tuple[float, float]] = []
    ps: list[tuple[float, float]] = []
    for i, (x, y) in enumerate(cam):
        xoc = i / (len(cam) - 1)
        h = _naca_half_thickness(xoc, t_c) * c
        # Blend to prescribed LE/TE radii near the ends (NACA LE radius ≈ 1.1019 t^2 c)
        if xoc < 0.08:
            # NACA already sets LE radius from t; cap to r_le
            h = min(h, math.sqrt(max(2.0 * r_le * xoc * c, 0.0)) if False else h)
        tx, ty, nx, ny = _tan_norm(i)
        ss.append((x + nx * h, y + ny * h))
        ps.append((x - nx * h, y - ny * h))

    # Circular LE cap: replace the first few SS/PS points with an arc around LE.
    tx0, ty0, nx0, ny0 = _tan_norm(0)
    # LE centre sits along -tangent (into the metal) from the LE camber point.
    le = cam[0]
    # Camber LE is on the surface for a sharp plate; for finite r, centre is
    # r along +camber from the geometric LE so the cap is in front.
    # Put centre at camber point + r_le * tangent (into the chord).
    le_c = (le[0] + tx0 * r_le, le[1] + ty0 * r_le)
    # Arc from PS side (-n) to SS side (+n) going around the nose (against tangent).
    n_cap = max(10, spec.n_points // 16)
    le_arc: list[tuple[float, float]] = []
    # Angle of -normal (PS) and +normal (SS) at LE relative to centre.
    a_ps = math.atan2(-ny0, -nx0)
    a_ss = math.atan2(ny0, nx0)
    # Go the long way around the nose: from PS through -tangent to SS.
    # From a_ps to a_ss CCW passing through angle of -tangent.
    a_nose = math.atan2(-ty0, -tx0)

    def _sweep(a0: float, a1: float, via: float, nseg: int) -> list[float]:
        # Parameterize a0 → via → a1, each the short arc, concatenated.
        def short(a, b):
            d = (b - a + math.pi) % (2 * math.pi) - math.pi
            return d

        d1 = short(a0, via)
        d2 = short(via, a1)
        out = []
        n1 = max(nseg // 2, 3)
        n2 = max(nseg - n1, 3)
        for k in range(n1):
            out.append(a0 + d1 * k / n1)
        for k in range(n2 + 1):
            out.append(via + d2 * k / n2)
        return out

    angs = _sweep(a_ps, a_ss, a_nose, n_cap)
    le_arc = [(le_c[0] + r_le * math.cos(a), le_c[1] + r_le * math.sin(a)) for a in angs]

    # Circular TE cap around TE centre.
    tx1, ty1, nx1, ny1 = _tan_norm(len(cam) - 1)
    te = cam[-1]
    te_c = (te[0] - tx1 * r_te, te[1] - ty1 * r_te)
    a_ss_te = math.atan2(ny1, nx1)
    a_ps_te = math.atan2(-ny1, -nx1)
    a_tail = math.atan2(ty1, tx1)
    n_te = max(8, spec.n_points // 20)
    angs_te = _sweep(a_ss_te, a_ps_te, a_tail, n_te)
    te_arc = [(te_c[0] + r_te * math.cos(a), te_c[1] + r_te * math.sin(a)) for a in angs_te]

    # Drop camber-offset points that fall inside the LE/TE caps.
    def _outside_cap(p, centre, r, keep_away_from_le: bool) -> bool:
        d = math.hypot(p[0] - centre[0], p[1] - centre[1])
        return d >= r * 0.92

    ss_mid = [p for p in ss[3:-3] if _outside_cap(p, le_c, r_le, True) and _outside_cap(p, te_c, r_te, False)]
    ps_mid = [p for p in ps[3:-3] if _outside_cap(p, le_c, r_le, True) and _outside_cap(p, te_c, r_te, False)]

    # Walk CCW: LE arc (PS→nose→SS), SS to TE, TE arc (SS→tail→PS), PS back (reversed).
    poly: list[tuple[float, float]] = []
    poly.extend(le_arc)
    poly.extend(ss_mid)
    poly.extend(te_arc)
    poly.extend(reversed(ps_mid))
    poly = _closed(poly)

    # Ensure CCW (positive area = metal interior).
    if polygon_signed_area(poly) < 0:
        poly = list(reversed(poly))
        poly = _closed(poly)
    return poly


def clip_poly_to_y_strip(
    poly: list[tuple[float, float]],
    y_lo: float,
    y_hi: float,
) -> list[tuple[float, float]]:
    """Sutherland–Hodgman clip of a closed loop to y ∈ [y_lo, y_hi]. Strip is convex."""
    pts = list(poly)
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) < 3:
        return _closed(pts)

    def _clip(seq: list[tuple[float, float]], ylim: float, keep_le: bool) -> list[tuple[float, float]]:
        if not seq:
            return []
        def inside(p: tuple[float, float]) -> bool:
            return p[1] <= ylim + 1e-16 if keep_le else p[1] >= ylim - 1e-16
        def hit(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
            dy = b[1] - a[1]
            if abs(dy) < 1e-18:
                return (b[0], ylim)
            tt = (ylim - a[1]) / dy
            return (a[0] + tt * (b[0] - a[0]), ylim)
        out: list[tuple[float, float]] = []
        prev = seq[-1]
        for cur in seq:
            pin, cin = inside(prev), inside(cur)
            if cin:
                if not pin:
                    out.append(hit(prev, cur))
                out.append(cur)
            elif pin:
                out.append(hit(prev, cur))
            prev = cur
        return out

    pts = _clip(pts, y_hi, True)
    pts = _clip(pts, y_lo, False)
    return _closed(pts)


def center_in_pitch(
    poly: list[tuple[float, float]],
    pitch_m: float,
) -> tuple[list[tuple[float, float]], float]:
    """Translate profile in y so it sits mid-pitch.

    Do not shear the C. yspan > s is legal nesting when solids do not intersect
    (passage_gap / SS vs PS+s). Mesh uses cassette or H-O-H — do not raise here.
    """
    ys = [p[1] for p in poly]
    y_mid = 0.5 * (min(ys) + max(ys))
    shift = -y_mid
    out = [(p[0], p[1] + shift) for p in poly]
    return out, shift


PACK_Y_FIT = 0.94  # leftover; do not auto-open s when packing_driver is pitch


def fit_pitch_to_metal(job: dict[str, Any], poly: list[tuple[float, float]]) -> list[str]:
    """Report y-span vs s. Do not raise the user's spacing knob. Never flatten stems."""
    if not poly:
        return []
    from .job import pitch_m as _pitch_m
    g = job.setdefault("geometry", {})
    ys = [pt[1] for pt in poly]
    yspan = max(ys) - min(ys)
    p0 = float(_pitch_m(job))
    drv = str(g.get("packing_driver") or "")
    gap = p0 - yspan
    notes = []
    if gap < -1e-9:
        cut = 0.5 * (-gap)
        notes.append(
            f"s={p0*1e3:.2f} mm < y-span {yspan*1e3:.2f} mm: nested pack (legal if g_min>0). "
            "Mesh uses cassette / H-O-H, not a forced s-open. Outer C kept. Not σ. Not a clip."
        )
        return notes
    if gap < 4e-5:
        notes.append(
            f"tight pack: s={p0*1e3:.2f} mm, y-span={yspan*1e3:.2f} mm, gap={gap*1e3:.3f} mm. "
            "O-grid first cell shrinks. Unsigned."
        )
    return notes


def camber_end_tangents_deg(spec: BladeSpec) -> tuple[float, float]:
    """Finite-difference camber tangents at LE and TE, cascade axes, degrees."""
    cam = circular_arc_camber(spec, n=80)
    def ang(i0, i1):
        dx = cam[i1][0] - cam[i0][0]
        dy = cam[i1][1] - cam[i0][1]
        return math.degrees(math.atan2(dy, dx))
    return ang(0, 1), ang(-2, -1)


def split_ps_ss(
    poly: list[tuple[float, float]],
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], int, int]:
    """Split closed profile into PS and SS chains.

    LE = min-x vertex, TE = max-x vertex.
    Walking CCW from LE: first chain is PS (typically), second is SS — we
    label by mean y (SS = higher mean y for this camber).
    Returns (ps, ss, i_le, i_te) without duplicating the wrap point.
    """
    pts = poly[:-1] if poly and poly[0] == poly[-1] else list(poly)
    xs = [p[0] for p in pts]
    i_le = min(range(len(pts)), key=lambda i: xs[i])
    i_te = max(range(len(pts)), key=lambda i: xs[i])
    n = len(pts)

    def chain(i0: int, i1: int) -> list[tuple[float, float]]:
        out = []
        i = i0
        while True:
            out.append(pts[i])
            if i == i1:
                break
            i = (i + 1) % n
            if len(out) > n + 2:
                break
        return out

    c1 = chain(i_le, i_te)
    c2 = chain(i_te, i_le)
    m1 = sum(p[1] for p in c1) / max(len(c1), 1)
    m2 = sum(p[1] for p in c2) / max(len(c2), 1)
    if m1 > m2:
        ss, ps = c1, list(reversed(c2))
    else:
        ps, ss = c1, list(reversed(c2))
    # ps, ss both run LE → TE
    return ps, ss, i_le, i_te


def surface_coordinate(
    chain: list[tuple[float, float]],
) -> list[float]:
    """Arc-length s from the first point along a chain."""
    s = [0.0]
    for i in range(1, len(chain)):
        ds = math.hypot(chain[i][0] - chain[i - 1][0], chain[i][1] - chain[i - 1][1])
        s.append(s[-1] + ds)
    return s


def resample_open_arclength(
    chain: list[tuple[float, float]] | np.ndarray,
    n: int,
) -> np.ndarray:
    """n points along an open chain, even arc-length (never y(x))."""
    pts = np.asarray(chain, dtype=float).reshape(-1, 2)
    n = max(int(n), 2)
    if len(pts) < 2:
        return np.repeat(pts[:1], n, axis=0) if len(pts) else np.zeros((n, 2))
    s = surface_coordinate([(float(x), float(y)) for x, y in pts])
    total = s[-1] if s[-1] > 0 else 1.0
    out = np.zeros((n, 2), dtype=float)
    for k in range(n):
        target = total * k / (n - 1)
        if target <= 0:
            out[k] = pts[0]
            continue
        if target >= total:
            out[k] = pts[-1]
            continue
        j = 0
        while j < len(s) - 1 and s[j + 1] < target:
            j += 1
        span = s[j + 1] - s[j] or 1e-16
        tt = (target - s[j]) / span
        out[k] = (1.0 - tt) * pts[j] + tt * pts[j + 1]
    return out


def passage_gap(poly: list[tuple[float, float]], pitch_m: float) -> dict:
    """Wall-normal gap between SS of blade 0 and PS+(0,pitch). Arc-length pairing.

    g_i = n_hat · (PS1[i] - SS0[i]) with n_hat the SS normal toward PS (not Δy).
    g_min <= 0 means intersecting metal.
    """
    ps, ss, i_le, i_te = split_ps_ss(poly)
    n = max(len(ps), len(ss), 64)
    ss0 = resample_open_arclength(ss, n)
    ps0 = resample_open_arclength(ps, n)
    ps1 = ps0.copy()
    ps1[:, 1] = ps1[:, 1] + float(pitch_m)
    g = np.zeros(n, dtype=float)
    n_hat = np.zeros((n, 2), dtype=float)
    for i in range(n):
        i0 = max(i - 1, 0)
        i1 = min(i + 1, n - 1)
        e = ss0[i1] - ss0[i0]
        L = float(np.hypot(e[0], e[1])) or 1e-16
        nrm = np.array([e[1] / L, -e[0] / L], dtype=float)
        to_ps = ps1[i] - ss0[i]
        if float(nrm[0] * to_ps[0] + nrm[1] * to_ps[1]) < 0.0:
            nrm = -nrm
        n_hat[i] = nrm
        g[i] = float(nrm[0] * to_ps[0] + nrm[1] * to_ps[1])
    g_min = float(g.min()) if len(g) else 0.0
    return {
        "g_min": g_min,
        "g": g,
        "ss": ss0,
        "ps1": ps1,
        "n_hat": n_hat,
        "i_le": i_le,
        "i_te": i_te,
        "intersecting": g_min <= 0.0,
    }


def _cross(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_properly_intersect(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:
    """True if open segments ab and cd cross. Shared vertices do not count."""
    d1 = _cross(a, b, c)
    d2 = _cross(a, b, d)
    d3 = _cross(c, d, a)
    d4 = _cross(c, d, b)
    return (d1 > 0 and d2 < 0 or d1 < 0 and d2 > 0) and (
        d3 > 0 and d4 < 0 or d3 < 0 and d4 > 0
    )


def polygon_self_intersects(poly: list[tuple[float, float]]) -> bool:
    """Simple-polygon test. Adjacent and wrap-adjacent edges are allowed to share a vertex."""
    pts = poly[:-1] if poly and poly[0] == poly[-1] else list(poly)
    n = len(pts)
    if n < 4:
        return False
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or (i + 1) % n == j:
                continue
            c, d = pts[j], pts[(j + 1) % n]
            if a == c or a == d or b == c or b == d:
                continue
            if _segments_properly_intersect(a, b, c, d):
                return True
    return False


def profile_from_points(
    points: list,
    *,
    name: str = "profile_points",
) -> list[tuple[float, float]]:
    """Closed metal polygon from an assignment list of (x, y). Fail loud.

    Not circular-arc camber. First point must equal last (closed). Self-intersecting
    or zero-area loops raise. CW loops are reversed to CCW (metal interior).
    """
    if points is None:
        raise ValueError(f"{name} is missing")
    try:
        n_in = len(points)
    except TypeError as exc:
        raise ValueError(f"{name} must be a list of (x,y) pairs") from exc
    if n_in < 4:
        raise ValueError(f"{name}: need a closed loop of >= 4 vertices, got {n_in}")
    pts: list[tuple[float, float]] = []
    for i, p in enumerate(points):
        try:
            x, y = float(p[0]), float(p[1])
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError(f"{name}[{i}] is not an (x,y) pair") from exc
        pts.append((x, y))
    if abs(pts[0][0] - pts[-1][0]) > 1e-12 or abs(pts[0][1] - pts[-1][1]) > 1e-12:
        raise ValueError(
            f"{name} is an open polyline (first {pts[0]} != last {pts[-1]}); "
            "repeat the first point to close the blade"
        )
    # Force exact close.
    pts[-1] = pts[0]
    body = pts[:-1]
    # Drop consecutive duplicates (except the wrap we just set).
    cleaned: list[tuple[float, float]] = [body[0]]
    for q in body[1:]:
        if abs(q[0] - cleaned[-1][0]) > 1e-16 or abs(q[1] - cleaned[-1][1]) > 1e-16:
            cleaned.append(q)
    if len(cleaned) < 3:
        raise ValueError(f"{name} has fewer than 3 distinct vertices")
    pts = cleaned + [cleaned[0]]
    if polygon_self_intersects(pts):
        raise ValueError(f"{name} is self-intersecting")
    area = polygon_signed_area(pts)
    if abs(area) < 1e-16:
        raise ValueError(f"{name} has zero area ({area})")
    if area < 0:
        pts = list(reversed(pts))
        pts[-1] = pts[0]
    return pts


def _gget(g: dict[str, Any], *names: str, default: Any = None) -> Any:
    """First present geometry key (canonical or v2 alias)."""
    for n in names:
        if n in g and g[n] not in (None, ""):
            return g[n]
    return default



def _abs_len_m(g: dict[str, Any], *keys: str) -> float | None:
    """First present length. Keys ending _mm are millimetres; others metres."""
    for k in keys:
        if k in g and g[k] not in (None, ""):
            v = float(g[k])
            if k.endswith("_mm"):
                return v * 1e-3
            return v
    return None


def spec_from_job(job: dict[str, Any]) -> BladeSpec:
    """BladeSpec from job JSON. Metal β* = flow β − i / + δ. Used when no profile_points."""
    g = job["geometry"]
    inc = float(_gget(g, "incidence_deg", default=0.0) or 0.0)
    dev = float(_gget(g, "deviation_deg", default=0.0) or 0.0)
    b1 = float(_gget(g, "beta1_flow_deg", "beta1", "beta1_deg"))
    b2 = float(_gget(g, "beta2_flow_deg", "beta2", "beta2_deg"))
    chord = float(_gget(g, "chord_m", "chord", "c"))
    return BladeSpec(
        chord_m=chord,
        beta1_metal_deg=b1 - inc,
        beta2_metal_deg=b2 + dev,
        thickness_c=float(_gget(g, "thickness_c", "t_c", "tc", default=0.12)),
        le_radius_c=float(_gget(g, "le_radius_c", "le_fillet_r_c", "le", default=0.03)),
        te_radius_c=float(_gget(g, "te_radius_c", "te_fillet_r_c", "te", "te_fillet", default=0.012)),
        n_points=int(_gget(g, "n_profile_points", default=160)),
    )


def _rotate_poly(poly: list[tuple[float, float]], deg: float) -> list[tuple[float, float]]:
    if abs(deg) < 1e-12 or len(poly) < 2:
        return poly
    a = math.radians(deg)
    cg, sg = math.cos(a), math.sin(a)
    ox, oy = poly[0]
    out = []
    for x, y in poly:
        dx, dy = x - ox, y - oy
        out.append((ox + dx * cg - dy * sg, oy + dx * sg + dy * cg))
    return out


def profile_from_job(
    job: dict[str, Any],
    spec: BladeSpec | None = None,
) -> list[tuple[float, float]]:
    """Metal polygon: profile_points, else dual-arc bucket, else circular-arc foil.

    Knobs write this polygon. v2 aliases (upper_h, lower_h, t_c, Z, sigma, …) work.
    """
    g = job["geometry"]
    raw = _gget(g, "profile_points", "points")
    if raw:
        return profile_from_points(raw)
    fam = str(_gget(g, "profile_family", "family", "profile", default="") or "").lower()
    if fam in (
        "impulse_bucket", "dual_arc", "pelton", "bucket", "cup",
        "goldman", "goldman_vortex", "vortex_impulse",
    ):
        chord = float(_gget(g, "chord_m", "chord", "c"))
        try:
            from .job import pitch_m as _pitch_m
            pitch = _pitch_m(job)
        except Exception:
            sigma = float(_gget(g, "solidity", "sigma", default=1.0) or 1.0)
            pitch = chord / max(sigma, 1e-12)
        inc = float(_gget(g, "incidence_deg", default=0.0) or 0.0)
        dev = float(_gget(g, "deviation_deg", default=0.0) or 0.0)
        b1 = float(_gget(g, "beta1_flow_deg", "beta1", default=72.0) or 72.0)
        b2 = float(_gget(g, "beta2_flow_deg", "beta2", default=-72.0) or -72.0)
        b1m = _gget(g, "beta1_metal_deg")
        b2m = _gget(g, "beta2_metal_deg")
        beta1_metal = (float(b1m) if b1m not in (None, "") else (b1 - inc))
        beta2_metal = (float(b2m) if b2m not in (None, "") else (b2 + dev))
        poly = live_bucket_profile(
            chord_m=chord,
            beta1_metal_deg=beta1_metal,
            beta2_metal_deg=beta2_metal,
            upper_sagitta_c=float(_gget(g, "upper_sagitta_c", "upper_h", "hu_c", default=0.50)),
            lower_sagitta_c=float(_gget(g, "lower_sagitta_c", "lower_h", "hl_c", default=0.22)),
            le_fillet_r_c=float(_gget(g, "le_fillet_r_c", "le_radius_c", "le", default=0.04)),
            te_fillet_r_c=float(_gget(g, "te_fillet_r_c", "te_radius_c", "te", "te_fillet", default=0.04)),
            upper_sagitta_m=_abs_len_m(g, "upper_sagitta_m", "hu_mm", "upper_sagitta_mm"),
            lower_sagitta_m=_abs_len_m(g, "lower_sagitta_m", "hl_mm", "lower_sagitta_mm"),
            le_fillet_r_m=_abs_len_m(g, "le_fillet_r_m", "le_mm", "le_fillet_mm"),
            te_fillet_r_m=_abs_len_m(g, "te_fillet_r_m", "te_mm", "te_fillet_mm"),
            lin_m=_abs_len_m(g, "lin_m", "lin_mm"),
            lout_m=_abs_len_m(g, "lout_m", "lout_mm"),
            r_tr_m=_abs_len_m(g, "r_tr_m", "r_tr_mm"),
            r_main_m=_abs_len_m(g, "r_main_m", "r_main_mm"),
            t_m=_abs_len_m(g, "t_m", "t_mm"),
            psi_tr_deg=(None if _gget(g, "psi_tr_deg") in (None, "") else float(_gget(g, "psi_tr_deg"))),
            n_points=int(_gget(g, "n_profile_points", default=160)),
            pitch_m=pitch,
        )
        st = _gget(g, "stagger_deg", "stagger")
        if st in (None, "", "auto"):
            st = 0.5 * (beta1_metal + beta2_metal)
        poly = _rotate_poly(poly, float(st))
        return poly
    if spec is None:
        spec = spec_from_job(job)
        st = _gget(g, "stagger_deg", "stagger")
        if st not in (None, "", "auto"):
            # Free stagger override: rebuild camber at this chord angle via temporary betas
            # is wrong; closed_profile uses ½(β1+β2). Leave metal β* as the foil knobs.
            pass
    return closed_profile(spec)


# Pointed writer self-intersects above this hu when L_in or L_out > 0 (this C, t=1.4 mm).
HU_STEM_MAX_M = 0.0050


def metal_geom_bounds(job: dict[str, Any]) -> dict[str, dict[str, float | str]]:
    """Slider min/max for live metal. Not a signed cycle box."""
    g = job.get("geometry") or {}
    c = max(float(g.get("chord_m") or 0.01), 1e-6)
    lin = float(g.get("lin_m") or 0.0)
    lout = float(g.get("lout_m") or 0.0)
    hu = float(g.get("upper_sagitta_m") or 0.005)
    stems = lin > 1e-9 or lout > 1e-9
    hu_max = min(0.012, max(0.008, 1.4 * c))
    why_hu = "dual-arc C"
    if stems:
        hu_max = min(hu_max, HU_STEM_MAX_M)
        why_hu = "hu > 5.0 mm with stems self-intersects; drop L_in/L_out to raise hu"
    hl_max = min(0.010, 0.84 * max(hu, 0.001))
    t_max = min(0.004, 0.85 * max(hu - float(g.get("lower_sagitta_m") or 0.002), 0.0005))
    gap = max(hu - float(g.get("lower_sagitta_m") or 0.002), 0.0005)
    hl_min = 2.15 if stems else 0.4
    t_min = 1.4 if stems else 0.3
    return {
        "hu_mm": {"min": 1.0, "max": round(hu_max * 1e3, 2), "why": why_hu},
        "hl_mm": {"min": hl_min, "max": round(max(hl_min + 0.2, hl_max * 1e3), 2), "why": "pointed writer" if stems else "hl < 0.84 hu"},
        "t_mm": {"min": t_min, "max": round(max(t_min + 0.1, t_max * 1e3), 2), "why": "pointed writer t≥1.4 mm" if stems else "t < 0.85 (hu−hl)"},
        "le_mm": {"min": 0.0, "max": round(min(2.0, 0.45 * gap * 1e3), 2), "why": "fillet < 0.45 gap"},
        "te_mm": {"min": 0.0, "max": round(min(2.0, 0.45 * gap * 1e3), 2), "why": "fillet < 0.45 gap"},
        "lin_mm": {"min": 0.0, "max": 12.0, "why": "stem length"},
        "lout_mm": {"min": 0.0, "max": 12.0, "why": "stem length"},
        "r_main_mm": {"min": 1.0, "max": 6.0, "why": "inner trans self-intersects above 6 mm on this C"},
        "r_tr_mm": {"min": 0.5, "max": 8.0, "why": "outer blend"},
        "psi_tr_deg": {"min": 0.0, "max": 40.0, "why": "extra splay off outer tangent"},
    }


def apply_metal_bounds(job: dict[str, Any]) -> tuple[dict[str, Any], list[str], dict[str, dict[str, float | str]]]:
    """Clip metal knobs to live bounds. Returns (job, warnings, bounds)."""
    bounds = metal_geom_bounds(job)
    g = dict(job.get("geometry") or {})
    notes: list[str] = []

    def clip_m(key_m: str, bkey: str) -> None:
        if g.get(key_m) in (None, ""):
            return
        b = bounds[bkey]
        v = float(g[key_m])
        lo, hi = float(b["min"]) * 1e-3, float(b["max"]) * 1e-3
        if bkey.endswith("deg"):
            lo, hi = float(b["min"]), float(b["max"])
            if v < lo or v > hi:
                notes.append(f"{bkey} clipped to [{b['min']}, {b['max']}]")
                g[key_m] = min(max(v, lo), hi)
            return
        if v < lo - 1e-12 or v > hi + 1e-12:
            notes.append(f"{bkey} {v*1e3:.2f} → {min(max(v, lo), hi)*1e3:.2f} mm ({b['why']})")
            g[key_m] = min(max(v, lo), hi)

    clip_m("upper_sagitta_m", "hu_mm")
    clip_m("lower_sagitta_m", "hl_mm")
    # recompute after hu/hl
    bounds = metal_geom_bounds({**job, "geometry": g})
    clip_m("t_m", "t_mm")
    clip_m("le_fillet_r_m", "le_mm")
    clip_m("te_fillet_r_m", "te_mm")
    clip_m("lin_m", "lin_mm")
    clip_m("lout_m", "lout_mm")
    clip_m("r_main_m", "r_main_mm")
    clip_m("r_tr_m", "r_tr_mm")
    if g.get("psi_tr_deg") not in (None, ""):
        clip_m("psi_tr_deg", "psi_tr_deg")
    hu = float(g.get("upper_sagitta_m") or 0.005)
    hl = float(g.get("lower_sagitta_m") or 0.002)
    if hl >= hu - 5e-4:
        g["lower_sagitta_m"] = 0.45 * hu
        notes.append("hl was ≥ hu; set hl = 0.45 hu")
    out = dict(job)
    out["geometry"] = g
    return out, notes, metal_geom_bounds(out)


def safe_profile_from_job(
    job: dict[str, Any],
    spec: BladeSpec | None = None,
) -> tuple[list[tuple[float, float]], list[str]]:
    """Never raise for live metal. Stems dropped if the pointed writer intersects."""
    notes: list[str] = []
    try:
        return profile_from_job(job, spec), notes
    except (RuntimeError, ValueError) as exc:
        notes.append(str(exc))
        g = dict(job.get("geometry") or {})
        if float(g.get("lin_m") or 0) > 1e-12 or float(g.get("lout_m") or 0) > 1e-12:
            g["lin_m"] = 0.0
            g["lout_m"] = 0.0
            notes.append("L_in=L_out → 0 so the C still draws")
            job2 = dict(job)
            job2["geometry"] = g
            return profile_from_job(job2, spec), notes
        raise


def circular_arc_le_te(c: float, h: float, n: int) -> list[tuple[float, float]]:
    """Circular arc LE(0,0) → TE(c,0) with sagitta h in +y. Endpoints exact."""
    c = max(float(c), 1e-9)
    h = max(float(h), 1e-9)
    n = max(int(n), 8)
    R = (0.25 * c * c + h * h) / (2.0 * h)
    cx, cy = 0.5 * c, h - R
    a0 = math.atan2(0.0 - cy, 0.0 - cx)
    a1 = math.atan2(0.0 - cy, c - cx)
    best = None
    best_y = -1e99
    for flip in (0, 1):
        da = a1 - a0
        while da <= -math.pi:
            da += 2 * math.pi
        while da > math.pi:
            da -= 2 * math.pi
        if flip:
            da = da - 2 * math.pi if da > 0 else da + 2 * math.pi
        angs = [a0 + da * i / (n - 1) for i in range(n)]
        my = cy + R * math.sin(angs[n // 2])
        if my > best_y:
            best_y = my
            best = angs
    pts = [(cx + R * math.cos(a), cy + R * math.sin(a)) for a in best or [a0]]
    pts[0] = (0.0, 0.0)
    pts[-1] = (c, 0.0)
    return pts


def _arc_circle(c: float, h: float) -> tuple[float, float, float]:
    c = max(float(c), 1e-9)
    h = max(float(h), 1e-9)
    R = (0.25 * c * c + h * h) / (2.0 * h)
    return 0.5 * c, h - R, R


def _circle_hits(
    c0: tuple[float, float], r0: float, c1: tuple[float, float], r1: float
) -> list[tuple[float, float]]:
    x0, y0 = c0
    x1, y1 = c1
    dx, dy = x1 - x0, y1 - y0
    d = math.hypot(dx, dy)
    if d < 1e-15 or r0 < 0 or r1 < 0:
        return []
    if d > r0 + r1 + 1e-12 or d < abs(r0 - r1) - 1e-12:
        return []
    a = (r0 * r0 - r1 * r1 + d * d) / (2.0 * d)
    h2 = r0 * r0 - a * a
    if h2 < -1e-12:
        return []
    h = math.sqrt(max(h2, 0.0))
    xm = x0 + a * dx / d
    ym = y0 + a * dy / d
    if h < 1e-15:
        return [(xm, ym)]
    rx, ry = -dy * (h / d), dx * (h / d)
    return [(xm + rx, ym + ry), (xm - rx, ym - ry)]


def _fillet_arc(
    center: tuple[float, float],
    radius: float,
    p0: tuple[float, float],
    p1: tuple[float, float],
    prefer: tuple[float, float],
    n: int = 12,
) -> list[tuple[float, float]]:
    cx, cy = center
    r = max(float(radius), 1e-15)
    a0 = math.atan2(p0[1] - cy, p0[0] - cx)
    a1 = math.atan2(p1[1] - cy, p1[0] - cx)
    n = max(int(n), 3)

    def sweep(ccw: bool) -> list[float]:
        da = a1 - a0
        while da <= -math.pi:
            da += 2 * math.pi
        while da > math.pi:
            da -= 2 * math.pi
        if ccw and da < 0:
            da += 2 * math.pi
        if (not ccw) and da > 0:
            da -= 2 * math.pi
        return [a0 + da * i / (n - 1) for i in range(n)]

    def score(angs: list[float]) -> float:
        mid = angs[len(angs) // 2]
        px = cx + r * math.cos(mid)
        py = cy + r * math.sin(mid)
        return -((px - prefer[0]) ** 2 + (py - prefer[1]) ** 2)

    best = sweep(True)
    alt = sweep(False)
    if score(alt) > score(best):
        best = alt
    return [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in best]



def impulse_bucket_profile(
    *,
    chord_m: float,
    upper_sagitta_c: float = 0.50,
    lower_sagitta_c: float = 0.22,
    le_fillet_r_c: float = 0.04,
    te_fillet_r_c: float = 0.04,
    upper_sagitta_m: float | None = None,
    lower_sagitta_m: float | None = None,
    le_fillet_r_m: float | None = None,
    te_fillet_r_m: float | None = None,
    n_points: int = 160,
    pitch_m: float | None = None,
) -> list[tuple[float, float]]:
    """C-bucket: dual-arc Pelton section opening to the inlet (-x).

    Outer + inner circular arcs share LE/TE. Metal is the band between them.
    Absolute sagittas/fillets (*_m) do not scale with chord — dragging c
    stretches axial width and drops h/c. Fractional *_c is legacy /c mode.
    Passage is the pitch *gap*, not the solid. Not NACA-on-circular-camber.
    """
    c = max(float(chord_m), 1e-9)
    if upper_sagitta_m not in (None, ""):
        h_u = max(float(upper_sagitta_m), 1e-9)
    else:
        h_u = max(float(upper_sagitta_c) * c, 1e-9)
    if lower_sagitta_m not in (None, ""):
        h_l = max(float(lower_sagitta_m), 1e-12)
    else:
        h_l = max(float(lower_sagitta_c) * c, 1e-12)
    if h_l >= h_u - 1e-9:
        h_l = max(1e-12, 0.4 * h_u)
    n = max(int(n_points), 48)
    upper = circular_arc_le_te(c, h_u, n)
    lower = circular_arc_le_te(c, h_l, n)
    if le_fillet_r_m not in (None, ""):
        r_le = max(float(le_fillet_r_m), 0.0)
    else:
        r_le = max(float(le_fillet_r_c) * c, 0.0)
    if te_fillet_r_m not in (None, ""):
        r_te = max(float(te_fillet_r_m), 0.0)
    else:
        r_te = max(float(te_fillet_r_c) * c, 0.0)
    if pitch_m is not None and pitch_m > 0:
        # Fillet diameter must not eat the inter-bucket gap.
        r_cap = 0.45 * max(h_u - h_l, 1e-12)
        r_cap = min(r_cap, 0.45 * float(pitch_m), 0.20 * c)
        r_le = min(r_le, r_cap)
        r_te = min(r_te, r_cap)

    if r_le <= 1e-15 and r_te <= 1e-15:
        poly: list[tuple[float, float]] = list(upper)
        for i in range(n - 2, 0, -1):
            poly.append(lower[i])
        poly.append(upper[0])
        poly = _closed(poly)
        if polygon_signed_area(poly) < 0:
            poly = list(reversed(poly))
            poly = _closed(poly)
        if abs(polygon_signed_area(poly)) < 1e-16:
            raise RuntimeError("impulse bucket has zero area")
        if polygon_self_intersects(poly):
            raise RuntimeError("impulse bucket is self-intersecting")
        poly[-1] = poly[0]
        return poly

    cx_u, cy_u, R_u = _arc_circle(c, h_u)
    cx_l, cy_l, R_l = _arc_circle(c, h_l)
    Cu, Cl = (cx_u, cy_u), (cx_l, cy_l)

    def tip(r: float, want_le: bool):
        if r <= 1e-15 or R_u <= r + 1e-12:
            return None
        hits = _circle_hits(Cu, R_u - r, Cl, R_l + r)
        if not hits:
            return None
        F = min(hits, key=lambda p: p[0]) if want_le else max(hits, key=lambda p: p[0])
        dxu, dyu = F[0] - Cu[0], F[1] - Cu[1]
        Lu = math.hypot(dxu, dyu) or 1.0
        p_u = (Cu[0] + dxu / Lu * R_u, Cu[1] + dyu / Lu * R_u)
        dxl, dyl = F[0] - Cl[0], F[1] - Cl[1]
        Ll = math.hypot(dxl, dyl) or 1.0
        p_l = (Cl[0] + dxl / Ll * R_l, Cl[1] + dyl / Ll * R_l)
        return {"F": F, "r": r, "p_u": p_u, "p_l": p_l}

    le = tip(r_le, True)
    te = tip(r_te, False)

    def nearest(pts, q):
        return min(range(len(pts)), key=lambda i: (pts[i][0] - q[0]) ** 2 + (pts[i][1] - q[1]) ** 2)

    i_u0 = nearest(upper, le["p_u"]) if le else 0
    i_u1 = nearest(upper, te["p_u"]) if te else n - 1
    i_l0 = nearest(lower, le["p_l"]) if le else 0
    i_l1 = nearest(lower, te["p_l"]) if te else n - 1
    i_u0 = max(0, min(i_u0, n - 3))
    i_u1 = max(i_u0 + 2, min(i_u1, n - 1))
    i_l0 = max(0, min(i_l0, n - 3))
    i_l1 = max(i_l0 + 2, min(i_l1, n - 1))

    poly = []
    if le:
        poly.append(le["p_u"])
        start_u = i_u0 + 1
    else:
        poly.append(upper[0])
        start_u = 1
    end_u = i_u1 if te else n - 1
    for i in range(start_u, end_u):
        poly.append(upper[i])
    if te:
        poly.append(te["p_u"])
        poly.extend(_fillet_arc(te["F"], te["r"], te["p_u"], te["p_l"], (c, 0.0), 12)[1:])
    else:
        poly.append(upper[-1])
    start_l = (i_l1 - 1) if te else (n - 2)
    end_l = i_l0 if le else 0
    for i in range(start_l, end_l, -1):
        poly.append(lower[i])
    if le:
        poly.append(le["p_l"])
        poly.extend(_fillet_arc(le["F"], le["r"], le["p_l"], le["p_u"], (0.0, 0.0), 14)[1:])
    else:
        poly.append(lower[0])
    poly = _closed(poly)
    if polygon_signed_area(poly) < 0:
        poly = list(reversed(poly))
        poly = _closed(poly)
    if abs(polygon_signed_area(poly)) < 1e-16:
        raise RuntimeError("impulse bucket has zero area")
    if polygon_self_intersects(poly):
        raise RuntimeError("impulse bucket is self-intersecting")
    poly[-1] = poly[0]
    return poly


def live_bucket_profile(
    *,
    chord_m: float,
    beta1_metal_deg: float = 72.0,
    beta2_metal_deg: float = -72.0,
    upper_sagitta_c: float = 0.50,
    lower_sagitta_c: float = 0.22,
    le_fillet_r_c: float = 0.04,
    te_fillet_r_c: float = 0.04,
    upper_sagitta_m: float | None = None,
    lower_sagitta_m: float | None = None,
    le_fillet_r_m: float | None = None,
    te_fillet_r_m: float | None = None,
    lin_m: float | None = None,
    lout_m: float | None = None,
    r_tr_m: float | None = None,
    r_main_m: float | None = None,
    t_m: float | None = None,
    psi_tr_deg: float | None = None,
    n_points: int = 160,
    pitch_m: float | None = None,
) -> list[tuple[float, float]]:
    """One turbine section. L_in=L_out=0 → dual-arc + circular LE/TE fillets.
    L_in/L_out>0 → pointed parallel-sided legs + round fillet caps (never flat).
    Millimetre knobs are absolute; chord is axial length, not a zoom.
    """
    lin = float(lin_m or 0.0)
    lout = float(lout_m or 0.0)
    if lin <= 1e-12 and lout <= 1e-12:
        return impulse_bucket_profile(
            chord_m=chord_m,
            upper_sagitta_c=upper_sagitta_c,
            lower_sagitta_c=lower_sagitta_c,
            le_fillet_r_c=le_fillet_r_c,
            te_fillet_r_c=te_fillet_r_c,
            upper_sagitta_m=upper_sagitta_m,
            lower_sagitta_m=lower_sagitta_m,
            le_fillet_r_m=le_fillet_r_m,
            te_fillet_r_m=te_fillet_r_m,
            n_points=n_points,
            pitch_m=pitch_m,
        )
    return pointed_bucket_profile(
        chord_m=chord_m,
        beta1_metal_deg=beta1_metal_deg,
        beta2_metal_deg=beta2_metal_deg,
        lin_m=lin,
        lout_m=lout,
        r_tr_m=r_tr_m,
        r_main_m=r_main_m,
        t_m=t_m,
        psi_tr_deg=psi_tr_deg,
        le_fillet_r_m=le_fillet_r_m,
        te_fillet_r_m=te_fillet_r_m,
        le_fillet_r_c=le_fillet_r_c,
        te_fillet_r_c=te_fillet_r_c,
        upper_sagitta_m=upper_sagitta_m,
        lower_sagitta_m=lower_sagitta_m,
        n=max(16, int(n_points) // 6),
    )


def _tangents_to_circle(
    P: tuple[float, float], C: tuple[float, float], R: float
) -> list[tuple[float, float]]:
    dx, dy = P[0] - C[0], P[1] - C[1]
    d = math.hypot(dx, dy)
    if d <= R + 1e-10 or R <= 0:
        return []
    beta = math.acos(max(-1.0, min(1.0, R / d)))
    ang = math.atan2(dy, dx)
    return [
        (C[0] + R * math.cos(ang + s), C[1] + R * math.sin(ang + s))
        for s in (beta, -beta)
    ]


def _pick_tangency(
    cands: list[tuple[float, float]],
    arc: list[tuple[float, float]],
    want_upper: bool,
) -> tuple[float, float] | None:
    if not cands:
        return None
    xs = [p[0] for p in arc]
    ys = [p[1] for p in arc]
    xmin, xmax = min(xs) - 1e-6, max(xs) + 1e-6
    ymin, ymax = min(ys) - 1e-6, max(ys) + 1e-6

    def ok(p):
        return xmin <= p[0] <= xmax and ymin <= p[1] <= ymax

    on = [p for p in cands if ok(p)]
    pool = on or cands
    return max(pool, key=lambda p: p[1]) if want_upper else min(pool, key=lambda p: p[1])


def _fillet_between_straights(
    T: tuple[float, float],
    A: tuple[float, float],
    B: tuple[float, float],
    r: float,
) -> list[tuple[float, float]]:
    """Points from A-side tangent to B-side, replacing T. r=0 → [T]."""
    if r <= 1e-15:
        return [T]
    va = (A[0] - T[0], A[1] - T[1])
    vb = (B[0] - T[0], B[1] - T[1])
    la = math.hypot(*va) or 1.0
    lb = math.hypot(*vb) or 1.0
    ua, ub = (va[0] / la, va[1] / la), (vb[0] / lb, vb[1] / lb)
    dot = max(-1.0, min(1.0, ua[0] * ub[0] + ua[1] * ub[1]))
    alpha = 0.5 * math.acos(dot)
    if alpha < 1e-6 or abs(math.sin(alpha)) < 1e-9:
        return [T]
    d = r / math.tan(alpha)
    if d > 0.90 * la or d > 0.90 * lb:
        return [T]
    Pa = (T[0] + ua[0] * d, T[1] + ua[1] * d)
    Pb = (T[0] + ub[0] * d, T[1] + ub[1] * d)
    bx, by = ua[0] + ub[0], ua[1] + ub[1]
    bl = math.hypot(bx, by) or 1.0
    F = (T[0] + bx / bl * (r / math.sin(alpha)), T[1] + by / bl * (r / math.sin(alpha)))
    arc = _fillet_arc(F, r, Pa, Pb, T, 12)
    if not arc:
        return [T]
    return arc


def _fillet_circle_to_line(
    Cl: tuple[float, float],
    Rl: float,
    line_p: tuple[float, float],
    line_h: float,
    R: float,
    *,
    want_le: bool,
    chord_m: float,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None:
    """Fillet center F, tangency on circle, tangency on line. Or None."""
    if R <= 1e-9 or Rl <= 1e-9:
        return None
    u = (math.cos(line_h), math.sin(line_h))
    n = (-math.sin(line_h), math.cos(line_h))
    cands: list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = []
    for rho in (Rl + R, abs(Rl - R)):
        if rho < 1e-9:
            continue
        for sign in (1.0, -1.0):
            V0 = (
                line_p[0] - Cl[0] + sign * R * n[0],
                line_p[1] - Cl[1] + sign * R * n[1],
            )
            b = V0[0] * u[0] + V0[1] * u[1]
            c0 = V0[0] * V0[0] + V0[1] * V0[1] - rho * rho
            disc = b * b - c0
            if disc < -1e-16:
                continue
            root = math.sqrt(max(disc, 0.0))
            for s in (-b + root, -b - root):
                F = (
                    line_p[0] + s * u[0] + sign * R * n[0],
                    line_p[1] + s * u[1] + sign * R * n[1],
                )
                vx, vy = F[0] - Cl[0], F[1] - Cl[1]
                lv = math.hypot(vx, vy) or 1.0
                pc = (Cl[0] + vx / lv * Rl, Cl[1] + vy / lv * Rl)
                pln = (F[0] - sign * R * n[0], F[1] - sign * R * n[1])
                if pc[1] < -0.002:
                    continue
                if want_le and pc[0] > 0.55 * chord_m:
                    continue
                if (not want_le) and pc[0] < 0.45 * chord_m:
                    continue
                cands.append((F, pc, pln))
    if not cands:
        return None
    # prefer F above the inner wall (cuts the down-kink) and pc not past mid
    def key(item):
        F, pc, pln = item
        return (-F[1], abs(pc[0] - (0.0 if want_le else chord_m)))
    return min(cands, key=key)



def pointed_bucket_profile(
    *,
    chord_m: float,
    beta1_metal_deg: float,
    beta2_metal_deg: float,
    lin_m: float,
    lout_m: float,
    r_tr_m: float | None = None,
    r_main_m: float | None = None,
    t_m: float | None = None,
    psi_tr_deg: float | None = None,
    le_fillet_r_m: float | None = None,
    te_fillet_r_m: float | None = None,
    le_fillet_r_c: float = 0.04,
    te_fillet_r_c: float = 0.04,
    upper_sagitta_m: float | None = None,
    lower_sagitta_m: float | None = None,
    n: int = 24,
) -> list[tuple[float, float]]:
    """Left-side recipe (mirrored on the right): outer G1 along the C, inner trans-arc, then parallel t.

    Trim where gap = t. Outer straight follows the outer-arc tangent (original path).
    Inner wall uses R_tr to that heading, then runs parallel (constant width).
    t/2 nose at the end. ψ is extra splay off that tangent, default 0.
    """
    c = max(float(chord_m), 1e-9)
    L_in = max(float(lin_m), 0.0)
    L_out = max(float(lout_m), 0.0)
    if upper_sagitta_m not in (None, ""):
        h_u = max(float(upper_sagitta_m), 1e-9)
    else:
        h_u = 0.005
    if lower_sagitta_m not in (None, ""):
        h_l = max(float(lower_sagitta_m), 1e-12)
    else:
        h_l = 0.0022
    if h_l >= h_u - 1e-9:
        h_l = max(1e-12, 0.4 * h_u)
    if t_m not in (None, ""):
        thick = max(float(t_m), 1e-6)
    else:
        thick = max(0.5 * (h_u - h_l), 1e-6)
    g_tgt = min(thick, 0.90 * (h_u - h_l))
    if r_tr_m not in (None, ""):
        R_tr = max(float(r_tr_m), 1e-6)
    else:
        R_tr = 0.002
    if r_main_m not in (None, ""):
        R_inner = max(float(r_main_m), 1e-6)
    else:
        R_inner = 0.006
    R_inner = min(R_inner, 0.006)
    psi = float(psi_tr_deg) if psi_tr_deg not in (None, "") else 0.0
    nn = max(int(n), 16)
    n_arc = max(nn * 6, 96)
    upper = circular_arc_le_te(c, h_u, n_arc)
    lower = circular_arc_le_te(c, h_l, n_arc)
    Cl = _arc_circle(c, h_l)

    def _wrap(a: float) -> float:
        while a <= -math.pi:
            a += 2.0 * math.pi
        while a > math.pi:
            a -= 2.0 * math.pi
        return a

    def _hdg(a, b) -> float:
        return math.atan2(b[1] - a[1], b[0] - a[0])

    def _y_at_x(pts, x):
        for i in range(len(pts) - 1):
            x0, y0 = pts[i]
            x1, y1 = pts[i + 1]
            if (x0 - x) * (x1 - x) <= 0.0:
                if abs(x1 - x0) < 1e-15:
                    return y0, i
                tt = (x - x0) / (x1 - x0)
                return y0 + tt * (y1 - y0), i
        return pts[-1][1], len(pts) - 2

    def _mouth_x(want_le: bool) -> float:
        xs = [p[0] for p in upper]
        xmin, xmax = min(xs), max(xs)
        if want_le:
            lo, hi = xmin + 1e-9, 0.5 * (xmin + xmax)
        else:
            lo, hi = 0.5 * (xmin + xmax), xmax - 1e-9
        hit = hi if want_le else lo
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            yu, _ = _y_at_x(upper, mid)
            yl, _ = _y_at_x(lower, mid)
            g = yu - yl
            if want_le:
                if g >= g_tgt:
                    hit, hi = mid, mid
                else:
                    lo = mid
            else:
                if g >= g_tgt:
                    hit, lo = mid, mid
                else:
                    hi = mid
        return hit

    def _arc_to(pts, p, h, R, h_tgt, other):
        dpsi = _wrap(h_tgt - h)
        if abs(dpsi) < 1e-5:
            return p, h
        Cplus = (p[0] + R * math.cos(h + math.pi / 2.0), p[1] + R * math.sin(h + math.pi / 2.0))
        Cminus = (p[0] + R * math.cos(h - math.pi / 2.0), p[1] + R * math.sin(h - math.pi / 2.0))
        ox, oy = other[0] - p[0], other[1] - p[1]
        def _dot(C):
            return (C[0] - p[0]) * ox + (C[1] - p[1]) * oy
        s = 1.0 if _dot(Cplus) < _dot(Cminus) else -1.0
        d_forced = dpsi
        if s > 0 and d_forced < 0:
            d_forced += 2.0 * math.pi
        elif s < 0 and d_forced > 0:
            d_forced -= 2.0 * math.pi
        if abs(d_forced) <= math.pi + 1e-6:
            dpsi = d_forced
            s_use = s
        else:
            s_use = 1.0 if dpsi > 0 else -1.0
        left = h + s_use * math.pi / 2.0
        C = (p[0] + R * math.cos(left), p[1] + R * math.sin(left))
        a0 = math.atan2(p[1] - C[1], p[0] - C[0])
        a1 = a0 + dpsi
        ns = max(int(nn * abs(dpsi) / (math.pi / 4.0)), 8)
        for i in range(1, ns + 1):
            a = a0 + (a1 - a0) * i / ns
            pts.append((C[0] + R * math.cos(a), C[1] + R * math.sin(a)))
        return pts[-1], h_tgt

    def _straight(pts, p, h, L, nseg=12):
        if L <= 1e-12:
            return p, h
        ex = p[0] + L * math.cos(h)
        ey = p[1] + L * math.sin(h)
        for i in range(1, nseg + 1):
            tt = i / nseg
            pts.append((p[0] + tt * (ex - p[0]), p[1] + tt * (ey - p[1])))
        return (ex, ey), h

    def _cap(u, l, h, r_fillet: float):
        """Nose: r=0 is a point; r>0 is a circular cap of that radius, G1 to the walls.

        Parallel stems at thickness t only admit a tangent circle of r ≤ t/2.
        Smaller r tapers the last bit; r=0 is a cusp. Never a t/2 dummy.
        """
        tloc = math.hypot(u[0] - l[0], u[1] - l[1])
        if tloc < 1e-12:
            return []
        ux = (u[0] - l[0]) / tloc
        uy = (u[1] - l[1]) / tloc
        mx, my = 0.5 * (u[0] + l[0]), 0.5 * (u[1] + l[1])
        hx, hy = math.cos(h), math.sin(h)
        r = max(float(r_fillet), 0.0)
        half = 0.5 * tloc
        if r < 1e-7:
            apex = (mx + 0.12 * tloc * hx, my + 0.12 * tloc * hy)
            return [apex]
        r_use = min(r, half - 1e-9)
        shrink = half - r_use
        u2 = (u[0] - ux * shrink + shrink * hx, u[1] - uy * shrink + shrink * hy)
        l2 = (l[0] + ux * shrink + shrink * hx, l[1] + uy * shrink + shrink * hy)
        mx2, my2 = 0.5 * (u2[0] + l2[0]), 0.5 * (u2[1] + l2[1])
        a0 = math.atan2(u2[1] - my2, u2[0] - mx2)
        a1 = math.atan2(l2[1] - my2, l2[0] - mx2)
        da = _wrap(a1 - a0)
        mid = a0 + 0.5 * da
        through = (mx2 + r_use * math.cos(mid), my2 + r_use * math.sin(mid))
        prefer = (mx2 + r_use * hx, my2 + r_use * hy)
        if (through[0] - mx2) * (prefer[0] - mx2) + (through[1] - my2) * (prefer[1] - my2) < 0:
            da = da - 2.0 * math.pi if da > 0 else da + 2.0 * math.pi
        ncap = max(10, int(12 * max(r_use / max(half, 1e-9), 0.35)))
        arc = [
            (mx2 + r_use * math.cos(a0 + da * i / ncap), my2 + r_use * math.sin(a0 + da * i / ncap))
            for i in range(0, ncap + 1)
        ]
        return arc

    x_le = _mouth_x(True)
    x_te = _mouth_x(False)
    yu_le, iu_le = _y_at_x(upper, x_le)
    yl_le, il_le = _y_at_x(lower, x_le)
    yu_te, iu_te = _y_at_x(upper, x_te)
    yl_te, il_te = _y_at_x(lower, x_te)
    pu_le, pl_le = (x_le, yu_le), (x_le, yl_le)
    pu_te, pl_te = (x_te, yu_te), (x_te, yl_te)
    u_mid = [pu_le] + [p for p in upper if x_le < p[0] < x_te] + [pu_te]
    l_mid = [pl_le] + [p for p in lower if x_le < p[0] < x_te] + [pl_te]

    # outer tangent at the mouth (toward the tip)
    h_u_le = _hdg(u_mid[1], u_mid[0]) if len(u_mid) > 1 else -math.pi / 2.0
    h_l_le = _hdg(l_mid[1], l_mid[0]) if len(l_mid) > 1 else -math.pi / 2.0
    h_u_te = _hdg(u_mid[-2], u_mid[-1]) if len(u_mid) > 1 else -math.pi / 2.0
    h_l_te = _hdg(l_mid[-2], l_mid[-1]) if len(l_mid) > 1 else -math.pi / 2.0
    # extra splay off the outer path; 0 follows the C
    h_stem_le = h_u_le - math.radians(psi)
    h_stem_te = h_u_te + math.radians(psi)

    def _inner_start(want_le: bool):
        # leave the inner C further inboard so the trans arc can be large
        x_join = min(0.28 * c, 0.5 * (x_le + x_te))
        if want_le:
            x = max(x_le + 0.08 * c, min(x_join, 0.32 * c))
        else:
            x = min(x_te - 0.08 * c, max(c - x_join, 0.68 * c))
        y, _ = _y_at_x(lower, x)
        return (x, y)

    def _side(pu, pl, h_outer, h_inner, h_stem, L, want_le: bool):
        u_extra: list[tuple[float, float]] = []
        l_extra: list[tuple[float, float]] = []
        if abs(_wrap(h_stem - h_outer)) > math.radians(2.0):
            _arc_to(u_extra, pu, h_outer, R_tr, h_stem, pl)
        _straight(u_extra, u_extra[-1] if u_extra else pu, h_stem, L)
        p_in = _inner_start(want_le)
        # heading of inner C at p_in, toward the tip
        if want_le:
            h_in = _hdg((p_in[0] + 1e-4, _y_at_x(lower, p_in[0] + 1e-4)[0]), p_in)
        else:
            h_in = _hdg((p_in[0] - 1e-4, _y_at_x(lower, p_in[0] - 1e-4)[0]), p_in)
        # large trans; center away from outer = below, cuts inside the red kink
        R_use = min(R_inner, max(0.003, 0.9 * (L + 0.004)))
        dpsi = _wrap(h_stem - h_in)
        if abs(dpsi) > 1e-5:
            s = 1.0 if dpsi > 0.0 else -1.0
            left = h_in + s * math.pi / 2.0
            Cfil = (p_in[0] + R_use * math.cos(left), p_in[1] + R_use * math.sin(left))
            a0 = math.atan2(p_in[1] - Cfil[1], p_in[0] - Cfil[0])
            a1 = a0 + dpsi
            ns = max(int(nn * abs(dpsi) / (math.pi / 4.0)), 10)
            for i in range(1, ns + 1):
                a = a0 + (a1 - a0) * i / ns
                l_extra.append((Cfil[0] + R_use * math.cos(a), Cfil[1] + R_use * math.sin(a)))
            p_line = l_extra[-1]
        else:
            p_line = p_in
        if u_extra:
            tip_u = u_extra[-1]
            s_need = (tip_u[0] - p_line[0]) * math.cos(h_stem) + (tip_u[1] - p_line[1]) * math.sin(h_stem)
            _straight(l_extra, p_line, h_stem, max(s_need, 0.0))
        elif L > 1e-12:
            _straight(l_extra, p_line, h_stem, L)
        return u_extra, l_extra, p_in

    u_left, l_left, pc_le = _side(pu_le, pl_le, h_u_le, h_l_le, h_stem_le, L_in, True)
    u_right, l_right, pc_te = _side(pu_te, pl_te, h_u_te, h_l_te, h_stem_te, L_out, False)
    if pc_le is not None and pc_te is not None and pc_te[0] > pc_le[0] + 1e-6:
        l_mid = [pc_le] + [p for p in lower if pc_le[0] < p[0] < pc_te[0]] + [pc_te]

    if le_fillet_r_m not in (None, ""):
        r_le = max(float(le_fillet_r_m), 0.0)
    else:
        r_le = max(float(le_fillet_r_c) * c, 0.0)
    if te_fillet_r_m not in (None, ""):
        r_te = max(float(te_fillet_r_m), 0.0)
    else:
        r_te = max(float(te_fillet_r_c) * c, 0.0)

    u_tip_le = u_left[-1] if u_left else pu_le
    l_tip_le = l_left[-1] if l_left else pl_le
    u_tip_te = u_right[-1] if u_right else pu_te
    l_tip_te = l_right[-1] if l_right else pl_te
    te_cap = _cap(u_tip_te, l_tip_te, h_stem_te, r_te)
    le_cap = _cap(l_tip_le, u_tip_le, h_stem_le, r_le)

    poly: list[tuple[float, float]] = []
    poly.extend(reversed(u_left))
    poly.extend(u_mid)
    poly.extend(u_right)
    poly.extend(te_cap)
    poly.extend(reversed(l_right))
    poly.extend(reversed(l_mid[1:-1]))
    poly.append(l_mid[0])
    poly.extend(l_left)
    poly.extend(le_cap)
    cleaned: list[tuple[float, float]] = []
    for q in poly:
        if not cleaned or math.hypot(q[0] - cleaned[-1][0], q[1] - cleaned[-1][1]) > 1e-12:
            cleaned.append(q)
    poly = _closed(cleaned)
    if polygon_signed_area(poly) < 0:
        poly = _closed(list(reversed(poly[:-1])))
    if abs(polygon_signed_area(poly)) < 1e-16:
        raise RuntimeError("pointed bucket has zero area")
    if polygon_self_intersects(poly):
        raise RuntimeError("pointed bucket is self-intersecting")
    poly[-1] = poly[0]
    return poly
