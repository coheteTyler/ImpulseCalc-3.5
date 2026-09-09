"""Goldman / TN D-4421 vortex-impulse blade. PREDICTED.

Designer inputs: γ, Mw1 (=Mi), Mw2 (default Mi), ML, MU, βi.
U, ṁ, OF never enter.

Construction (NASA TN D-4421 / FORTRAN listing, Goldman & Scullin 1968):
  vortex core M* R* = 1 on concentric circular arcs
  inlet/outlet transition arcs by MOC (straight Mach lines × wall segments of Δν)
  LE/TE close-out: short straights parallel to inlet/outlet flow (fig. 1, BD)

Not the 1968 printer-plot FORTRAN. Same equations. Unique incidence is NOT
an output of D-4421 — βi is an input. Axial-subsonic unique-incidence flag
lives on the meanline (Max,1 = Mw1 cos βi).
"""

from __future__ import annotations

import math
from typing import Any, Callable


def m_star(M: float, gamma: float) -> float:
    g = float(gamma)
    m = max(float(M), 1e-6)
    return m * math.sqrt((g + 1.0) / (2.0 + (g - 1.0) * m * m))


def mach_from_mstar(ms: float, gamma: float) -> float:
    g = float(gamma)
    gampi = (g + 1.0) / 2.0
    gammi = (g - 1.0) / 2.0
    ms = max(float(ms), 1e-9)
    den = 1.0 - (gammi / gampi) * ms * ms
    if den <= 1e-12:
        return 1e6
    return math.sqrt((ms * ms / gampi) / den)


def prandtl_meyer(M: float, gamma: float) -> float:
    """ν(M) radians. 0 if M≤1."""
    g = float(gamma)
    m = float(M)
    if m <= 1.0 + 1e-12:
        return 0.0
    k = math.sqrt((g + 1.0) / (g - 1.0))
    a = math.sqrt((g - 1.0) / (g + 1.0) * (m * m - 1.0))
    b = math.sqrt(m * m - 1.0)
    return k * math.atan(a) - math.atan(b)


def _clip_asin(x: float) -> float:
    return math.asin(max(-1.0, min(1.0, x)))


def fofrs(Rstar: float, gamma: float) -> float:
    """D-4421 eq (10b) / FUNCTION FOFRS. R* dimensionless vortex radius."""
    g = float(gamma)
    gammi = (g - 1.0) / 2.0
    gampi = (g + 1.0) / 2.0
    perm = math.sqrt(gampi / gammi)
    x = max(float(Rstar), 1e-9)
    arg1 = 2.0 * gammi / (x * x) - g
    arg2 = 2.0 * gampi * x * x - g
    return perm * _clip_asin(arg1) + _clip_asin(arg2)


def _f_target(V: float, fn: float, delv: float, gamma: float) -> float:
    """FORTRAN F(V,FN) = 2V − (π/2)(PERM−1) − 2(FN−1)Δν. V, Δν radians."""
    g = float(gamma)
    perm = math.sqrt(((g + 1.0) / 2.0) / ((g - 1.0) / 2.0))
    return (2.0 * V) - (0.5 * math.pi * (perm - 1.0)) - (2.0 * (fn - 1.0) * delv)


def invert_fofrs(target: float, gamma: float) -> float:
    """R* in [1/PERM, 1) such that FOFRS(R*) = target. Bisection."""
    g = float(gamma)
    perm = math.sqrt(((g + 1.0) / 2.0) / ((g - 1.0) / 2.0))
    lo, hi = 1.0 / perm + 1e-12, 0.999999
    flo, fhi = fofrs(lo, g) - target, fofrs(hi, g) - target
    if flo * fhi > 0:
        # pick nearer bound
        return lo if abs(flo) < abs(fhi) else hi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        fm = fofrs(mid, g) - target
        if abs(fm) < 1e-12 or (hi - lo) < 1e-12:
            return mid
        if flo * fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


def _um_of_R(Rstar: float, gamma: float) -> float:
    """Mach-angle-like term in the FORTRAN: asin sqrt(GAMPI R*^2 − GAMMI)."""
    g = float(gamma)
    gampi = (g + 1.0) / 2.0
    gammi = (g - 1.0) / 2.0
    return _clip_asin(math.sqrt(max(0.0, gampi * Rstar * Rstar - gammi)))


def _close(poly: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pts = [(float(x), float(y)) for x, y in poly]
    if len(pts) < 3:
        return pts
    if abs(pts[0][0] - pts[-1][0]) > 1e-14 or abs(pts[0][1] - pts[-1][1]) > 1e-14:
        pts.append(pts[0])
    return pts


def _signed_area(poly: list[tuple[float, float]]) -> float:
    pts = poly[:-1] if poly and poly[0] == poly[-1] else poly
    a = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return 0.5 * a


def _rot(x: float, y: float, ang: float) -> tuple[float, float]:
    c, s = math.cos(ang), math.sin(ang)
    return x * c - y * s, x * s + y * c


def beta_out_deg(Mi: float, Mo: float, beta_in_deg: float, gamma: float) -> float:
    """D-4421 eq (9). Impulse Mi=Mo ⇒ βo = −βi."""
    g = float(gamma)
    gammi = (g - 1.0) / 2.0
    gampi = (g + 1.0) / 2.0
    bi = math.radians(beta_in_deg)
    if abs(Mo - Mi) < 1e-12:
        return -float(beta_in_deg)
    temp = ((gammi * Mo * Mo + 1.0) / (gammi * Mi * Mi + 1.0)) ** (gampi / (2.0 * gammi))
    arg = math.cos(bi) * (Mi / max(Mo, 1e-9)) * temp
    arg = max(-1.0, min(1.0, arg))
    return -math.degrees(math.acos(arg))


def _lower_transition(
    *,
    R_l: float,
    nu_long: float,
    nu_l: float,
    kmax: int,
    delv: float,
    gamma: float,
) -> list[tuple[float, float]]:
    """Unrotated lower trans. Junction at (0, R_l), k decreasing. FORTRAN 2791–2830."""
    if kmax < 1:
        return [(0.0, R_l)]
    xs: list[float] = [0.0]
    ys: list[float] = [R_l]
    txlo, tylo = 0.0, R_l
    phikp1 = -(nu_long - nu_l) + kmax * delv
    umkp1 = _um_of_R(R_l, gamma)
    for kk in range(1, kmax + 1):
        k = kmax + 1 - kk
        phik = phikp1 - delv
        tr = invert_fofrs(_f_target(nu_long, float(k), delv, gamma), gamma)
        tx = tr * math.sin(phik)
        ty = tr * math.cos(phik)
        emwk = math.tan(-phikp1) if abs(math.cos(phikp1)) > 1e-12 else 1e6
        umk = _um_of_R(tr, gamma)
        emk = -math.tan((phik + umk + phikp1 + umkp1) / 2.0)
        temp = tylo - emwk * txlo
        tempp = ty - emk * tx
        temppp = emk - emwk
        if abs(temppp) < 1e-16:
            txlo, tylo = tx, ty
        else:
            txlo = (temp - tempp) / temppp
            tylo = (emk * temp - emwk * tempp) / temppp
        xs.append(txlo)
        ys.append(tylo)
        phikp1 = phik
        umkp1 = umk
    return list(zip(xs, ys))


def _upper_transition(
    *,
    R_u: float,
    nu_short: float,
    nu_u: float,
    jmax: int,
    delv: float,
    gamma: float,
) -> list[tuple[float, float]]:
    """Unrotated upper trans. Junction (0, R_u). FORTRAN 2860+."""
    if jmax < 1:
        return [(0.0, R_u)]
    xs: list[float] = [0.0]
    ys: list[float] = [R_u]
    txup, tyup = 0.0, R_u
    phijp1 = -(nu_u - nu_short) + jmax * delv
    umjp1 = _um_of_R(R_u, gamma)
    for jj in range(1, jmax + 1):
        j = jmax + 1 - jj
        phij = phijp1 - delv
        tr = invert_fofrs(_f_target(nu_short, float(-j + 2), delv, gamma), gamma)
        tx = tr * math.sin(phij)
        ty = tr * math.cos(phij)
        emwj = math.tan(-phijp1) if abs(math.cos(phijp1)) > 1e-12 else 1e6
        umj = _um_of_R(tr, gamma)
        emj = math.tan((-phij + umj - phijp1 + umjp1) / 2.0)
        temp = tyup - emwj * txup
        tempp = ty - emj * tx
        temppp = emj - emwj
        if abs(temppp) < 1e-16:
            txup, tyup = tx, ty
        else:
            txup = (temp - tempp) / temppp
            tyup = (emj * temp - emwj * tempp) / temppp
        xs.append(txup)
        ys.append(tyup)
        phijp1 = phij
        umjp1 = umj
    return list(zip(xs, ys))


def _rotate_pts(pts: list[tuple[float, float]], ang: float) -> list[tuple[float, float]]:
    return [_rot(x, y, ang) for x, y in pts]


def _flip_x(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return [(-x, y) for x, y in pts]


def _dedup_pts(pts: list[tuple[float, float]], eps: float = 1e-14) -> list[tuple[float, float]]:
    if not pts:
        return pts
    out = [pts[0]]
    for q in pts[1:]:
        if abs(q[0] - out[-1][0]) > eps or abs(q[1] - out[-1][1]) > eps:
            out.append(q)
    return out


def _bd_station(
    pa: tuple[float, float],
    pb: tuple[float, float],
    heading_rad: float,
    *,
    upstream: bool,
    extra: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Finite BD: both walls to one plane perpendicular to β, plus a short stub.

    upstream=True is the LE (against +β). Connecting pa'–pb' is the fig. 1 BD face.
    """
    ux, uy = math.cos(heading_rad), math.sin(heading_rad)
    sa = pa[0] * ux + pa[1] * uy
    sb = pb[0] * ux + pb[1] * uy
    extra = max(float(extra), 0.0)
    if upstream:
        s = min(sa, sb) - extra
    else:
        s = max(sa, sb) + extra

    def ext(p: tuple[float, float], s0: float) -> tuple[float, float]:
        ds = s - s0
        return (p[0] + ds * ux, p[1] + ds * uy)

    return ext(pa, sa), ext(pb, sb)


def _arc_about_origin(
    p0: tuple[float, float],
    p1: tuple[float, float],
    n: int,
) -> list[tuple[float, float]]:
    r0 = math.hypot(*p0)
    r1 = math.hypot(*p1)
    r = 0.5 * (r0 + r1)
    if r < 1e-16:
        return [p0, p1]
    a0 = math.atan2(p0[1], p0[0])
    a1 = math.atan2(p1[1], p1[0])
    da = a1 - a0
    while da <= -math.pi:
        da += 2.0 * math.pi
    while da > math.pi:
        da -= 2.0 * math.pi
    nn = max(int(n), 2)
    out = []
    for i in range(nn):
        a = a0 + da * i / (nn - 1)
        out.append((r * math.cos(a), r * math.sin(a)))
    out[0] = p0
    out[-1] = p1
    return out


def goldman_moc_profile(
    *,
    chord_m: float,
    beta1_deg: float,
    gamma: float,
    Mw1: float,
    Mw2: float | None,
    ML: float,
    MU: float,
    delv_deg: float = 1.0,
    n_arc: int = 36,
) -> tuple[list[tuple[float, float]], dict[str, Any]]:
    """Closed CCW polygon from D-4421 MOC. Raises ValueError if the net is illegal."""
    g = float(gamma)
    Mi = float(Mw1)
    Mo = float(Mw1 if Mw2 is None else Mw2)
    if Mi < 1.02 or Mo < 1.02:
        raise ValueError("Goldman MOC needs supersonic Mi, Mo")
    if not (ML <= Mi <= MU or ML <= Mo <= MU):
        # still allow ML < min(Mi,Mo) < MU which is the real constraint
        pass
    if ML > min(Mi, Mo) + 1e-9:
        raise ValueError("ML must be ≤ min(Mi, Mo) (ν_L ≤ ν_i, ν_o)")
    if MU < max(Mi, Mo) - 1e-9:
        raise ValueError("MU must be ≥ max(Mi, Mo)")
    vi = prandtl_meyer(Mi, g)
    vo = prandtl_meyer(Mo, g)
    vl = prandtl_meyer(ML, g)
    vu = prandtl_meyer(MU, g)
    bi = math.radians(beta1_deg)
    bo = math.radians(beta_out_deg(Mi, Mo, beta1_deg, g))
    # circular turning (6–7). Must not go the wrong way.
    a_li = bi - (vi - vl)
    a_lo = bo + (vo - vl)
    a_ui = bi - (vu - vi)
    a_uo = bo + (vu - vo)
    if a_li <= 0.0 and a_lo >= 0.0:
        raise ValueError("lower circular turning illegal: ν_L too small vs βi")
    if a_ui <= 0.0 and a_uo >= 0.0:
        raise ValueError("upper circular turning illegal: ν_U too large vs βi")

    delv = math.radians(max(float(delv_deg), 0.2))
    vnl = vi - vl
    vol = vo - vl
    kmaxn = max(int(round(vnl / delv)), 0)
    kmaxo = max(int(round(vol / delv)), 0)
    kmn = max(kmaxn, kmaxo)
    vui = vu - vi
    vuo = vu - vo
    jmaxn = max(int(round(vui / delv)), 0)
    jmaxo = max(int(round(vuo / delv)), 0)
    jmn = max(jmaxn, jmaxo)

    ms_l = m_star(ML, g)
    ms_u = m_star(MU, g)
    R_l = 1.0 / max(ms_l, 1e-9)
    R_u = 1.0 / max(ms_u, 1e-9)

    nu_long_l = max(vi, vo)
    low_unrot = _lower_transition(
        R_l=R_l, nu_long=nu_long_l, nu_l=vl, kmax=max(kmn, 1), delv=delv, gamma=g
    )
    nu_short_u = min(vi, vo)
    up_unrot = _upper_transition(
        R_u=R_u, nu_short=nu_short_u, nu_u=vu, jmax=max(jmn, 1), delv=delv, gamma=g
    )

    # Paper (20): rotate trans by α_i / α_o. Unrotated trans extends to −x
    # from the (0, R*) junction. Rotating that same curve by α_o folds the
    # outlet back over the circular arc (self-intersect). Reflect x first so
    # the outlet extends away from the vortex, then rotate by α_o. For
    # impulse that is the y-axis mirror of the inlet.
    low_in = _rotate_pts(low_unrot, a_li)
    up_in = _rotate_pts(up_unrot, a_ui)
    low_out = _rotate_pts(_flip_x(low_unrot), a_lo)
    up_out = _rotate_pts(_flip_x(up_unrot), a_uo)

    junc_l_in = low_in[0]
    junc_l_out = low_out[0]
    junc_u_in = up_in[0]
    junc_u_out = up_out[0]

    low_circ = _arc_about_origin(junc_l_in, junc_l_out, n_arc)
    up_circ = _arc_about_origin(junc_u_in, junc_u_out, n_arc)

    # Surface from inlet end → circular → outlet end.
    # unrotated trans: index 0 = circular junction, last = uniform end.
    def surface(trans_in, circ, trans_out):
        inlet = list(reversed(trans_in[1:])) + [trans_in[0]]
        outlet = trans_out[1:]
        return inlet[:-1] + circ + outlet

    lower = surface(low_in, low_circ, low_out)
    upper = surface(up_in, up_circ, up_out)

    le_low, te_low = lower[0], lower[-1]
    le_up, te_up = upper[0], upper[-1]
    # Fig. 1 BD: short straights along βi / βo to a common station, then the
    # thickness face (perpendicular to flow). Not a one-point [te_up] jump.
    bd_extra = 0.05 * abs(R_l - R_u)
    le_l2, le_u2 = _bd_station(le_low, le_up, bi, upstream=True, extra=bd_extra)
    te_l2, te_u2 = _bd_station(te_low, te_up, bo, upstream=False, extra=bd_extra)
    poly = (
        [le_l2, le_low]
        + lower[1:-1]
        + [te_low, te_l2, te_u2, te_up]
        + list(reversed(upper[1:-1]))
        + [le_up, le_u2]
    )
    poly = _close(_dedup_pts(poly))

    # Put LE→TE along +x, LE at origin, scale R* net to chord_m.
    lx, ly = le_l2
    tx, ty = te_l2
    ang = math.atan2(ty - ly, tx - lx)
    rot = [_rot(x - lx, y - ly, -ang) for x, y in poly]
    L = math.hypot(tx - lx, ty - ly)
    if L < 1e-12:
        L = 1.0
    s = float(chord_m) / L
    poly = [(x * s, y * s) for x, y in rot]
    poly = _close(poly)
    if _signed_area(poly) < 0:
        poly = _close(list(reversed(poly[:-1])))

    meta = {
        "method": "goldman_tn_d4421_moc",
        "predicted": True,
        "ntrs_id": "19680009151",
        "ntrs_id_analysis": "19680010807",
        "gamma": g,
        "Mw1": Mi,
        "Mw2": Mo,
        "ML": float(ML),
        "MU": float(MU),
        "nu_i_deg": math.degrees(vi),
        "nu_o_deg": math.degrees(vo),
        "nu_L_deg": math.degrees(vl),
        "nu_U_deg": math.degrees(vu),
        "beta1_deg": float(beta1_deg),
        "beta2_deg": math.degrees(bo),
        "alpha_l_i_deg": math.degrees(a_li),
        "alpha_u_i_deg": math.degrees(a_ui),
        "Rstar_L": R_l,
        "Rstar_U": R_u,
        "bd_extra_Rstar": bd_extra,
        "section_Rstar": L,
        "note": (
            "TN D-4421 MOC transitions + vortex circular arcs + BD straights. "
            "βi is an input, not unique-incidence output. PREDICTED. "
            "Off the 1968 1.5–5 chart if Mw1<1.5 — run the net, do not read "
            "thickness/solidity off those plots."
        ),
    }
    return poly, meta



def goldman_five_piece_profile(
    *,
    chord_m: float,
    beta1_deg: float,
    beta2_deg: float,
    lin_m: float = 0.0015,
    lout_m: float = 0.0015,
    r_tr_m: float = 0.0020,
    r_main_m: float = 0.0040,
    t_m: float = 0.0014,
    psi_tr_deg: float = 25.0,
    n: int = 24,
    beta1_metal_deg: float | None = None,
    beta2_metal_deg: float | None = None,
) -> tuple[list[tuple[float, float]], dict[str, Any]]:
    """Straight / transition / main vortex arc / transition / straight. PREDICTED.

    Geometric first-cut of D-4421 fig. 1 (BD + transition + vortex). Not the
    MOC net. Chord is a length. t is wall thickness. No U/ṁ/OF.
    """
    from .geometry import polygon_self_intersects, polygon_signed_area

    c = max(float(chord_m), 1e-9)
    b1 = math.radians(float(beta1_metal_deg if beta1_metal_deg is not None else beta1_deg))
    b2 = math.radians(float(beta2_metal_deg if beta2_metal_deg is not None else beta2_deg))
    turn = b2 - b1
    while turn <= -math.pi:
        turn += 2.0 * math.pi
    while turn > math.pi:
        turn -= 2.0 * math.pi
    sgn = 1.0 if turn >= 0.0 else -1.0
    abs_turn = abs(turn)
    psi_tr = min(max(math.radians(float(psi_tr_deg)), 0.05), 0.40 * abs_turn)
    psi_main = max(abs_turn - 2.0 * psi_tr, 0.05)
    L_in = max(float(lin_m), 0.0)
    L_out = max(float(lout_m), 0.0)
    R_tr = max(float(r_tr_m), 1e-6)
    R_main = max(float(r_main_m), 1e-6)
    thick = max(float(t_m), 1e-6)
    nn = max(int(n), 8)

    def add_straight(pts, p, h, L, nseg):
        if L <= 1e-12:
            return p, h
        ex = p[0] + L * math.cos(h)
        ey = p[1] + L * math.sin(h)
        for i in range(1, nseg + 1):
            t = i / nseg
            pts.append((p[0] + t * (ex - p[0]), p[1] + t * (ey - p[1])))
        return (ex, ey), h

    def add_arc(pts, p, h, R, dpsi, nseg):
        # sgn>0 CCW, center to the left of heading
        left = h + sgn * math.pi / 2.0
        C = (p[0] + R * math.cos(left), p[1] + R * math.sin(left))
        a0 = math.atan2(p[1] - C[1], p[0] - C[0])
        a1 = a0 + sgn * dpsi
        for i in range(1, nseg + 1):
            a = a0 + (a1 - a0) * i / nseg
            pts.append((C[0] + R * math.cos(a), C[1] + R * math.sin(a)))
        h2 = h + sgn * dpsi
        p2 = pts[-1]
        return p2, h2

    cl: list[tuple[float, float]] = [(0.0, 0.0)]
    p, h = (0.0, 0.0), b1
    p, h = add_straight(cl, p, h, L_in, 8)
    p, h = add_arc(cl, p, h, R_tr, psi_tr, nn)
    p, h = add_arc(cl, p, h, R_main, psi_main, nn * 2)
    p, h = add_arc(cl, p, h, R_tr, psi_tr, nn)
    p, h = add_straight(cl, p, h, L_out, 8)

    # Rotate/scale so LE→TE is (0,0)→(c,0)
    x0, y0 = cl[0]
    x1, y1 = cl[-1]
    ang = math.atan2(y1 - y0, x1 - x0)
    ca, sa = math.cos(-ang), math.sin(-ang)
    rot = []
    for x, y in cl:
        dx, dy = x - x0, y - y0
        rot.append((dx * ca - dy * sa, dx * sa + dy * ca))
    # Do not scale to chord — L_in, R, t stay millimetres. Chord is not a zoom.
    cam = rot

    # Constant wall thickness on the BD straights. Caps only at LE/TE.
    def tan_norm(i):
        if i == 0:
            dx, dy = cam[1][0] - cam[0][0], cam[1][1] - cam[0][1]
        elif i == len(cam) - 1:
            dx, dy = cam[-1][0] - cam[-2][0], cam[-1][1] - cam[-2][1]
        else:
            dx, dy = cam[i + 1][0] - cam[i - 1][0], cam[i + 1][1] - cam[i - 1][1]
        Lh = math.hypot(dx, dy) or 1.0
        tx, ty = dx / Lh, dy / Lh
        return tx, ty, -ty, tx  # left normal

    half = 0.5 * thick
    ss_pts, ps_pts = [], []
    for i, (x, y) in enumerate(cam):
        tx, ty, nx, ny = tan_norm(i)
        ss_pts.append((x + nx * half, y + ny * half))
        ps_pts.append((x - nx * half, y - ny * half))

    def _cap(center, p_from, p_to, through):
        a0 = math.atan2(p_from[1] - center[1], p_from[0] - center[0])
        a1 = math.atan2(p_to[1] - center[1], p_to[0] - center[0])
        da = a1 - a0
        while da <= -math.pi:
            da += 2.0 * math.pi
        while da > math.pi:
            da -= 2.0 * math.pi
        amid = math.atan2(through[1] - center[1], through[0] - center[0])
        # pick the semicircle that passes `through` (outward tip)
        dmid = amid - a0
        while dmid <= -math.pi:
            dmid += 2.0 * math.pi
        while dmid > math.pi:
            dmid -= 2.0 * math.pi
        if da * dmid < 0:
            da = da - 2.0 * math.pi if da > 0 else da + 2.0 * math.pi
        ncap = 8
        pts = []
        for i in range(1, ncap):
            a = a0 + da * i / ncap
            pts.append((center[0] + half * math.cos(a), center[1] + half * math.sin(a)))
        return pts

    tx0, ty0, _, _ = tan_norm(0)
    le_tip = (cam[0][0] - tx0 * half, cam[0][1] - ty0 * half)
    tx1, ty1, _, _ = tan_norm(len(cam) - 1)
    te_tip = (cam[-1][0] + tx1 * half, cam[-1][1] + ty1 * half)
    # SS LE→TE, TE cap SS→PS, PS TE→LE, LE cap PS→SS.
    te_cap = _cap(cam[-1], ss_pts[-1], ps_pts[-1], te_tip)
    le_cap = _cap(cam[0], ps_pts[0], ss_pts[0], le_tip)
    poly = ss_pts + te_cap + [ps_pts[-1]] + list(reversed(ps_pts[1:-1])) + [ps_pts[0]] + le_cap + [ss_pts[0]]
    poly = _close(poly)
    if polygon_signed_area(poly) < 0:
        poly = _close(list(reversed(poly[:-1])))
    if polygon_self_intersects(poly):
        raise ValueError("five-piece Goldman wall self-intersects")
    return poly, {
        "method": "goldman_five_piece_bd",
        "predicted": True,
        "lin_m": L_in,
        "lout_m": L_out,
        "r_tr_m": R_tr,
        "r_main_m": R_main,
        "t_m": thick,
        "psi_tr_deg": math.degrees(psi_tr),
        "beta1_metal_deg": math.degrees(b1),
        "beta2_metal_deg": math.degrees(b2),
        "note": "Straight + transition + main vortex arc + transition + straight. Parallel BD (const t). PREDICTED. Not MOC.",
    }


def goldman_impulse_profile(
    *,
    chord_m: float,
    beta1_deg: float,
    beta2_deg: float,
    gamma: float,
    ML: float,
    MU: float,
    Mw1: float | None = None,
    Mw2: float | None = None,
    n_arc: int = 48,
    use_moc: bool = True,
    lin_m: float | None = None,
    lout_m: float | None = None,
    r_tr_m: float | None = None,
    r_main_m: float | None = None,
    t_m: float | None = None,
    psi_tr_deg: float | None = None,
    beta1_metal_deg: float | None = None,
    beta2_metal_deg: float | None = None,
    upper_sagitta_c: float = 0.50,
    lower_sagitta_c: float = 0.22,
    hu: float | None = None,
    hl: float | None = None,
    le_fillet_r_c: float = 0.008,
    te_fillet_r_c: float = 0.008,
    upper_sagitta_m: float | None = None,
    lower_sagitta_m: float | None = None,
    le_fillet_r_m: float | None = None,
    te_fillet_r_m: float | None = None,
    pitch_m: float | None = None,
) -> tuple[list[tuple[float, float]], dict[str, Any]]:
    """Live metal: TN D-4421 MOC. No five-piece, no dual-arc cup.

    use_moc is accepted and ignored (always MOC). Illegal ML/MU/Mw1 raise.
    U, ṁ, OF never enter. Extra kwargs kept so old callers do not explode.
    """
    from .geometry import polygon_self_intersects

    mw1 = float(Mw1) if Mw1 not in (None, "") else 1.404
    mw2 = float(Mw2) if Mw2 not in (None, "") else mw1
    poly, meta = goldman_moc_profile(
        chord_m=chord_m,
        beta1_deg=beta1_deg,
        gamma=gamma,
        Mw1=mw1,
        Mw2=mw2,
        ML=ML,
        MU=MU,
        n_arc=max(24, n_arc // 2),
    )
    if polygon_self_intersects(poly):
        raise ValueError(
            "Goldman MOC polygon self-intersects — D-4421 net is not meshed metal"
        )
    return poly, meta



def _vortex_arcs_fallback(
    *,
    chord_m: float,
    beta1_deg: float,
    beta2_deg: float,
    gamma: float,
    ML: float,
    MU: float,
    n_arc: int,
) -> tuple[list[tuple[float, float]], dict[str, Any]]:
    c = max(float(chord_m), 1e-9)
    turn = abs(float(beta1_deg) - float(beta2_deg)) * math.pi / 180.0
    turn = min(max(turn, 0.35), 2.8)
    g = float(gamma)
    ms_l = m_star(ML, g)
    ms_u = m_star(MU, g)
    r_inner, r_outer = 1.0 / max(ms_u, 1e-9), 1.0 / max(ms_l, 1e-9)
    if r_inner > r_outer:
        r_inner, r_outer = r_outer, r_inner
    r_mean = 0.5 * (r_inner + r_outer)
    R_target = c / (2.0 * max(math.sin(turn / 2.0), 1e-6))
    scale = R_target / max(r_mean, 1e-12)
    r_i, r_o = r_inner * scale, r_outer * scale
    r_m = 0.5 * (r_i + r_o)
    a0, a1 = -0.5 * turn, 0.5 * turn
    n = max(int(n_arc), 12)

    def arc(r: float, aa: float, bb: float, nn: int) -> list[tuple[float, float]]:
        return [
            (r * math.cos(aa + (bb - aa) * i / (nn - 1)), r * math.sin(aa + (bb - aa) * i / (nn - 1)))
            for i in range(nn)
        ]

    d_cusp = 0.08 * turn
    le = (r_m * math.cos(a0 - d_cusp), r_m * math.sin(a0 - d_cusp))
    te = (r_m * math.cos(a1 + d_cusp), r_m * math.sin(a1 + d_cusp))
    poly = [le] + arc(r_o, a0, a1, n) + [te] + arc(r_i, a1, a0, n)
    lx, ly = le
    ang = math.atan2(te[1] - ly, te[0] - lx)
    rot = [_rot(x - lx, y - ly, -ang) for x, y in poly]
    L = math.hypot(rot[n + 1][0], rot[n + 1][1]) or c
    s = c / L
    poly = _close([(x * s, y * s) for x, y in rot])
    if _signed_area(poly) < 0:
        poly = _close(list(reversed(poly[:-1])))
    return poly, {
        "method": "vortex_arcs_fallback_not_moc",
        "predicted": True,
        "ML": float(ML),
        "MU": float(MU),
        "gamma": g,
        "beta1_metal_deg": beta1_deg,
        "beta2_metal_deg": beta2_deg,
        "note": "MOC net illegal; concentric vortex arcs only. PREDICTED.",
    }


def le_te_tangents_deg(poly: list[tuple[float, float]]) -> tuple[float, float]:
    pts = poly[:-1] if poly and poly[0] == poly[-1] else list(poly)
    if len(pts) < 4:
        return 0.0, 0.0
    b1 = math.degrees(math.atan2(pts[2][1] - pts[1][1], pts[2][0] - pts[1][0]))
    imax = max(range(len(pts)), key=lambda i: pts[i][0])
    xt, yt = pts[imax]
    xa, ya = pts[(imax - 1) % len(pts)]
    xb, yb = pts[(imax + 1) % len(pts)]
    b2 = math.degrees(math.atan2(yt - ya, xt - xa))
    b2b = math.degrees(math.atan2(yb - yt, xb - xt))
    if abs(b2b) > abs(b2):
        b2 = b2b
    return b1, b2


def goldman_from_job(job: dict[str, Any]) -> dict[str, Any]:
    g = job.get("geometry") or {}
    gas = job.get("gas") or {}
    raw = g.get("goldman") if isinstance(g.get("goldman"), dict) else {}
    gamma = float(gas.get("gamma") or raw.get("gamma") or 1.3)
    w1 = float(gas.get("w1_m_s") or 0.0)
    t1 = float(gas.get("t1_k") or 0.0)
    r = float(gas.get("r_specific_j_kg_k") or 0.0)
    a1 = math.sqrt(gamma * r * t1) if (gamma > 0 and r > 0 and t1 > 0) else 0.0
    mw1 = (w1 / a1) if a1 > 0 else float(raw.get("Mw1") or 0.0)
    ml = float(raw.get("ML") or g.get("ML") or 1.15)
    mu = float(raw.get("MU") or g.get("MU") or 1.55)
    mw2 = raw.get("Mw2")
    mw2_held = bool(raw.get("mw2_held"))
    if mw2 in (None, "") or not mw2_held:
        mw2 = mw1
    else:
        mw2 = float(mw2)
    scoping = bool(raw.get("scoping") or g.get("goldman_scoping"))
    return {
        "Mw1": mw1,
        "Mw2": mw2,
        "ML": ml,
        "MU": mu,
        "gamma": gamma,
        "te": str(raw.get("te") or "cusp"),
        "le": str(raw.get("le") or "cusp"),
        "delta_star": bool(raw.get("delta_star")),
        "scoping": scoping,
        "mw2_held": mw2_held,
        "method": "goldman_tn_d4421_moc",
        "use_moc": True,
        "predicted": True,
        "ntrs_id": "19680009151",
    }
