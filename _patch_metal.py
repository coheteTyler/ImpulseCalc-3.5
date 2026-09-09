from pathlib import Path

g = Path("/workspace/ImpulseCalc3/impulsecalc3/geometry.py")
t = g.read_text()

old_cip = '''def center_in_pitch(
    poly: list[tuple[float, float]],
    pitch_m: float,
) -> tuple[list[tuple[float, float]], float]:
    """Translate profile in y so it sits in the middle of one pitch strip.

    If the C is taller than s, clip to ±s/2 (cyclic cut). That is adjacent-blade
    intersection, not a σ bug. Never flatten stems.
    """
    ys = [p[1] for p in poly]
    y_mid = 0.5 * (min(ys) + max(ys))
    shift = -y_mid
    out = [(p[0], p[1] + shift) for p in poly]
    ys2 = [p[1] for p in out]
    half = 0.5 * float(pitch_m)
    # 0.25 mm cyclic fluid. Tighter slits invert pyramids in the O-collar.
    margin = 0.25e-3
    if max(ys2) >= half - margin or min(ys2) <= -half + margin:
        out = clip_poly_to_y_strip(out, -half + margin, half - margin)
        if len(out) < 8 or abs(polygon_signed_area(out)) < 1e-12:
            raise ValueError(
                f"pitch clip emptied the metal (y-span [{min(ys2):.5f}, {max(ys2):.5f}] vs "
                f"±{half:.5f} m). s too small for a closed section."
            )
        if polygon_signed_area(out) < 0:
            out = list(reversed(out))
            out = _closed(out)
    return out, shift
'''
# file may still have 40e-6 comment after 0.25e-3 replace
import re
pat = re.compile(
    r'def center_in_pitch\([\s\S]*?return out, shift\n',
    re.M,
)
new_cip = '''def center_in_pitch(
    poly: list[tuple[float, float]],
    pitch_m: float,
) -> tuple[list[tuple[float, float]], float]:
    """Translate profile in y so it sits in the middle of one pitch strip.

    Do not shear the C. Overlap (yspan > s) is a packing fail, not a mill cut.
    """
    ys = [p[1] for p in poly]
    y_mid = 0.5 * (min(ys) + max(ys))
    shift = -y_mid
    out = [(p[0], p[1] + shift) for p in poly]
    ys2 = [p[1] for p in out]
    half = 0.5 * float(pitch_m)
    margin = 0.25e-3
    if max(ys2) >= half - margin or min(ys2) <= -half + margin:
        yspan = max(ys2) - min(ys2)
        raise ValueError(
            f"s={pitch_m*1e3:.2f} mm < metal y-span {yspan*1e3:.2f} mm; "
            f"min s ≈ {(yspan + 2*margin)*1e3:.2f} mm for a 0.25 mm cyclic gap. "
            "Not a license to flatten the outer arc."
        )
    return out, shift
'''
nt, nsub = pat.subn(new_cip, t, count=1)
if nsub != 1:
    raise SystemExit(f"center_in_pitch replace failed {nsub}")
t = nt

old_note = '''        notes.append(
            f"s={p0*1e3:.2f} mm < y-span {yspan*1e3:.2f} mm: metal clipped ±{cut*1e3:.2f} mm "
            "at the cyclics (adjacent overlap). Passage is the remaining strip. Stems kept. Not σ."
        )'''
new_note = '''        notes.append(
            f"s={p0*1e3:.2f} mm < y-span {yspan*1e3:.2f} mm — adjacent blades overlap. "
            f"Min non-overlap s ≈ {yspan*1e3 + 0.50:.2f} mm. Outer C kept. Not σ."
        )'''
if old_note not in t:
    raise SystemExit("fit note missing")
t = t.replace(old_note, new_note, 1)

old_cap = '''    def _cap(u, l, h):
        mx, my = 0.5 * (u[0] + l[0]), 0.5 * (u[1] + l[1])
        r = 0.5 * math.hypot(u[0] - l[0], u[1] - l[1])
        if r < 1e-9:
            return []
        a0 = math.atan2(u[1] - my, u[0] - mx)
        a1 = math.atan2(l[1] - my, l[0] - mx)
        da = _wrap(a1 - a0)
        mid = a0 + 0.5 * da
        through = (mx + r * math.cos(mid), my + r * math.sin(mid))
        prefer = (mx + r * math.cos(h), my + r * math.sin(h))
        if (through[0] - mx) * (prefer[0] - mx) + (through[1] - my) * (prefer[1] - my) < 0:
            da = da - 2.0 * math.pi if da > 0 else da + 2.0 * math.pi
        ncap = 12
        return [
            (mx + r * math.cos(a0 + da * i / ncap), my + r * math.sin(a0 + da * i / ncap))
            for i in range(1, ncap)
        ]
'''
new_cap = '''    def _cap(u, l, h, r_fillet: float):
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
'''
if old_cap not in t:
    raise SystemExit("old _cap missing")
t = t.replace(old_cap, new_cap, 1)

# resolve r_le / r_te and pass to _cap
old_tips = '''    u_tip_le = u_left[-1] if u_left else pu_le
    l_tip_le = l_left[-1] if l_left else pl_le
    u_tip_te = u_right[-1] if u_right else pu_te
    l_tip_te = l_right[-1] if l_right else pl_te
    te_cap = _cap(u_tip_te, l_tip_te, h_stem_te)
    le_cap = _cap(l_tip_le, u_tip_le, h_stem_le)
'''
new_tips = '''    if le_fillet_r_m not in (None, ""):
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
'''
if old_tips not in t:
    raise SystemExit("tips block missing")
t = t.replace(old_tips, new_tips, 1)

g.write_text(t)
print("geometry patched")
