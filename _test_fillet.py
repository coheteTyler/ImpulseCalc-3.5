import sys
sys.path.insert(0, "/workspace/ImpulseCalc3")
from impulsecalc3.geometry import pointed_bucket_profile, polygon_self_intersects, polygon_signed_area

def tip_width(poly, want_le):
    xs = [p[0] for p in poly]
    if want_le:
        i = min(range(len(poly)), key=lambda k: xs[k])
    else:
        i = max(range(len(poly)), key=lambda k: xs[k])
    return poly[i]

kw = dict(chord_m=0.01, beta1_metal_deg=72, beta2_metal_deg=-72,
          lin_m=0.00425, lout_m=0.00425, r_tr_m=0.0035, r_main_m=0.0048,
          t_m=0.0014, psi_tr_deg=0, upper_sagitta_m=0.005, lower_sagitta_m=0.0022)
for name, le, te in [("point", 0.0, 0.0), ("le04_te0", 0.0004, 0.0), ("le07_te07", 0.0007, 0.0007)]:
    p = pointed_bucket_profile(**kw, le_fillet_r_m=le, te_fillet_r_m=te)
    ys = [q[1] for q in p]
    print(name, "n", len(p), "area", round(polygon_signed_area(p), 8),
          "yspan_mm", round((max(ys)-min(ys))*1e3, 3),
          "self", polygon_self_intersects(p),
          "xmin", round(min(q[0] for q in p)*1e3, 3),
          "xmax", round(max(q[0] for q in p)*1e3, 3),
          "le_pt", [round(x*1e3,3) for x in tip_width(p, True)],
          "te_pt", [round(x*1e3,3) for x in tip_width(p, False)])
