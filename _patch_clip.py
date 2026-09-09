from pathlib import Path
p = Path("/workspace/ImpulseCalc3/impulsecalc3/geometry.py")
t = p.read_text()
t = t.replace("    margin = 40e-6\n", "    margin = 0.25e-3\n", 1)
old = "        out = clip_poly_to_y_strip(out, -half + margin, half - margin)\n        if len(out) < 8"
new = "        out = clip_poly_to_y_strip(out, -half + margin, half - margin)\n        if len(out) < 8"
if "margin = 0.25e-3" not in t:
    raise SystemExit("margin replace failed")
p.write_text(t)
print("patched", p)
