from pathlib import Path
p = Path("/workspace/ImpulseCalc3/impulsecalc3/geometry.py")
t = p.read_text()
a = '''            f"s={p0*1e3:.2f} mm < y-span {yspan*1e3:.2f} mm — adjacent blades overlap. "
            f"Min non-overlap s ≈ {yspan*1e3 + 0.50:.2f} mm. Outer C kept. Not σ."'''
b = '''            f"s={p0*1e3:.2f} mm < y-span {yspan*1e3:.2f} mm: Mark III nesting. "
            "Passage O+H (SS0 vs PS0+s). Outer C kept. Not σ."'''
if a not in t:
    raise SystemExit('note missing')
p.write_text(t.replace(a, b, 1))
print('note ok')
