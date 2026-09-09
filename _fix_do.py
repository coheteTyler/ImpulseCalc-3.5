from pathlib import Path
p = Path("/workspace/ImpulseCalc3/impulsecalc3/mesh.py")
t = p.read_text()
old = '''    d_o = min(0.00045, 0.22 * max(clearance_y, 2e-6), 0.06 * spec.chord_m)
    fam = str((job.get("geometry") or {}).get("profile_family") or "")
'''
new = '''    if passage is None:
        d_o = min(0.00045, 0.22 * max(clearance_y, 2e-6), 0.06 * spec.chord_m)
    fam = str((job.get("geometry") or {}).get("profile_family") or "")
'''
if old not in t:
    raise SystemExit('d_o assign missing')
p.write_text(t.replace(old, new, 1))
print('d_o guarded')
