from pathlib import Path
p = Path("/workspace/ImpulseCalc3/impulsecalc3/mesh.py")
t = p.read_text()
old = '''        for fverts, ow, _c in buckets[name]:
            all_faces.append(fverts)
            owners.append(ow)
'''
new = '''        for fverts, ow, _c in buckets[name]:
            fv = list(reversed(fverts)) if name == "top" else fverts
            all_faces.append(fv)
            owners.append(ow)
'''
if old not in t:
    raise SystemExit('append faces missing')
p.write_text(t.replace(old, new, 1))
print('reversed top')
