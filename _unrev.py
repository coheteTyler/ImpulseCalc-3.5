from pathlib import Path
p = Path("/workspace/ImpulseCalc3/impulsecalc3/mesh.py")
t = p.read_text()
t = t.replace(
'''        for fverts, ow, _c in buckets[name]:
            fv = list(reversed(fverts)) if name == "top" else fverts
            all_faces.append(fv)
            owners.append(ow)
''',
'''        for fverts, ow, _c in buckets[name]:
            all_faces.append(fverts)
            owners.append(ow)
''')
p.write_text(t)
print('reverted reverse')
