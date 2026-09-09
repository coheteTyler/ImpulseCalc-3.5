from pathlib import Path
p = Path("/workspace/ImpulseCalc3/impulsecalc3/mesh.py")
t = p.read_text()
old = '''        elif abs(x - x_in) < 1e-7:
            buckets["inlet"].append((fverts, owner, c))
        elif abs(x - x_out) < 1e-7:
            buckets["outlet"].append((fverts, owner, c))
'''
new = '''        elif abs(x - x_in) < 2e-4:
            buckets["inlet"].append((fverts, owner, c))
        elif abs(x - x_out) < 2e-4:
            buckets["outlet"].append((fverts, owner, c))
'''
if old not in t:
    raise SystemExit('xy tol missing')
t = t.replace(old, new, 1)
old = '''        else:
            unclassified += 1
    if unclassified:
        raise RuntimeError(f"{unclassified} boundary faces not on wall/inlet/outlet/cyclic/empty")
'''
new = '''        else:
            if passage is not None:
                ymid = 0.5 * (passage.y_min + passage.y_max)
                name = "blade0" if y < ymid else "blade1"
                buckets[name].append((fverts, owner, c))
            else:
                unclassified += 1
    if unclassified:
        raise RuntimeError(f"{unclassified} boundary faces not on wall/inlet/outlet/cyclic/empty")
'''
if old not in t:
    raise SystemExit('else unclass missing')
p.write_text(t.replace(old, new, 1))
print('bc fallback')
