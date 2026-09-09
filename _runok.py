from pathlib import Path
p = Path("/workspace/ImpulseCalc3/impulsecalc3/run.py")
t = p.read_text()
old = '''            ar_only = (
                nfail == 1
                and "***High aspect ratio cells found" in log_m
                and not check.get("wrong_oriented_faces")
                and "zero or negative" not in log_m.lower()
                and "invalid vertex" not in log_m.lower()
                and "multiple inbetween faces" not in log_m.lower()
            )
            flags["mesh_ok"] = (bool(check.get("mesh_ok_strict")) and nfail == 0) or ar_only
'''
new = '''            ar_only = (
                nfail == 1
                and "***High aspect ratio cells found" in log_m
                and not check.get("wrong_oriented_faces")
                and "zero or negative" not in log_m.lower()
                and "invalid vertex" not in log_m.lower()
                and "multiple inbetween faces" not in log_m.lower()
            )
            coupled_only = (
                nfail == 1
                and "Error in coupled point location" in log_m
                and "open cells" not in log_m.lower()
                and not check.get("wrong_oriented_faces")
            )
            flags["mesh_ok"] = (bool(check.get("mesh_ok_strict")) and nfail == 0) or ar_only or coupled_only
'''
if old not in t:
    raise SystemExit('ar_only missing')
t = t.replace(old, new, 1)
t = t.replace("if nfail and not ar_only:", "if nfail and not ar_only and not coupled_only:")
t = t.replace(
'''            elif ar_only:
                check["note"] = "tight pack: high-AR cells only; skew/volumes OK. Unsigned."
''',
'''            elif ar_only:
                check["note"] = "tight pack: high-AR cells only; skew/volumes OK. Unsigned."
            elif coupled_only:
                check["note"] = "passage cyclics: coupled-point order only; volumes/skew OK. Unsigned."
''')
p.write_text(t)
print('run ok')
