from pathlib import Path
p = Path("/workspace/ImpulseCalc3/impulsecalc3/passage.py")
t = p.read_text()
a = '    o_s = _pos_block(_layers(south_m, south_o, n_rad, stretch), "o_south")\n    o_n = _pos_block(_layers(north_m, north_o, n_rad, stretch), "o_north")\n'
b = '    o_s = _layers(south_m, south_o, n_rad, stretch)\n    o_n = _layers(north_m, north_o, n_rad, stretch)\n'
if a not in t:
    raise SystemExit('o pos not found')
p.write_text(t.replace(a, b, 1))
print('ok')
