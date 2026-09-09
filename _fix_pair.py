from pathlib import Path
p = Path("/workspace/ImpulseCalc3/impulsecalc3/passage.py")
t = p.read_text()
fn = '''
def _y_at_x(chain: np.ndarray, x: float) -> float:
    xs = chain[:, 0]
    if x <= float(xs.min()):
        i = int(np.argmin(xs))
        return float(chain[i, 1])
    if x >= float(xs.max()):
        i = int(np.argmax(xs))
        return float(chain[i, 1])
    for i in range(len(chain) - 1):
        x0, x1 = float(chain[i, 0]), float(chain[i + 1, 0])
        if (x0 - x) * (x1 - x) <= 0.0:
            if abs(x1 - x0) < 1e-16:
                return float(chain[i, 1])
            tt = (x - x0) / (x1 - x0)
            return float(chain[i, 1] + tt * (chain[i + 1, 1] - chain[i, 1]))
    return float(chain[int(np.argmin(np.abs(xs - x))), 1])


def _pair_by_x(south: np.ndarray, north: np.ndarray, n_st: int) -> tuple[np.ndarray, np.ndarray]:
    x0 = max(float(south[:, 0].min()), float(north[:, 0].min()))
    x1 = min(float(south[:, 0].max()), float(north[:, 0].max()))
    xs = np.linspace(x0, x1, n_st + 1)
    s = np.column_stack([xs, np.array([_y_at_x(south, x) for x in xs])])
    n = np.column_stack([xs, np.array([_y_at_x(north, x) for x in xs])])
    return s, n


'''
t = t.replace("def offset_open(", fn + "def offset_open(", 1)
old = """    n_st = max(int(n_stream), 24)
    south_m = _resample(south, n_st)
    north_m = _resample(north, n_st)
"""
new = """    n_st = max(int(n_stream), 24)
    south_m, north_m = _pair_by_x(south, north, n_st)
"""
if old not in t:
    raise SystemExit("pair block missing")
t = t.replace(old, new, 1)
old = """    s_head = _lin((x_in, float(south_o[0, 1])), south_o[0], n_in)
    n_head = _lin((x_in, float(north_o[0, 1])), north_o[0], n_in)
"""
new = """    s_head = _lin((x_in, float(south_o[0, 1])), south_o[0], n_in)
    n_head = s_head.copy()
    n_head[:, 1] = s_head[:, 1] + float(pitch)
"""
if old not in t:
    raise SystemExit("head missing")
t = t.replace(old, new, 1)
old = """    s_tail = _lin(south_o[-1], (x_out, float(south_o[-1, 1])), n_out)
    n_tail = _lin(north_o[-1], (x_out, float(north_o[-1, 1])), n_out)
"""
new = """    s_tail = _lin(south_o[-1], (x_out, float(south_o[-1, 1])), n_out)
    n_tail = s_tail.copy()
    n_tail[:, 1] = s_tail[:, 1] + float(pitch)
"""
if old not in t:
    raise SystemExit("tail missing")
t = t.replace(old, new, 1)
# pos_block the O layers
old = """    o_s = _layers(south_m, south_o, n_rad, stretch)
    o_n = _layers(north_m, north_o, n_rad, stretch)
"""
new = """    o_s = _pos_block(_layers(south_m, south_o, n_rad, stretch), "o_south")
    o_n = _pos_block(_layers(north_m, north_o, n_rad, stretch), "o_north")
"""
if old not in t:
    raise SystemExit("layers missing")
p.write_text(t.replace(old, new, 1))
print("paired")
