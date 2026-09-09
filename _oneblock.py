from pathlib import Path

# add_struct wall_hi
p = Path("/workspace/ImpulseCalc3/impulsecalc3/mesh.py")
t = p.read_text()
t = t.replace(
    "    def add_struct(pts: np.ndarray, dy: float, periodic_i: bool, wall_patch: str | None = None) -> None:",
    "    def add_struct(pts: np.ndarray, dy: float, periodic_i: bool, wall_patch: str | None = None, wall_hi: str | None = None) -> None:",
)
t = t.replace(
    "                add_hex(verts, wall_patch=wall_patch if (j == 0 and wall_patch) else None)",
    "                wp = wall_patch if (j == 0 and wall_patch) else (wall_hi if (j == n_jc - 1 and wall_hi) else None)\n                add_hex(verts, wall_patch=wp)",
)
t = t.replace(
    """    if passage is not None:
        add_struct(passage.o_south, 0.0, False, wall_patch="blade0")
        add_struct(passage.o_north, 0.0, False, wall_patch="blade1")
        for hb in h_blocks:
            add_struct(hb, 0.0, False)
""",
    """    if passage is not None:
        add_struct(passage.h_core, 0.0, False, wall_patch="blade0", wall_hi="blade1")
        add_struct(passage.h_inlet, 0.0, False)
        add_struct(passage.h_outlet, 0.0, False)
""",
)
p.write_text(t)

# passage.py one conformal core
p = Path("/workspace/ImpulseCalc3/impulsecalc3/passage.py")
t = p.read_text()
old = '''    toward = 0.5 * (south_m[n_st // 2] + north_m[n_st // 2])
    south_o = offset_open(south_m, d_use, toward)
    north_o = offset_open(north_m, d_use, toward)
    # collars must not cross
    if float(np.linalg.norm(north_o - south_o, axis=1).min()) <= 1e-8:
        d_use *= 0.5
        south_o = offset_open(south_m, d_use, toward)
        north_o = offset_open(north_m, d_use, toward)

    o_s = _layers(south_m, south_o, n_rad, stretch)
    o_n = _layers(north_m, north_o, n_rad, stretch)

    n_sp = max(int(n_span), 6)
    west = _lin(south_o[0], north_o[0], n_sp)
    east = _lin(south_o[-1], north_o[-1], n_sp)
    h_core = tfi_block(south_o, north_o, west, east)

    s_in = _extend_x(south_o, x_in, south_o[0, 0], n_in, 2)
    n_inw = _extend_x(north_o, x_in, north_o[0, 0], n_in, 2)
    # only the inlet head (x_in → metal LE of offset)
    s_head = _lin((x_in, float(south_o[0, 1])), south_o[0], n_in)
    n_head = s_head.copy()
    n_head[:, 1] = s_head[:, 1] + float(pitch)
    h_inlet = tfi_block(s_head, n_head, _lin(s_head[0], n_head[0], n_sp), _lin(s_head[-1], n_head[-1], n_sp))
    s_tail = _lin(south_o[-1], (x_out, float(south_o[-1, 1])), n_out)
    n_tail = s_tail.copy()
    n_tail[:, 1] = s_tail[:, 1] + float(pitch)
    h_outlet = tfi_block(s_tail, n_tail, _lin(s_tail[0], n_tail[0], n_sp), _lin(s_tail[-1], n_tail[-1], n_sp))
'''
new = '''    n_eta = max(int(n_rad) + int(n_span), 10)
    h_core = np.zeros((south_m.shape[0], n_eta + 1, 2), dtype=float)
    for j in range(n_eta + 1):
        # cosine clustering at both walls (O-like first cell without a second block)
        t = 0.5 - 0.5 * math.cos(math.pi * j / n_eta)
        h_core[:, j, :] = (1.0 - t) * south_m + t * north_m
    d_use = float(np.mean(np.linalg.norm(h_core[:, 1, :] - h_core[:, 0, :], axis=1)))

    n_sp = n_eta
    s_head = _lin((x_in, float(south_m[0, 1])), south_m[0], n_in)
    n_head = s_head.copy()
    n_head[:, 1] = s_head[:, 1] + float(pitch)
    # inlet east MUST be the core west (same nodes)
    east_in = h_core[0, :, :]
    west_in = _lin(s_head[0], n_head[0], n_sp)
    if east_in.shape[0] != west_in.shape[0]:
        west_in = _lin(s_head[0], n_head[0], east_in.shape[0] - 1)
        n_sp = east_in.shape[0] - 1
        s_head = _lin((x_in, float(south_m[0, 1])), south_m[0], n_in)
        n_head = s_head.copy()
        n_head[:, 1] = s_head[:, 1] + float(pitch)
    h_inlet = tfi_block(s_head, n_head, west_in, east_in)

    s_tail = _lin(south_m[-1], (x_out, float(south_m[-1, 1])), n_out)
    n_tail = s_tail.copy()
    n_tail[:, 1] = s_tail[:, 1] + float(pitch)
    west_out = h_core[-1, :, :]
    east_out = _lin(s_tail[-1], n_tail[-1], west_out.shape[0] - 1)
    h_outlet = tfi_block(s_tail, n_tail, west_out, east_out)

    o_s = h_core[:, :2, :]
    o_n = h_core[:, -2:, :]
'''
if old not in t:
    raise SystemExit('core block missing')
t = t.replace(old, new, 1)
t = t.replace(
    "    first = float(np.mean(np.linalg.norm(o_s[:, 1, :] - o_s[:, 0, :], axis=1)))",
    "    first = float(np.mean(np.linalg.norm(h_core[:, 1, :] - h_core[:, 0, :], axis=1)))",
)
p.write_text(t)
print('one-block')
