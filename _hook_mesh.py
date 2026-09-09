from pathlib import Path
p = Path("/workspace/ImpulseCalc3/impulsecalc3/mesh.py")
t = p.read_text()
t = t.replace(
    "                add_hex(verts, wall_patch=wall_patch if (j == 0 and periodic_i) else None)",
    "                add_hex(verts, wall_patch=wall_patch if (j == 0 and wall_patch) else None)",
)
old = '''    poly0 = list(poly) if poly is not None else profile_from_job(job, spec)
    if polygon_signed_area(poly0) <= 0:
        raise RuntimeError("profile is not a CCW metal interior (zero or negative area)")
    poly0, y_shift = center_in_pitch(poly0, pitch)
'''
new = '''    poly0 = list(poly) if poly is not None else profile_from_job(job, spec)
    if polygon_signed_area(poly0) <= 0:
        raise RuntimeError("profile is not a CCW metal interior (zero or negative area)")
    ys0 = [p[1] for p in poly0]
    yspan = max(ys0) - min(ys0)
    fam0 = str((job.get("geometry") or {}).get("profile_family") or "")
    use_passage = (yspan + 5e-4 > pitch) or fam0 in (
        "goldman_impulse", "goldman", "goldman_vortex",
    )
    passage = None
    if use_passage:
        from .passage import build_passage_oh
        passage = build_passage_oh(
            poly0,
            pitch=pitch,
            x_in=x_in,
            x_out=x_out,
            n_in=n_in,
            n_out=n_out_x,
            n_stream=max(n_cyc * 4, 64),
            n_span=max(n_fill, 8),
            n_rad=n_rad,
            stretch=stretch,
            d_o=min(0.00045, 0.06 * spec.chord_m),
        )
        y_shift = 0.0
        n_blades = 2
        ogrid = passage.o_south
        h_blocks = [passage.h_core, passage.h_inlet, passage.h_outlet]
        a2 = passage.min_area_2d
        first_cell = passage.first_cell_m
        n_i = ogrid.shape[0]
        oh_notes = list(passage.notes)
        d_o = passage.d_o
        y_bot, y_top = passage.y_min, passage.y_max
    else:
        poly0, y_shift = center_in_pitch(poly0, pitch)
'''
if old not in t:
    raise SystemExit('poly0 block missing')
t = t.replace(old, new, 1)
# indent the rest of the rectangle path under else - currently the next lines use poly0 after center.
# After my replace, clearance_y still runs for passage too. Need to skip rectangle path.
old = '''        y_shift = 0.0
        n_blades = 2
        ogrid = passage.o_south
        h_blocks = [passage.h_core, passage.h_inlet, passage.h_outlet]
        a2 = passage.min_area_2d
        first_cell = passage.first_cell_m
        n_i = ogrid.shape[0]
        oh_notes = list(passage.notes)
        d_o = passage.d_o
        y_bot, y_top = passage.y_min, passage.y_max
    else:
        poly0, y_shift = center_in_pitch(poly0, pitch)
    xs = [p[0] for p in poly0]
'''
new = '''        y_shift = 0.0
        n_blades = 2
        ogrid = passage.o_south
        h_blocks = [passage.h_core, passage.h_inlet, passage.h_outlet]
        a2 = passage.min_area_2d
        first_cell = passage.first_cell_m
        n_i = ogrid.shape[0]
        oh_notes = list(passage.notes)
        d_o = passage.d_o
        y_bot, y_top = passage.y_min, passage.y_max
    if passage is None:
        poly0, y_shift = center_in_pitch(poly0, pitch)
    xs = [p[0] for p in poly0]
'''
if old not in t:
    raise SystemExit('else wrap missing')
t = t.replace(old, new, 1)
# wrap rectangle mesh build in `if passage is None`
old = '''    clearance_y = min(y_top - ymax, ymin - y_bot)
    if clearance_y <= 1e-9:
        raise RuntimeError("profile clearance to cyclic is non-positive after pitch clip")
'''
new = '''    clearance_y = min(y_top - ymax, ymin - y_bot) if passage is None else 1.0
    if passage is None and clearance_y <= 1e-9:
        raise RuntimeError("profile clearance to cyclic is non-positive after pitch clip")
'''
if old not in t:
    raise SystemExit('clearance missing')
t = t.replace(old, new, 1)
old = '''    if use_cavity:
'''
new = '''    if passage is None and use_cavity:
'''
if old not in t:
    raise SystemExit('use_cavity missing')
t = t.replace(old, new, 1)
old = '''    else:
        # Tight AABB around a wall-normal offset (not metal+d_o). Cartesian H-blocks
'''
new = '''    elif passage is None:
        # Tight AABB around a wall-normal offset (not metal+d_o). Cartesian H-blocks
'''
if old not in t:
    raise SystemExit('else aabb missing')
t = t.replace(old, new, 1)

old = '''    for k in range(n_blades):
        dy = k * pitch
        add_struct(ogrid, dy, True, wall_patch=f"blade{k}")
        for hb in h_blocks:
            add_struct(hb, dy, False)
'''
new = '''    if passage is not None:
        add_struct(passage.o_south, 0.0, False, wall_patch="blade0")
        add_struct(passage.o_north, 0.0, False, wall_patch="blade1")
        for hb in h_blocks:
            add_struct(hb, 0.0, False)
    else:
        for k in range(n_blades):
            dy = k * pitch
            add_struct(ogrid, dy, True, wall_patch=f"blade{k}")
            for hb in h_blocks:
                add_struct(hb, dy, False)
'''
if old not in t:
    raise SystemExit('assembly loop missing')
t = t.replace(old, new, 1)

old = '''    y_min = y_bot
    y_max = y_bot + n_blades * pitch
    blade_polys = [[(xy[0], xy[1] + k * pitch) for xy in poly0] for k in range(n_blades)]
'''
new = '''    if passage is not None:
        y_min, y_max = passage.y_min, passage.y_max
        blade_polys = [list(poly0), [(xy[0], xy[1] + pitch) for xy in poly0]]
    else:
        y_min = y_bot
        y_max = y_bot + n_blades * pitch
        blade_polys = [[(xy[0], xy[1] + k * pitch) for xy in poly0] for k in range(n_blades)]
'''
if old not in t:
    raise SystemExit('blade_polys missing')
t = t.replace(old, new, 1)

old = '''        elif abs(y - y_min) < 1e-7:
            buckets["bottom"].append((fverts, owner, c))
        elif abs(y - y_max) < 1e-7:
            buckets["top"].append((fverts, owner, c))
'''
new = '''        elif passage is not None and passage.cyclic and (
            abs(y - passage.o_south[0, 0, 1]) < 1e-6
            or abs(y - passage.o_south[-1, 0, 1]) < 1e-6
        ):
            buckets["bottom"].append((fverts, owner, c))
        elif passage is not None and passage.cyclic and (
            abs(y - (passage.o_south[0, 0, 1] + pitch)) < 1e-6
            or abs(y - (passage.o_south[-1, 0, 1] + pitch)) < 1e-6
        ):
            buckets["top"].append((fverts, owner, c))
        elif passage is None and abs(y - y_min) < 1e-7:
            buckets["bottom"].append((fverts, owner, c))
        elif passage is None and abs(y - y_max) < 1e-7:
            buckets["top"].append((fverts, owner, c))
'''
if old not in t:
    raise SystemExit('y class missing')
t = t.replace(old, new, 1)

old = '''    if len(buckets["bottom"]) != len(buckets["top"]):
        raise RuntimeError(
            f"cyclic face count mismatch bottom={len(buckets['bottom'])} top={len(buckets['top'])}"
        )
'''
new = '''    if buckets["bottom"] or buckets["top"]:
        if len(buckets["bottom"]) != len(buckets["top"]):
            raise RuntimeError(
                f"cyclic face count mismatch bottom={len(buckets['bottom'])} top={len(buckets['top'])}"
            )
'''
if old not in t:
    raise SystemExit('cyclic count missing')
t = t.replace(old, new, 1)

old = '''    patch_order = ["inlet", "outlet", "bottom", "top", "frontAndBack", *blade_names]
'''
new = '''    patch_order = ["inlet", "outlet"]
    if buckets["bottom"] or buckets["top"]:
        patch_order.extend(["bottom", "top"])
    patch_order.extend(["frontAndBack", *blade_names])
'''
if old not in t:
    raise SystemExit('patch_order missing')
t = t.replace(old, new, 1)

old = '''        mesh_kind="body_fitted_OH",
'''
new = '''        mesh_kind=("passage_OH" if passage is not None else "body_fitted_OH"),
'''
if old not in t:
    raise SystemExit('mesh_kind missing')
t = t.replace(old, new, 1)

old = '''        "mesh: body-fitted O-grid on the metal + Cartesian H-blocks to the pitch rectangle.",
        "NOT Cartesian subsetMesh stair-step.",
        f"{n_blades} blades, cyclic pitch {pitch:.6g} m, empty frontAndBack, slab {zth} m.",
'''
new = '''        (
            "mesh: Goldman/Katsanis passage O+H (SS0 vs PS0+s). Not a pitch rectangle."
            if passage is not None
            else "mesh: body-fitted O-grid on the metal + Cartesian H-blocks to the pitch rectangle."
        ),
        "NOT Cartesian subsetMesh stair-step.",
        f"{n_blades} blades, pitch {pitch:.6g} m, empty frontAndBack, slab {zth} m.",
'''
if old not in t:
    raise SystemExit('notes missing')
t = t.replace(old, new, 1)
p.write_text(t)
print('mesh hooked', t.count('use_passage'))
