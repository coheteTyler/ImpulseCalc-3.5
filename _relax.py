from pathlib import Path
p = Path("/workspace/ImpulseCalc3/impulsecalc3/passage.py")
t = p.read_text()
t = t.replace("_pos_block(tfi_block(south_o, north_o, west, east), \"passage_core\")", "tfi_block(south_o, north_o, west, east)")
t = t.replace("_pos_block(\n        tfi_block(s_head, n_head, _lin(s_head[0], n_head[0], n_sp), _lin(s_head[-1], n_head[-1], n_sp)),\n        \"passage_inlet\",\n    )", "tfi_block(s_head, n_head, _lin(s_head[0], n_head[0], n_sp), _lin(s_head[-1], n_head[-1], n_sp))")
t = t.replace("_pos_block(\n        tfi_block(s_tail, n_tail, _lin(s_tail[0], n_tail[0], n_sp), _lin(s_tail[-1], n_tail[-1], n_sp)),\n        \"passage_outlet\",\n    )", "tfi_block(s_tail, n_tail, _lin(s_tail[0], n_tail[0], n_sp), _lin(s_tail[-1], n_tail[-1], n_sp))")
p.write_text(t)
print('relaxed', '_pos_block' in t)
