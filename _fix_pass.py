from pathlib import Path
p = Path("/workspace/ImpulseCalc3/impulsecalc3/passage.py")
t = p.read_text()
t = t.replace(
    "from .geometry import polygon_signed_area, split_ps_ss\nfrom .mesh import _lin, _pos_block, _stretch, min_cell_area_2d_rect, tfi_block\n",
    "from .geometry import split_ps_ss\n",
)
old = "def _resample("
new = """def _lin(p0, p1, n_seg: int):
    p0 = np.asarray(p0, dtype=float).reshape(2)
    p1 = np.asarray(p1, dtype=float).reshape(2)
    tt = np.linspace(0.0, 1.0, max(int(n_seg), 1) + 1)[:, None]
    return (1.0 - tt) * p0[None, :] + tt * p1[None, :]


def _stretch(j: int, n: int, r: float) -> float:
    if n <= 0:
        return 0.0
    if abs(r - 1.0) < 1e-9:
        return j / n
    return (r ** j - 1.0) / (r ** n - 1.0)


def _resample("""
if old not in t:
    raise SystemExit("no _resample")
t = t.replace(old, new, 1)
old = "    notes: list[str] = []\n    ps, ss, _, _ = split_ps_ss(poly)\n"
new = "    from .mesh import _pos_block, min_cell_area_2d_rect, tfi_block\n\n    notes: list[str] = []\n    ps, ss, _, _ = split_ps_ss(poly)\n"
if old not in t:
    raise SystemExit("build start missing")
p.write_text(t.replace(old, new, 1))
print("fixed")
