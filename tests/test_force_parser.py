from pathlib import Path

from impulsecalc3.forces import parse_force_dat, plateau_flags


def test_parse_v2412_ten_column(tmp_path: Path):
    p = tmp_path / "force.dat"
    p.write_text(
        "# Time  tot_x tot_y tot_z  p_x p_y p_z  v_x v_y v_z\n"
        "1e-5  1.0  2.0  0  0.9  1.9  0  0.1  0.1  0\n"
        "2e-5  1.1  2.2  0  1.0  2.1  0  0.1  0.1  0\n",
        encoding="utf-8",
    )
    rows = parse_force_dat(p)
    assert len(rows) == 2
    assert rows[-1]["Fy"] == 2.2
    assert rows[-1]["Fx"] == 1.1
    assert "Ft" not in rows[-1]  # raw FO columns; scaling happens later


def test_plateau_detects_climbing():
    t = [1e-6 * i for i in range(1, 11)]
    ft = [10.0 * i for i in range(1, 11)]  # still rising
    d = plateau_flags(t, ft, t_chord=5e-6, rel_tol=0.04)
    assert d["climbing"] is True
    assert d["plateau"] is False


def test_plateau_flat_window():
    t = [1e-5 * i for i in range(1, 20)]
    ft = [42.0] * 19
    d = plateau_flags(t, ft, t_chord=2e-5)
    assert d["plateau"] is True
    assert d["climbing"] is False


def test_no_n_blades_clone_in_parser():
    """Parser returns one patch's force.dat; scaling must not divide by 3."""
    import inspect
    from impulsecalc3 import forces as m

    src = inspect.getsource(m.load_blade_forces)
    assert "/ n_blades" not in src
    assert "/ 3" not in src.replace("/ 3.0", "")
