"""v2412 force.dat parser. Ft=Fy, Fd=Fx. No total/n_blades clone."""

from pathlib import Path

from impulsecalc3.forces import parse_force_dat, plateau_flags


def test_parse_force_dat(tmp_path: Path):
    p = tmp_path / "force.dat"
    p.write_text(
        "# Time total(x y z) pressure(x y z) viscous(x y z)\n"
        "1e-6 (0.1 0.2 0) (0.08 0.15 0) (0.02 0.05 0)\n"
        "2e-6 (0.11 0.21 0) (0.09 0.16 0) (0.02 0.05 0)\n"
    )
    rows = parse_force_dat(p)
    assert len(rows) == 2
    assert rows[0]["Fx"] == 0.1  # Fd
    assert rows[0]["Fy"] == 0.2  # Ft
    assert rows[0]["Fvx"] == 0.02


def test_plateau_climbing():
    t = [0.0, 1e-5, 2e-5, 3e-5, 4e-5, 5e-5]
    climbing = [i * 1.0 for i in range(6)]
    flags = plateau_flags(t, climbing, t_chord=1e-5)
    assert flags["climbing"] is True
    flat = [1.0] * 6
    flags2 = plateau_flags(t, flat, t_chord=1e-5)
    assert flags2["plateau"] is True
