"""Wall raw parser picks a p column, never T-as-p."""

from pathlib import Path

from impulsecalc3.sample import parse_raw_surfaces


def test_parse_p_not_t(tmp_path: Path):
    p = tmp_path / "blade0Wall.raw"
    p.write_text(
        "# x y z T p\n"
        "0.001 0.0 0.0005 1100 550000\n"
        "0.002 0.001 0.0005 1090 548000\n"
        "0.003 -0.001 0.0005 1110 552000\n"
    )
    rows = parse_raw_surfaces(p)
    assert len(rows) == 3
    assert abs(rows[0]["p"] - 550000) < 1e-6
    assert rows[0]["p"] != 1100


def test_fallback_xyz_p(tmp_path: Path):
    p = tmp_path / "blade0Wall.raw"
    p.write_text("0.0 0.0 0.0 5.5e5\n0.01 0.0 0.0 5.4e5\n")
    rows = parse_raw_surfaces(p)
    assert len(rows) == 2
    assert rows[0]["p"] == 5.5e5
