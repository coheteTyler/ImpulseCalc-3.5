from pathlib import Path

from impulsecalc3.sample import parse_raw_surfaces


def test_parses_p_column_not_t(tmp_path: Path):
    p = tmp_path / "p_blade0Wall.raw"
    p.write_text(
        "# x y z T p\n"
        "0.001 0.0 0.0005 1237.7 550000\n"
        "0.002 0.0 0.0005 1240.0 551000\n",
        encoding="utf-8",
    )
    rows = parse_raw_surfaces(p)
    assert len(rows) == 2
    assert abs(rows[0]["p"] - 550000) < 1
    assert abs(rows[0]["p"] - 1237.7) > 1000  # would fail if T-as-p


def test_wall_assert_low_p(tmp_path: Path):
    from impulsecalc3.sample import load_wall_pressure
    import impulsecalc3.sample as sm

    raw = tmp_path / "postProcessing" / "surfaces_blade0" / "0" / "blade0Wall.raw"
    raw.parent.mkdir(parents=True)
    # p ~ 1 kPa while p1 is 5.5 bar
    raw.write_text("# x y z p\n0 0 0.0005 1200\n0.01 0 0.0005 1300\n", encoding="utf-8")
    # other blades too
    for k in (1, 2):
        q = tmp_path / "postProcessing" / f"surfaces_blade{k}" / "0" / f"blade{k}Wall.raw"
        q.parent.mkdir(parents=True)
        q.write_text("# x y z p\n0 0 0.0005 1200\n0.01 0 0.0005 1300\n", encoding="utf-8")
    r = load_wall_pressure(tmp_path, p1_pa=5.5e5, chord_m=0.01)
    assert r["success"] is False
    assert r["on_wall"] is False


def test_numeric_time_sort_not_lexicographic(tmp_path):
    """6.57e-06 is BEFORE 5.26e-05 numerically; lex sort gets this backwards."""
    from impulsecalc3.sample import find_wall_raw, load_wall_pressure

    def dump(tname: str, pval: float, blade: int = 0) -> None:
        d = tmp_path / "postProcessing" / f"surfaces_blade{blade}" / tname
        d.mkdir(parents=True, exist_ok=True)
        (d / f"p_blade{blade}Wall.raw").write_text(
            f"# x y z p\n0.0 0.0 0.0005 {pval}\n0.01 0.0 0.0005 {pval}\n",
            encoding="utf-8",
        )

    dump("6.5789474e-06", 800000.0)
    dump("5.2631579e-05", 704000.0)
    for k in (1, 2):
        dump("6.5789474e-06", 800000.0, k)
        dump("5.2631579e-05", 704000.0, k)
    (tmp_path / "0").mkdir()
    (tmp_path / "0" / "p").write_text(
        "FoamFile { class volScalarField; }\ninternalField uniform 550000;\n",
        encoding="utf-8",
    )
    latest = find_wall_raw(tmp_path, 0)
    assert latest is not None
    assert "5.2631579e-05" in str(latest)
    r = load_wall_pressure(tmp_path, p1_pa=5.5e5, chord_m=0.01)
    assert abs(r["t0_mean_p"] - 550000) < 1
    assert abs(r["blades"][0]["mean_p"] - 704000) < 1
    assert r["on_wall"] is True
