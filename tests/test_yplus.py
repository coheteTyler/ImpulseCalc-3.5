"""After-solve y+ is the LAST time, never t=0, never a zeroed postProcess dump."""

from pathlib import Path

from viewer.ofio import parse_yplus


def _write_yp_field(path: Path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""FoamFile {{ class volScalarField; object yPlus; }}
internalField   uniform {value};
boundaryField
{{
    blade0 {{ type calculated; value uniform {value}; }}
}}
""",
        encoding="utf-8",
    )


def test_parse_yplus_last_time_dat_not_t0_or_zero_field(tmp_path: Path):
    (tmp_path / "0").mkdir()
    (tmp_path / "0" / "p").write_text("internalField uniform 550000;\n", encoding="utf-8")
    _write_yp_field(tmp_path / "0" / "yPlus", 0.0)
    last = tmp_path / "7.9827174e-05"
    last.mkdir()
    (last / "p").write_text("internalField uniform 550000;\n", encoding="utf-8")
    # postProcess trap: last-time volField all zeros
    _write_yp_field(last / "yPlus", 0.0)
    # sibling postProcess dat also zeros at last time
    zdir = tmp_path / "postProcessing" / "yPlus" / "0"
    zdir.mkdir(parents=True)
    (zdir / "yPlus.dat").write_text(
        "# Time patch min max average\n"
        "7.9827174e-05\tblade0\t0\t0\t0\n"
        "7.9827174e-05\tblade1\t0\t0\t0\n",
        encoding="utf-8",
    )
    # in-run FO table has t=0 and last time; last time is after-solve
    y1 = tmp_path / "postProcessing" / "yPlus1" / "0"
    y1.mkdir(parents=True)
    (y1 / "yPlus.dat").write_text(
        "# Time patch min max average\n"
        "0\tblade0\t1.0\t2.0\t1.5\n"
        "7.9827174e-05\tblade0\t0.701982\t24.47189\t12.093743\n"
        "7.9827174e-05\tblade1\t0.701985\t24.471892\t12.093739\n"
        "7.9827174e-05\tblade2\t0.701975\t24.469286\t12.093664\n",
        encoding="utf-8",
    )
    y = parse_yplus(tmp_path)
    assert y["time_s"] is not None
    assert abs(float(y["time_s"]) - 7.9827174e-05) / 7.9827174e-05 < 1e-6
    assert float(y["min"]) > 0.5
    assert float(y["max"]) > 20.0
    assert abs(float(y["average"]) - 12.0937) < 0.01
    assert "yPlus1" in str(y["source"])


def test_parse_yplus_log_fallback_skips_zero_hits(tmp_path: Path):
    (tmp_path / "7.98e-05").mkdir()
    (tmp_path / "7.98e-05" / "p").write_text("internalField uniform 1;\n", encoding="utf-8")
    log = """
yPlus yPlus1 write:
    writing field yPlus
    patch blade0 y+ : min = 0.701982, max = 24.47189, average = 12.093743
    patch blade1 y+ : min = 0.7019852, max = 24.471892, average = 12.093739
    patch blade2 y+ : min = 0.70197531, max = 24.469286, average = 12.093664
yPlus yPlus write:
    writing field yPlus
    patch blade0 y+ : min = 0, max = 0, average = 0
"""
    y = parse_yplus(tmp_path, log)
    assert float(y["average"]) > 10.0
    assert float(y["max"]) > 20.0
    assert y["time_s"] is not None and float(y["time_s"]) > 0
