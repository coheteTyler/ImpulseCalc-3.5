from impulsecalc3.geometry import apply_metal_bounds, safe_profile_from_job
from impulsecalc3.preview import knobs_to_job, outline_from_knobs


def test_hu12_with_stems_clips_and_still_draws():
    job = knobs_to_job({"c": 0.01, "hu_mm": 12, "hl_mm": 2.2, "lin_mm": 3, "lout_mm": 3, "t_mm": 1.4})
    j2, notes, bounds = apply_metal_bounds(job)
    assert j2["geometry"]["upper_sagitta_m"] <= 0.0050001
    assert any("hu_mm" in n for n in notes)
    poly, more = safe_profile_from_job(j2)
    assert poly[0] == poly[-1]
    assert len(poly) > 20
    out = outline_from_knobs({"c": 0.01, "hu_mm": 12, "hl_mm": 2.2, "lin_mm": 3, "lout_mm": 3, "t_mm": 1.4, "W1": 950})
    assert out.get("ok") is True
    assert out.get("geom_bounds")["hu_mm"]["max"] <= 5.01
    assert out.get("poly_area_m2") > 0


def test_nominal_stems_no_warning():
    out = outline_from_knobs({"c": 0.01, "hu_mm": 5, "hl_mm": 2.2, "lin_mm": 3, "lout_mm": 3, "t_mm": 1.4, "r_main_mm": 6, "W1": 950})
    assert out.get("geom_warnings") == []
    assert out.get("ok") is True
