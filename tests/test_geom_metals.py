"""Other metals: circular-arc Marlin foil still builds; dual-arc bucket; profile_points; open fails."""

from pathlib import Path

import pytest

from impulsecalc3.case import write_case
from impulsecalc3.geometry import (
    closed_profile,
    impulse_bucket_profile,
    polygon_self_intersects,
    polygon_signed_area,
    profile_from_job,
    profile_from_points,
    spec_from_job,
)
from impulsecalc3.job import DEFAULT_PATH, domain_x, load_job, pitch_m
from impulsecalc3.meanline import compute_meanline
from impulsecalc3.times import compute_times

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"


def _write(tmp: Path, job_path: Path):
    job = load_job(job_path)
    spec = spec_from_job(job)
    ml = compute_meanline(job)
    spec.beta1_metal_deg = ml.beta1_metal_deg
    spec.beta2_metal_deg = ml.beta2_metal_deg
    g, gas = job["geometry"], job["gas"]
    xin, xout = domain_x(job)
    times = compute_times(
        chord_m=g["chord_m"],
        w1_m_s=gas["w1_m_s"],
        gamma=gas["gamma"],
        r_specific=gas["r_specific_j_kg_k"],
        t1_k=gas["t1_k"],
        x_in_m=xin,
        x_out_m=xout,
        n_chords_min=job["cfd"]["n_chords_min"],
    )
    case = tmp / job["name"]
    mesh, t_end = write_case(case, job, ml, times, spec)
    return job, mesh, t_end, case


def test_default_job_is_still_marlin():
    assert DEFAULT_PATH.name == "marlin_v2_rotor.json"
    job = load_job()
    assert job["name"] == "marlin_v2_rotor"
    assert job.get("geometry_test") is False
    assert not job["geometry"].get("profile_points")
    assert job["geometry"].get("profile_family") != "impulse_bucket"


def test_circular_arc_marlin_still_builds(tmp_path: Path):
    job, mesh, t_end, case = _write(tmp_path, CONFIGS / "marlin_v2_rotor.json")
    assert mesh.n_cells > 500
    assert mesh.mesh_kind == "body_fitted_OH"
    assert mesh.min_area_2d > 0
    notes = " ".join(mesh.check_notes).lower()
    assert "stair" in notes
    btxt = (case / "constant" / "polyMesh" / "boundary").read_text()
    for name in ("blade0", "blade1", "blade2", "inlet", "outlet", "bottom", "top"):
        assert name in btxt
    assert "type            cyclic" in btxt
    poly = profile_from_job(job)
    assert poly[0] == poly[-1]
    assert polygon_signed_area(poly) > 0


def test_impulse_bucket_builds_or_fails_loud(tmp_path: Path):
    """Dual-arc U-bucket. A loud mesh fail is a finding — do not special-case Marlin."""
    path = CONFIGS / "geom_impulse_bucket.json"
    job = load_job(path)
    assert job["geometry_test"] is True
    assert job["name"] != "marlin_v2_rotor"
    assert "GEOMETRY TEST" in job["article"]
    g = job["geometry"]
    assert g["profile_family"] == "impulse_bucket"
    assert g["upper_sagitta_c"] >= 0.45
    assert g["lower_sagitta_c"] < g["upper_sagitta_c"] - 0.05
    assert not g.get("profile_points")
    # table envelope is still the unsigned V2 numbers — photo is not the signed table
    assert g["n_blades_machine"] == 25
    assert abs(g["solidity"] - 1.13688) < 1e-6
    foil = closed_profile(spec_from_job(load_job()))  # live Marlin 12% foil, not this job's t/c
    cup = profile_from_job(job)
    assert polygon_signed_area(cup) > 1.5 * polygon_signed_area(foil)
    try:
        job, mesh, t_end, case = _write(tmp_path, path)
    except (RuntimeError, ValueError) as exc:
        # Fail loud: cup could not mesh. That is the finding.
        assert "fold" in str(exc).lower() or "pitch" in str(exc).lower() or "area" in str(exc).lower()
        return
    assert mesh.n_cells > 500
    assert mesh.mesh_kind == "body_fitted_OH"
    assert mesh.min_area_2d > 0
    notes = " ".join(mesh.check_notes)
    assert "C-shape" not in notes
    assert "NOT" in notes and "stair" in notes.lower()
    btxt = (case / "constant" / "polyMesh" / "boundary").read_text()
    assert "type            wall" in btxt
    assert "type            cyclic" in btxt


def test_profile_points_closed_path_builds(tmp_path: Path):
    job, mesh, t_end, case = _write(tmp_path, CONFIGS / "geom_points.json")
    pts = job["geometry"]["profile_points"]
    assert pts[0] == pts[-1]
    poly = profile_from_job(job)
    assert poly[0] == poly[-1]
    assert polygon_signed_area(poly) > 0
    circ = closed_profile(spec_from_job(job))
    assert poly != circ
    assert mesh.n_cells > 500
    assert mesh.mesh_kind == "body_fitted_OH"
    assert mesh.min_area_2d > 0
    notes = " ".join(mesh.check_notes).lower()
    assert "stair" in notes
    assert "type            cyclic" in (case / "constant" / "polyMesh" / "boundary").read_text()


def test_open_polyline_fails():
    open_poly = [(0.0, 0.0), (0.01, 0.0), (0.01, 0.002), (0.0, 0.002)]
    with pytest.raises(ValueError, match="open polyline"):
        profile_from_points(open_poly)


def test_zero_area_fails():
    with pytest.raises(ValueError, match="zero area"):
        profile_from_points([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (0.0, 0.0)])


def test_self_intersecting_fails():
    bowtie = [(0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)]
    with pytest.raises(ValueError, match="self-intersect"):
        profile_from_points(bowtie)


def test_bucket_tight_pitch_builds_or_fails_loud(tmp_path: Path):
    """Tighter pitch on the cup stresses cyclic / metal-in-pitch. Loud fail is a finding."""
    path = CONFIGS / "geom_bucket_tight_pitch.json"
    try:
        job, mesh, t_end, case = _write(tmp_path, path)
    except (RuntimeError, ValueError) as exc:
        assert "pitch" in str(exc).lower() or "cyclic" in str(exc).lower() or "fold" in str(exc).lower()
        return
    assert job["geometry"]["n_blades_machine"] > 25
    assert job["geometry"]["profile_family"] == "impulse_bucket"
    assert mesh.n_cells > 500
    assert mesh.patches["bottom"] == mesh.patches["top"]
    assert mesh.min_area_2d > 0


def test_bucket_is_a_u_not_a_thin_foil():
    cup = impulse_bucket_profile(
        chord_m=0.01, upper_sagitta_c=0.50, lower_sagitta_c=0.22,
        le_fillet_r_c=0.04, te_fillet_r_c=0.04, n_points=80, pitch_m=0.008796,
    )
    assert cup[0] == cup[-1]
    assert polygon_signed_area(cup) > 0
    ys = [p[1] for p in cup]
    # Deep U: outer sagitta ~0.5c
    assert max(ys) - min(ys) > 0.35 * 0.01


def test_bucket_absolute_mm_chord_is_not_a_zoom():
    kw = dict(
        upper_sagitta_m=0.005, lower_sagitta_m=0.0022,
        le_fillet_r_m=8e-5, te_fillet_r_m=8e-5, n_points=80,
    )
    p1 = impulse_bucket_profile(chord_m=0.01, **kw)
    p2 = impulse_bucket_profile(chord_m=0.02, **kw)
    assert p1[0] == p1[-1] and p2[0] == p2[-1]
    y1 = max(p[1] for p in p1) - min(p[1] for p in p1)
    y2 = max(p[1] for p in p2) - min(p[1] for p in p2)
    assert abs(y1 - y2) < 0.12 * y1  # wall-band height ~constant
    x1 = max(p[0] for p in p1) - min(p[0] for p in p1)
    x2 = max(p[0] for p in p2) - min(p[0] for p in p2)
    assert x2 > 1.5 * x1
    s = 2.0
    scaled_ymax = s * max(p[1] for p in p1)
    assert abs(max(p[1] for p in p2) - scaled_ymax) > 1.5e-3  # not profile * c2/c1


def test_default_knobs_outline_is_dual_arc_round_fillets():
    from impulsecalc3.preview import knobs_to_job
    job = knobs_to_job({"W1": 950, "c": 0.01, "hu_mm": 5.0, "hl_mm": 2.2, "le_mm": 0.4, "te_mm": 0.4})
    assert job["geometry"]["profile_family"] == "impulse_bucket"
    poly = profile_from_job(job)
    assert poly[0] == poly[-1]
    assert polygon_signed_area(poly) > 0
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    # Round ends: not a vertical flat at min-x / max-x (many points on a x=const cut)
    xmin, xmax = min(xs), max(xs)
    n_left = sum(1 for x in xs if abs(x - xmin) < 1e-6)
    n_right = sum(1 for x in xs if abs(x - xmax) < 1e-6)
    assert n_left <= 3
    assert n_right <= 3
    assert len(poly) >= 40


def test_lin_mm_switches_to_pointed_round_caps():
    from impulsecalc3.preview import knobs_to_job
    a = knobs_to_job({"c": 0.01, "hu_mm": 5.0, "lin_mm": 0.0, "lout_mm": 0.0, "le_mm": 0.4})
    b = knobs_to_job({"c": 0.01, "hu_mm": 5.0, "lin_mm": 1.8, "lout_mm": 1.8, "le_mm": 0.4, "t_mm": 1.4, "psi_tr_deg": 20})
    pa, pb = profile_from_job(a), profile_from_job(b)
    assert pa != pb
    assert polygon_self_intersects(pb) is False
    assert polygon_signed_area(pb) > 0
    # Stemmed tips extend past the L=0 dual-arc envelope (left of x=0 and/or down).
    assert min(x for x, _ in pb) < min(x for x, _ in pa) - 0.0003


def test_lout_is_a_stem_not_a_chord_cut():
    """L grows converging straights to tip T along the C-tip path, not a chord cut."""
    from impulsecalc3.geometry import pointed_bucket_profile
    kw = dict(chord_m=0.01, beta1_metal_deg=72, beta2_metal_deg=-72,
              lout_m=0.0, r_tr_m=0.002, r_main_m=0.004, t_m=0.0014, psi_tr_deg=20,
              le_fillet_r_m=0.0, te_fillet_r_m=0.0,
              upper_sagitta_m=0.005, lower_sagitta_m=0.0022)
    a = pointed_bucket_profile(lin_m=0.001, **kw)
    b = pointed_bucket_profile(lin_m=0.006, **kw)
    assert polygon_self_intersects(a) is False
    assert polygon_self_intersects(b) is False
    # Longer Lin pushes the LE tip further out along the stem heading.
    assert min(p[0] for p in b) < min(p[0] for p in a) - 0.002
    tip = min(b, key=lambda p: p[0])
    assert tip[0] < -0.002
    # Straights meet at a single tip (not a flat bar): few points share xmin.
    xmin = min(p[0] for p in b)
    assert sum(1 for p in b if abs(p[0] - xmin) < 1e-7) <= 3


def test_fillet_fattens_from_pointed_tip():
    """r=0 sharp at dual-arc tip; larger r moves the tip up the bisector."""
    from impulsecalc3.geometry import pointed_bucket_profile
    kw = dict(chord_m=0.01, beta1_metal_deg=65, beta2_metal_deg=-65,
              lin_m=0.0, lout_m=0.0, upper_sagitta_m=0.005, lower_sagitta_m=0.0022)
    sharp = pointed_bucket_profile(**kw, le_fillet_r_m=0.0, te_fillet_r_m=0.0)
    fat = pointed_bucket_profile(**kw, le_fillet_r_m=0.0004, te_fillet_r_m=0.0004)
    assert polygon_self_intersects(sharp) is False
    assert polygon_self_intersects(fat) is False
    tip_s = min(sharp, key=lambda p: p[0])
    tip_f = min(fat, key=lambda p: p[0])
    assert abs(tip_s[0]) < 1e-9 and abs(tip_s[1]) < 1e-9
    assert tip_f[1] > tip_s[1] + 0.0003
    assert polygon_signed_area(fat) < polygon_signed_area(sharp)
