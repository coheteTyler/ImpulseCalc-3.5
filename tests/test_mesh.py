"""Body-fitted mesh patch names: blade0/1/2, cyclic, empty frontAndBack."""

from pathlib import Path

from impulsecalc3.case import write_case
from impulsecalc3.geometry import BladeSpec
from impulsecalc3.job import domain_x, load_job
from impulsecalc3.meanline import compute_meanline
from impulsecalc3.times import compute_times


def test_mesh_patch_names(tmp_path: Path):
    job = load_job()
    ml = compute_meanline(job)
    g, gas = job["geometry"], job["gas"]
    xin, xout = domain_x(job)
    times = compute_times(
        chord_m=g["chord_m"], w1_m_s=gas["w1_m_s"], gamma=gas["gamma"],
        r_specific=gas["r_specific_j_kg_k"], t1_k=gas["t1_k"],
        x_in_m=xin, x_out_m=xout, n_chords_min=job["cfd"]["n_chords_min"],
    )
    spec = BladeSpec(
        chord_m=g["chord_m"], beta1_metal_deg=ml.beta1_metal_deg,
        beta2_metal_deg=ml.beta2_metal_deg, thickness_c=g["thickness_c"],
        le_radius_c=g["le_radius_c"], te_radius_c=g["te_radius_c"],
        n_points=int(g["n_profile_points"]),
    )
    mesh, _ = write_case(tmp_path / "case", job, ml, times, spec)
    btxt = (tmp_path / "case" / "constant" / "polyMesh" / "boundary").read_text()
    for name in ("blade0", "blade1", "blade2", "inlet", "outlet", "bottom", "top", "frontAndBack"):
        assert f"{name}" in btxt
        assert name in mesh.patches
        assert mesh.patches[name] > 0
    assert "type            wall" in btxt
    assert "type            cyclic" in btxt
    assert "type            empty" in btxt
    assert mesh.min_area_2d > 0
    notes = " ".join(mesh.check_notes)
    assert "NOT" in notes and "stair" in notes.lower()
    assert mesh.mesh_kind == "body_fitted_OH"
