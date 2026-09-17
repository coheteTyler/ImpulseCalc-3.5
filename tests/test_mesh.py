"""Body-fitted 1-pitch mesh: blade0, cyclic period=pitch, empty frontAndBack."""

from pathlib import Path

from impulsecalc3.case import write_case
from impulsecalc3.geometry import BladeSpec
from impulsecalc3.job import domain_x, load_job, pitch_m
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
    # Freeze B: force body_fitted 1-pitch even if JSON says hoh/cassette
    job.setdefault("cfd", {})["mesh"] = "body_fitted_OH"
    mesh, _ = write_case(tmp_path / "case", job, ml, times, spec)
    btxt = (tmp_path / "case" / "constant" / "polyMesh" / "boundary").read_text()
    for name in ("blade0", "inlet", "outlet", "bottom", "top", "frontAndBack"):
        assert name in btxt
        assert name in mesh.patches
        assert mesh.patches[name] > 0
    assert "blade1" not in mesh.patches or mesh.patches.get("blade1", 0) == 0
    assert "type            wall" in btxt
    assert "type            cyclic" in btxt
    assert "type            empty" in btxt
    assert mesh.min_area_2d > 0
    notes = " ".join(mesh.check_notes)
    assert "NOT" in notes and "stair" in notes.lower()
    assert mesh.mesh_kind == "body_fitted_OH"
    assert mesh.patches["bottom"] == mesh.patches["top"]
    # separationVector period = 1×pitch
    assert f"(0 {pitch_m(job):.12g} 0)" in btxt or f"(0 {float(pitch_m(job)):.12g} 0)" in btxt
    assert abs(mesh.y_max - mesh.y_min - pitch_m(job)) < 1e-9


def test_face_classification_leftover_zero_tiny(tmp_path: Path):
    """Assert every boundary face lands on inlet/outlet/cyclic/blade/empty before promote."""
    job = load_job()
    job.setdefault("cfd", {})["mesh"] = "body_fitted_OH"
    # tiny ring counts keep the write fast
    job["cfd"]["n_around"] = 40
    job["cfd"]["n_radial"] = 6
    job["cfd"]["n_inlet"] = 6
    job["cfd"]["n_outlet"] = 8
    job["cfd"]["n_cyclic"] = 8
    job["cfd"]["n_pitch_fill"] = 4
    ml = compute_meanline(job)
    g, gas = job["geometry"], job["gas"]
    xin, xout = domain_x(job)
    times = compute_times(
        chord_m=g["chord_m"], w1_m_s=gas["w1_m_s"], gamma=gas["gamma"],
        r_specific=gas["r_specific_j_kg_k"], t1_k=gas["t1_k"],
        x_in_m=xin, x_out_m=xout, n_chords_min=1.0,
    )
    spec = BladeSpec(
        chord_m=g["chord_m"], beta1_metal_deg=ml.beta1_metal_deg,
        beta2_metal_deg=ml.beta2_metal_deg, thickness_c=g["thickness_c"],
        le_radius_c=g["le_radius_c"], te_radius_c=g["te_radius_c"],
        n_points=int(g["n_profile_points"]),
    )
    mesh, _ = write_case(tmp_path / "tiny", job, ml, times, spec)
    assert mesh.patches["bottom"] == mesh.patches["top"]
    assert mesh.patches["inlet"] > 0 and mesh.patches["outlet"] > 0
    assert mesh.patches["blade0"] > 0
    assert "blade1" not in mesh.patches
    # leftover would have raised in write_polymesh


def test_dump_xs_1c_packing_and_no_te_fence_helpers():
    import numpy as np
    from impulsecalc3.mesh import dump_xs_1c, te_plane_fence_hit, _dump_block_from_west

    x_te, c, dx = 0.01, 0.01, 2e-4
    xs = dump_xs_1c(x_te, c, dx, n_near=10, stretch_max=1.25, L_dump_c=1.0)
    assert abs(xs[0] - x_te) < 1e-15
    assert xs[-1] >= x_te + 1.0 * c - 1e-12
    assert (xs[1] - xs[0]) > 0
    # first 0.4c clustered: more nodes in near band than a uniform 1c split would put there
    n_near = int(np.sum(xs <= x_te + 0.4 * c + 1e-12)) - 1
    assert 8 <= n_near <= 12
    # curved west dump is not a vertical fence
    west = np.column_stack([np.linspace(x_te - 0.001, x_te, 20), np.linspace(-0.004, 0.004, 20)])
    dump = _dump_block_from_west(west, xs)
    assert dump.shape[0] == len(xs)
    # synthetic full-pitch vertical column → fence hit
    ys = np.linspace(-0.004, 0.004, 30)
    cliff = np.zeros((2, len(ys), 2))
    cliff[0, :, 0] = x_te
    cliff[0, :, 1] = ys
    cliff[1, :, 0] = x_te + 0.002
    cliff[1, :, 1] = ys
    assert te_plane_fence_hit([cliff], x_te=x_te, chord=c, y_bot=-0.004, y_top=0.004)
    # curved dump should not trip the fence
    assert not te_plane_fence_hit([dump], x_te=x_te, chord=c, y_bot=-0.004, y_top=0.004)
