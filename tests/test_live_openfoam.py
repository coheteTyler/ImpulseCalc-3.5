"""Live ESI v2412: write mesh, checkMesh, solver starts. Design run is start.sh."""

from pathlib import Path

import pytest

from impulsecalc3.case import write_case
from impulsecalc3.geometry import BladeSpec
from impulsecalc3.job import domain_x, load_job
from impulsecalc3.meanline import compute_meanline
from impulsecalc3.ofenv import foam_env, openfoam_available, run_foam
from impulsecalc3.times import compute_times
from viewer.ofio import parse_checkmesh


@pytest.mark.live
def test_live_checkmesh_and_solver_starts(tmp_path: Path):
    if not openfoam_available():
        pytest.skip("OpenFOAM 2412 not on this box")
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
    case = tmp_path / "of"
    mesh, t_end = write_case(case, job, ml, times, spec)
    assert t_end >= 5.0 * times.t_chord_convective_s
    env = foam_env()
    rc = run_foam(["checkMesh"], cwd=case, log_name="log.checkMesh", env=env)
    log = (case / "log.checkMesh").read_text(encoding="utf-8", errors="replace")
    chk = parse_checkmesh(log)
    assert rc == 0 or "Mesh stats" in log
    assert chk.get("n_hex") == mesh.n_cells or chk.get("n_cells") == mesh.n_cells
    assert not chk.get("negative_volume")
    assert not chk.get("open_cells")
    # Short start: prove rhoCentralFoam accepts the case. Not the design 5-chord job.
    cd = case / "system" / "controlDict"
    txt = cd.read_text()
    txt = txt.replace("endTime         5.2631579e-05", "endTime         3e-08")
    # whatever the written endTime, clamp stop by replacing the endTime line
    import re
    txt = re.sub(r"endTime\s+\S+;", "endTime         3e-08;", txt)
    txt = re.sub(r"writeInterval\s+\S+;", "writeInterval   3e-08;", txt)
    cd.write_text(txt)
    rc2 = run_foam(["rhoCentralFoam"], cwd=case, log_name="log.rhoCentralFoam", env=env)
    slog = (case / "log.rhoCentralFoam").read_text(encoding="utf-8", errors="replace")
    assert "FOAM FATAL" not in slog
    assert rc2 == 0
    assert any((case / n).is_file() for n in ("log.rhoCentralFoam",))
