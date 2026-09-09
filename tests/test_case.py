"""Case writer rejects p1~Pc and does not Dirichlet 0.95 p1."""

import copy
from pathlib import Path

import pytest

from impulsecalc3.case import write_case
from impulsecalc3.geometry import BladeSpec
from impulsecalc3.job import load_job, validate_job, domain_x
from impulsecalc3.meanline import compute_meanline
from impulsecalc3.times import compute_times


def test_rejects_p1_near_pc():
    job = copy.deepcopy(load_job())
    pc = float(job["engine"]["Pc_bar"]) * 1e5
    job["gas"]["p1_pa"] = pc
    with pytest.raises(ValueError, match="p1 looks like Pc"):
        validate_job(job)


def test_written_p_is_not_095_p1(tmp_path: Path):
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
    case = tmp_path / "case"
    mesh, t_end = write_case(case, job, ml, times, spec)
    ptxt = (case / "0" / "p").read_text()
    p1 = float(gas["p1_pa"])
    assert "waveTransmissive" in ptxt or "inletOutlet" in ptxt
    assert f"{0.95 * p1:.8g}" not in ptxt
    assert "noSlip" in (case / "0" / "U").read_text()
    cd = (case / "system" / "controlDict").read_text()
    assert "rhoCentralFoam" in cd
    assert t_end >= times.t_end_floor_s
    assert t_end >= 1.2 * times.t_domain_acoustic_s - 1e-16
    assert "blade0" in mesh.patches and mesh.patches["blade0"] > 0
    meta = (case / "impulsecalc3_case_meta.json").read_text()
    assert "eta_design_proxy" not in meta
