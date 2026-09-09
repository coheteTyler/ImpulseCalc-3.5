from pathlib import Path

from impulsecalc3.case import write_case
from impulsecalc3.geometry import BladeSpec
from impulsecalc3.job import domain_x, load_job
from impulsecalc3.meanline import compute_meanline
from impulsecalc3.times import compute_times


def _write(tmp: Path):
    job = load_job()
    g, gas = job["geometry"], job["gas"]
    spec = BladeSpec(
        chord_m=g["chord_m"],
        beta1_metal_deg=g["beta1_flow_deg"],
        beta2_metal_deg=g["beta2_flow_deg"],
        thickness_c=g["thickness_c"],
        le_radius_c=g["le_radius_c"],
        te_radius_c=g["te_radius_c"],
    )
    ml = compute_meanline(job)
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
    case = tmp / "case"
    mesh, t_end = write_case(case, job, ml, times, spec)
    return case, mesh, t_end, times, job


def test_case_has_noslip_per_blade_not_slip(tmp_path: Path):
    case, mesh, t_end, times, job = _write(tmp_path)
    u = (case / "0" / "U").read_text()
    assert "noSlip" in u
    assert "type slip" not in u
    assert "blade0" in u and "blade1" in u and "blade2" in u
    p = (case / "0" / "p").read_text()
    assert "0.95" not in p
    assert "waveTransmissive" in p or "inletOutlet" in p
    assert "522500" not in p  # the old cooked dump
    meta = (case / "impulsecalc3_case_meta.json").read_text()
    assert "eta_design_proxy" not in meta
    readme = (case / "README.txt").read_text()
    assert "t_chord_convective" in readme or "c/W1" in readme
    assert "t_chord_acoustic" in readme or "c/a" in readme
    assert t_end >= 5 * times.t_chord_convective_s - 1e-12
    assert t_end >= 1.2 * times.t_domain_acoustic_s - 1e-12
    assert t_end > 1e-5  # never 2e-6
    assert mesh.n_cells > 500
    assert mesh.patches["blade0"] > 10
    assert "body-fitted" in " ".join(mesh.check_notes).lower() or "O-grid" in " ".join(mesh.check_notes)
    cd = (case / "system" / "controlDict").read_text()
    assert "rhoCentralFoam" in cd
    assert "forces_blade0" in cd
    assert "patches         (blade0)" in cd


def test_one_machine_in_default_json():
    job = load_job()
    assert job["name"] == "marlin_v2_rotor"
    assert job["geometry"]["n_blades_machine"] == 25
    assert job["gas"]["w1_m_s"] == 950
    assert job["predicted"] is True
