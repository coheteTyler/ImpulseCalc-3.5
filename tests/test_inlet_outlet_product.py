"""PREDICTED inlet EOS residual and six-chord dump defaults. Not STAND MEASURED."""

from __future__ import annotations

from impulsecalc3.job import (
    ROOT,
    cascade_view_xlim_m,
    eos_residual_frac,
    eos_residual_pct,
    perfect_gas_rho_kg_m3,
    validate_job,
)
from impulsecalc3.preview import knobs_to_job

MARLIN = ROOT / "configs" / "marlin_v2_rotor.json"


def test_rho_residual_independent_and_outlet_defaults():
    job = knobs_to_job({"family": "cup", "p1": 4.0e5, "T1": 900.0})
    gas = job["gas"]
    p, T, R = float(gas["p1_pa"]), float(gas["t1_k"]), float(gas["r_specific_j_kg_k"])
    rho_eos = perfect_gas_rho_kg_m3(p, R, T)
    assert abs(float(gas["rho1_kg_m3"]) - rho_eos) / rho_eos < 1e-12
    assert eos_residual_frac(p, gas["rho1_kg_m3"], R, T) < 1e-12
    assert abs(eos_residual_pct(p, gas["rho1_kg_m3"], R, T)) < 1e-9

    typed = 2.0 * rho_eos
    gas["rho1_kg_m3"] = typed
    p_before = p
    job = validate_job(job)
    gas = job["gas"]
    assert abs(float(gas["rho1_kg_m3"]) - typed) < 1e-12
    assert abs(float(gas["p1_pa"]) - p_before) < 1e-6
    assert abs(float(gas["t1_k"]) - T) < 1e-12
    assert abs(float(gas["r_specific_j_kg_k"]) - R) < 1e-12
    assert abs(eos_residual_pct(gas["p1_pa"], gas["rho1_kg_m3"], gas["r_specific_j_kg_k"], gas["t1_k"]) - 100.0) < 1e-6

    cfd = dict(job["cfd"])
    cfd.pop("x_dn_c")
    cfd.pop("n_outlet")
    job["cfd"] = cfd
    job = validate_job(job)
    assert abs(float(job["cfd"]["x_dn_c"]) - 6.0) < 1e-12
    assert int(job["cfd"]["n_outlet"]) >= 24
    assert abs(float(job["cfd"]["x_up_c"]) - 1.5) < 1e-12


def test_knobs_dump_and_of_does_not_rewrite_p1():
    job = knobs_to_job({"family": "cup", "p1": 4.0e5, "OF": 1.7, "mdot_engine_kg_s": 3.3})
    assert abs(job["gas"]["p1_pa"] - 4.0e5) < 1e-6
    assert abs(float(job["engine"]["OF"]) - 1.7) < 1e-12
    assert abs(float(job["cfd"]["x_dn_c"]) - 6.0) < 1e-12
    assert int(job["cfd"]["n_outlet"]) >= 24
    assert abs(float(job["cfd"]["x_up_c"]) - 1.5) < 1e-12
    assert job["cfd"]["outlet_p"] == "waveTransmissive"
    x0, x1 = cascade_view_xlim_m(job)
    c = float(job["geometry"]["chord_m"])
    assert abs(x0 + 1.5 * c) < 1e-12
    assert abs(x1 - 2.0 * c) < 1e-12


def test_marlin_json_not_rewritten():
    before = MARLIN.read_bytes()
    knobs_to_job({"family": "cup", "x_dn_c": 6.0, "n_outlet": 28})
    assert MARLIN.read_bytes() == before
    raw = __import__("json").loads(before)
    assert raw["cfd"]["x_dn_c"] == 2.0
    assert raw["cfd"]["n_outlet"] == 14
