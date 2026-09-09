"""Viewer knobs cover job geometry/CFD. Cited NTRS tags only. PREDICTED."""

from __future__ import annotations

from pathlib import Path

from impulsecalc3.job import ROOT
from impulsecalc3.preview import knobs_to_job

APP = ROOT / "viewer" / "app.html"
MARLIN = ROOT / "configs" / "marlin_v2_rotor.json"


def test_knobs_to_job_geometry_and_cfd_counts():
    job = knobs_to_job(
        {
            "family": "cup",
            "le_radius_c": 0.055,
            "te_radius_c": 0.033,
            "incidence_deg": 2.5,
            "deviation_deg": -1.5,
            "n_around": 72,
            "n_radial": 14,
            "n_inlet": 11,
            "n_outlet": 30,
            "n_cyclic": 18,
            "x_up_c": 1.7,
            "x_dn_c": 6.0,
            "stretch": 1.41,
        }
    )
    g = job["geometry"]
    cfd = job["cfd"]
    assert abs(float(g["le_radius_c"]) - 0.055) < 1e-12
    assert abs(float(g["le_fillet_r_c"]) - 0.055) < 1e-12
    assert abs(float(g["te_radius_c"]) - 0.033) < 1e-12
    assert abs(float(g["te_fillet_r_c"]) - 0.033) < 1e-12
    assert abs(float(g["incidence_deg"]) - 2.5) < 1e-12
    assert abs(float(g["deviation_deg"]) + 1.5) < 1e-12
    assert int(g["n_blades_cascade"]) == 3
    assert int(cfd["n_around"]) == 72
    assert int(cfd["n_radial"]) == 14
    assert int(cfd["n_inlet"]) == 11
    assert int(cfd["n_outlet"]) == 30
    assert int(cfd["n_cyclic"]) == 18
    assert abs(float(cfd["x_up_c"]) - 1.7) < 1e-12
    assert abs(float(cfd["x_dn_c"]) - 6.0) < 1e-12
    assert abs(float(cfd["stretch"]) - 1.41) < 1e-12
    assert cfd["outlet_p"] == "waveTransmissive"
    assert job["predicted"] is True


def test_app_html_orbit_knobs_and_cited_ntrs_only():
    html = APP.read_text(encoding="utf-8")
    for i in (
        "le_radius_c",
        "te_radius_c",
        "incidence_deg",
        "deviation_deg",
        "n_around",
        "n_radial",
        "n_inlet",
        "n_outlet",
        "n_cyclic",
        "x_up_c",
        "x_dn_c",
        "stretch",
        "btn-mesh-counts",
        "mesh_modal",
        "identities",
        "ntrs_stamp_sigma",
        "ntrs_stamp_beta1",
        "ntrs_stamp_beta2",
        "ntrs_stamp_w1",
    ):
        assert f'id="{i}"' in html
    assert "Identities (cheat sheet)" in html
    assert "Mesh counts" in html
    assert 'id="n_blades_cascade"' not in html
    assert "NOMINAL vs NTRS" in html
    assert "https://ntrs.nasa.gov/citations/19940018586" in html
    assert "https://ntrs.nasa.gov/citations/19920066198" in html
    assert "https://ntrs.nasa.gov/citations/19740025363" in html
    assert "https://ntrs.nasa.gov/citations/19680010807" in html
    assert "= 140 deg" in html
    assert "= 0.8" in html
    assert "1.5" in html and "5.0" in html
    assert "sonic = 1.0 INFO not a fail" in html
    assert "waveTransmissive" in html
    assert "fieldInf = p1" in html
    assert "η from CFD" in html and "forbidden" in html
    for knob_id in ("p1", "T1", "rho1", "mu", "gamma", "Rspec", "U"):
        assert f'id="ntrs_stamp_{knob_id}"' not in html
    assert "ntrs_stamp_W1" not in html  # W1=950 is not an NTRS nominal; Mw1 Goldman tag is ntrs_stamp_w1


def test_orbit_does_not_rewrite_marlin():
    before = MARLIN.read_bytes()
    knobs_to_job(
        {
            "family": "cup",
            "n_around": 80,
            "x_dn_c": 6,
            "le_radius_c": 0.05,
        }
    )
    assert MARLIN.read_bytes() == before
