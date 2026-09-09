"""Verification against ORBIT PURPL triangle sheet. UNSIGNED. Not Marlin live gas."""

from __future__ import annotations

import json
import math
from pathlib import Path

from impulsecalc3.job import ROOT, load_job
from impulsecalc3.meanline import FT_TO_M, compute_meanline, purpl_sheet_si

SHEET = purpl_sheet_si()


def test_ft_to_m_matches_plot_si():
    assert abs(FT_TO_M - 0.3048) < 1e-12
    assert abs(SHEET["c1_m_s"] - 1666.58) / 1666.58 < 0.0005
    assert abs(SHEET["c2_m_s"] - 1190.11) / 1190.11 < 0.0005
    assert abs(SHEET["w1_m_s"] - 1425.83) / 1425.83 < 0.0005
    assert abs(SHEET["u_m_s"] - 252.90) / 252.90 < 0.0005


def test_meanline_closes_purpl_impulse_vectors():
    job = load_job(ROOT / "configs" / "purpl_triangles.json")
    ml = compute_meanline(job)
    assert ml.predicted is True
    assert abs(ml.w1_m_s - ml.w2_m_s) / ml.w1_m_s < 1e-9
    assert abs(ml.w1_m_s - SHEET["w1_m_s"]) / SHEET["w1_m_s"] < 0.005
    assert abs(ml.u_m_s - SHEET["u_m_s"]) / SHEET["u_m_s"] < 0.005
    assert abs(ml.c1_m_s - SHEET["c1_m_s"]) / SHEET["c1_m_s"] < 0.005
    assert abs(ml.c2_m_s - SHEET["c2_m_s"]) / SHEET["c2_m_s"] < 0.005
    # C = W + U (U along +pitch)
    rec_c1 = math.hypot(ml.wx1, ml.wy1 + ml.u_m_s)
    rec_c2 = math.hypot(ml.wx2, ml.wy2 + ml.u_m_s)
    assert abs(rec_c1 - ml.c1_m_s) < 1e-9
    assert abs(rec_c2 - ml.c2_m_s) < 1e-9
    assert abs(ml.alpha1_abs_deg - SHEET["alpha1_abs_deg"]) < 0.3
    # Sheet α2=23.35 is a different angle convention than atan2(Cθ, Ca) ≈ -66.6 deg.
    # Do not fake a match. Vector |C2| is the accept.
    assert abs(ml.alpha2_abs_deg - (-66.6)) < 1.0
    assert ml.loss_scoping["authority"] == "SCOPING"
    assert ml.loss_scoping["predicted"] is True
    assert "eta_design_proxy" not in ml.to_dict()


def test_marlin_live_json_gas_not_overwritten():
    mar = json.loads((ROOT / "configs" / "marlin_v2_rotor.json").read_text(encoding="utf-8"))
    assert abs(float(mar["gas"]["w1_m_s"]) - 950.0) < 1e-6
    assert abs(float(mar["gas"]["blade_speed_u_m_s"]) - 450.0) < 1e-6
    pur = json.loads((ROOT / "configs" / "purpl_triangles.json").read_text(encoding="utf-8"))
    assert abs(float(pur["gas"]["w1_m_s"]) - 950.0) > 100.0


def test_app_html_has_triangle_img():
    html = (ROOT / "viewer" / "app.html").read_text(encoding="utf-8")
    assert 'id="triangles"' in html
    assert "/plot/triangles.png" in html
    assert 'id="force_history"' in html
    assert 'id="contour_p"' in html
    assert 'id="wall_cp_blade0"' in html
