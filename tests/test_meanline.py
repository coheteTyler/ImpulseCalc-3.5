from impulsecalc3.job import load_job
from impulsecalc3.meanline import compute_meanline


def test_euler_matches_two_u_w_sinbeta():
    job = load_job()
    ml = compute_meanline(job)
    import math

    u, w, b = ml.u_m_s, ml.w1_m_s, math.radians(ml.beta1_flow_deg)
    expect = 2 * u * w * math.sin(b)
    assert abs(ml.euler_work_j_kg - expect) / expect < 1e-6
    assert ml.predicted is True
    assert ml.ainley_mathieson["book"].startswith("Ainley-Mathieson")
    assert "eta_design_proxy" not in ml.to_dict()
    assert ml.Mw1 > 1.3  # W1=950 PREDICTED supersonic relative


def test_pc_not_used_as_p1():
    job = load_job()
    pc_bar = float(job["engine"]["Pc_bar"])
    p1 = float(job["gas"]["p1_pa"])
    assert abs(p1 - pc_bar * 1e5) / (pc_bar * 1e5) > 0.2
