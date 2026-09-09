"""endTime floor is 5 c/W1, never 2 us."""

from impulsecalc3.job import domain_x, load_job
from impulsecalc3.times import compute_times


def test_times_floor_five_chords():
    job = load_job()
    g, gas = job["geometry"], job["gas"]
    xin, xout = domain_x(job)
    t = compute_times(
        chord_m=g["chord_m"],
        w1_m_s=gas["w1_m_s"],
        gamma=gas["gamma"],
        r_specific=gas["r_specific_j_kg_k"],
        t1_k=gas["t1_k"],
        x_in_m=xin,
        x_out_m=xout,
        n_chords_min=5.0,
    )
    t_conv = g["chord_m"] / gas["w1_m_s"]
    assert abs(t.t_chord_convective_s - t_conv) / t_conv < 1e-12
    assert t.t_end_floor_s >= 5.0 * t_conv - 1e-16
    assert t.t_end_floor_s > 1e-5  # never 2e-6
    assert t.t_chord_acoustic_s > 0
    assert t.t_domain_acoustic_s > t.t_chord_acoustic_s
    assert t.t_end_floor_s >= 1.2 * t.t_domain_acoustic_s - 1e-16
    assert t.t_end_floor_s >= max(5.0 * t_conv, 1.2 * t.t_domain_acoustic_s) - 1e-16
