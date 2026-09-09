"""Geometry closed, metal angles, pitch fit."""

from impulsecalc3.geometry import (
    BladeSpec,
    camber_end_tangents_deg,
    center_in_pitch,
    closed_profile,
    polygon_signed_area,
)
from impulsecalc3.job import load_job, pitch_m
from impulsecalc3.meanline import compute_meanline


def test_profile_closed_ccw_fits_pitch():
    job = load_job()
    ml = compute_meanline(job)
    g = job["geometry"]
    spec = BladeSpec(
        chord_m=g["chord_m"],
        beta1_metal_deg=ml.beta1_metal_deg,
        beta2_metal_deg=ml.beta2_metal_deg,
        thickness_c=g["thickness_c"],
        le_radius_c=g["le_radius_c"],
        te_radius_c=g["te_radius_c"],
        n_points=int(g["n_profile_points"]),
    )
    poly = closed_profile(spec)
    assert poly[0] == poly[-1]
    assert polygon_signed_area(poly) > 0
    poly2, _ = center_in_pitch(poly, pitch_m(job))
    ys = [p[1] for p in poly2]
    half = 0.5 * pitch_m(job)
    assert min(ys) > -0.49 * 2 * half
    assert max(ys) < 0.49 * 2 * half
    b1, b2 = camber_end_tangents_deg(spec)
    assert abs(b1 - 72.0) < 3.0
    assert abs(b2 - (-72.0)) < 3.0
