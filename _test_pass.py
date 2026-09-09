import sys
sys.path.insert(0, "/workspace/ImpulseCalc3")
from impulsecalc3.preview import knobs_to_job
from impulsecalc3.geometry import profile_from_job, spec_from_job
from impulsecalc3.passage import build_passage_oh
from impulsecalc3.job import pitch_m, domain_x

knobs = dict(
    hu_mm=5, hl_mm=2.2, le_mm=0.4, te_mm=0,
    lin_mm=4.25, lout_mm=4.25, t_mm=1.4,
    r_tr_mm=3.5, r_main_mm=4.8, c=0.01,
    s_mm=6.0, packing_driver="pitch",
)
job = knobs_to_job(knobs)
spec = spec_from_job(job)
poly = profile_from_job(job, spec)
ys = [p[1] for p in poly]
print("yspan_mm", (max(ys)-min(ys))*1e3, "pitch_mm", pitch_m(job)*1e3)
xin, xout = domain_x(job)
g = build_passage_oh(
    poly, pitch=pitch_m(job), x_in=xin, x_out=xout,
    n_in=10, n_out=28, n_stream=80, n_span=12, n_rad=10, stretch=1.15, d_o=2e-4,
)
print("d_o", g.d_o, "first", g.first_cell_m, "amin", g.min_area_2d, "cyclic", g.cyclic)
print("notes:")
for n in g.notes:
    print(" ", n)
print("shapes", g.o_south.shape, g.h_core.shape, g.h_inlet.shape, g.h_outlet.shape)
