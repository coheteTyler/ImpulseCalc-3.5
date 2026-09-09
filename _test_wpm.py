import sys, traceback
sys.path.insert(0, "/workspace/ImpulseCalc3")
from pathlib import Path
from impulsecalc3.preview import knobs_to_job
from impulsecalc3.geometry import spec_from_job, profile_from_job
from impulsecalc3.mesh import write_polymesh

knobs = dict(
    hu_mm=5, hl_mm=2.2, le_mm=0.4, te_mm=0,
    lin_mm=4.25, lout_mm=4.25, t_mm=1.4,
    r_tr_mm=3.5, r_main_mm=4.8, c=0.01,
    s_mm=6.0, packing_driver="pitch",
)
job = knobs_to_job(knobs)
spec = spec_from_job(job)
poly = profile_from_job(job, spec)
case = Path("/tmp/passage_s6")
case.mkdir(parents=True, exist_ok=True)
try:
    mb = write_polymesh(case, job, spec, poly)
    print("kind", mb.mesh_kind, "cells", mb.n_cells, "patches", mb.patches)
    print("amin", mb.min_area_2d, "first", mb.first_cell_m)
    for n in mb.check_notes:
        print(" ", n)
except Exception:
    traceback.print_exc()
