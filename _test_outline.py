import sys
sys.path.insert(0, "/workspace/ImpulseCalc3")
from impulsecalc3.preview import knobs_to_job, write_preview
knobs = dict(
    hu_mm=5, hl_mm=2.2, le_mm=0.4, te_mm=0,
    lin_mm=4.25, lout_mm=4.25, t_mm=1.4,
    r_tr_mm=3.5, r_main_mm=4.8, c=0.01,
    s_mm=6.0, packing_driver="pitch",
)
job = knobs_to_job(knobs)
job["name"] = "fillet_outline"
job["output_dir"] = "output/geom_tests/fillet_outline"
r = write_preview(job)
print("ok", r.get("ok"), "kind", r.get("mesh_kind"), "n_cells", r.get("n_cells"))
print("warn", r.get("geom_warnings"))
print("yspan_mm", None if r.get("poly_ymin_m") is None else round((r["poly_ymax_m"]-r["poly_ymin_m"])*1e3, 3))
print("png", r.get("png"))
