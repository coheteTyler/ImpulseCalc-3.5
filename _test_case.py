import sys
sys.path.insert(0, "/workspace/ImpulseCalc3")
from pathlib import Path
from impulsecalc3.preview import knobs_to_job, write_preview
from impulsecalc3.ofenv import run_foam

knobs = dict(
    hu_mm=5, hl_mm=2.2, le_mm=0.4, te_mm=0,
    lin_mm=4.25, lout_mm=4.25, t_mm=1.4,
    r_tr_mm=3.5, r_main_mm=4.8, c=0.01,
    s_mm=6.0, packing_driver="pitch",
)
job = knobs_to_job(knobs)
job["name"] = "passage_s6"
job["output_dir"] = "output/geom_tests/passage_s6"
r = write_preview(job)
print("ok", r.get("ok"), "kind", r.get("mesh_kind"), "cells", r.get("n_cells"))
print("warn", r.get("geom_warnings"))
case = Path("/workspace/ImpulseCalc3/output/geom_tests/passage_s6/openfoam_cases/passage_s6")
print("ctrl", (case / "system" / "controlDict").is_file())
rc = run_foam(["checkMesh"], cwd=case, log_name="log.checkMesh")
print("checkMesh rc", rc)
log = (case / "log.checkMesh").read_text(errors="replace")
for line in log.splitlines():
    low = line.lower()
    if any(k in low for k in ("mesh ok", "failed", "aspect", "skew", "non-ortho", "cells:", " ***", "pyramid", "fatal")):
        print(line)
