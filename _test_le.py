import sys
sys.path.insert(0, "/workspace/ImpulseCalc3")
from impulsecalc3.preview import knobs_to_job
from impulsecalc3.geometry import profile_from_job, spec_from_job, split_ps_ss
from impulsecalc3.job import pitch_m
job = knobs_to_job(dict(hu_mm=5,hl_mm=2.2,le_mm=0.4,te_mm=0,lin_mm=4.25,lout_mm=4.25,t_mm=1.4,r_tr_mm=3.5,r_main_mm=4.8,c=0.01,s_mm=6.0,packing_driver="pitch"))
poly = profile_from_job(job, spec_from_job(job))
ps, ss, _, _ = split_ps_ss(poly)
s = pitch_m(job)
print("SS0 le", ss[0], "te", ss[-1])
print("PS0 le", ps[0], "te", ps[-1])
print("PS0+s le", (ps[0][0], ps[0][1]+s))
print("dy le", (ps[0][1]+s) - ss[0][1], "s", s)
print("dx le", ps[0][0]-ss[0][0])
