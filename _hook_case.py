from pathlib import Path
p = Path("/workspace/ImpulseCalc3/impulsecalc3/case.py")
t = p.read_text()
t = t.replace(
    "def _cyclic_empty_walls() -> tuple[str, str]:\n    \"\"\"U vs scalar patch blocks for the shared topology.\"\"\"\n    blades_u = \"\\n\".join(\n        f\"            blade{k} {{ type noSlip; }}\" for k in range(3)\n    )\n    blades_s = \"\\n\".join(\n        f\"            blade{k} {{ type zeroGradient; }}\" for k in range(3)\n    )\n    shared_u = textwrap.dedent(\n        f\"\"\"\\\n            bottom {{ type cyclic; }}\n            top    {{ type cyclic; }}\n            frontAndBack {{ type empty; }}\n        \"\"\"\n    )\n",
    "def _cyclic_empty_walls(n_blades: int = 3, cyclic: bool = True) -> tuple[str, str]:\n    \"\"\"U vs scalar patch blocks for the shared topology.\"\"\"\n    blades_u = \"\\n\".join(\n        f\"            blade{k} {{ type noSlip; }}\" for k in range(n_blades)\n    )\n    blades_s = \"\\n\".join(\n        f\"            blade{k} {{ type zeroGradient; }}\" for k in range(n_blades)\n    )\n    cyc = (\n        \"            bottom { type cyclic; }\\n            top    { type cyclic; }\\n\"\n        if cyclic\n        else \"\"\n    )\n    shared_u = textwrap.dedent(\n        f\"\"\"\\\n{cyc}            frontAndBack {{ type empty; }}\n        \"\"\"\n    )\n",
)
# control dict blade loop
t = t.replace(
    "    force_blocks = []\n    surf_blocks = []\n    for k in range(3):",
    "    n_blades = int(job.get(\"_n_blades_patches\") or job.get(\"geometry\", {}).get(\"n_blades_cascade\") or 3)\n    force_blocks = []\n    surf_blocks = []\n    for k in range(n_blades):",
)
t = t.replace(
    "    u_shared, s_shared = _cyclic_empty_walls()",
    "    n_blades = int(job.get(\"_n_blades_patches\") or 3)\n    cyclic = bool(job.get(\"_cyclic_pitch\", True))\n    u_shared, s_shared = _cyclic_empty_walls(n_blades=n_blades, cyclic=cyclic)",
)
old = '''    mesh = write_polymesh(case_dir, job, spec, poly=poly)
    write_thermophysical(case_dir, job)
    write_schemes(case_dir)
    write_solution(case_dir)
    t_end = write_control_dict(case_dir, job, times)
    write_fields(case_dir, job, ml)
'''
new = '''    mesh = write_polymesh(case_dir, job, spec, poly=poly)
    n_b = len([k for k in mesh.patches if str(k).startswith("blade")])
    job["_n_blades_patches"] = n_b
    job["_cyclic_pitch"] = bool(mesh.patches.get("bottom"))
    write_thermophysical(case_dir, job)
    write_schemes(case_dir)
    write_solution(case_dir)
    t_end = write_control_dict(case_dir, job, times)
    write_fields(case_dir, job, ml)
'''
if old not in t:
    raise SystemExit('write_case body missing')
p.write_text(t.replace(old, new, 1))
print('case hooked')
