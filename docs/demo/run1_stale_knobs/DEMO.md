# Demo: blade surface points -> labelled 2D cascade mesh

Status: **FAIL**. `checkMesh -meshQuality` reported `Failed 1 mesh checks.` See `FAIL.json`.
Stop rule applied: one mesh write, one checkMesh. No remesh and no knob changes.

## Command

```
.venv/bin/python examples/demo_points_to_mesh.py \
  --upper examples/demo_upper.csv --lower examples/demo_lower.csv \
  --scale 1e-3 --n-blades 25 --mean-radius 0.0375 --out output/demo_points
```

Run path: `impulsecalc3.run.run_job(job, skip_solve=True, run_solve=False, run_post=False)`
-> `write_case` -> `renumberMesh -overwrite` -> `checkMesh -meshQuality`. OpenFOAM ESI v2412 runs in Docker
(`opencfd/openfoam-run:2412`). No solver was run.

## Inputs

- `examples/demo_upper.csv` (140 pts) and `examples/demo_lower.csv` (152 pts), columns x_mm,y_mm, LE -> TE.
  They are `configs/geom_points.json` `profile_points` split at LE (min x) and TE (max x), written in mm.
- The script joins upper (LE->TE) and reversed lower (TE->LE) into one closed loop of 290 distinct points. It
  matches the original loop to 1.8e-18 m.
- Scale 1e-3 (mm -> m). Chord (LE->TE distance) = 9.767 mm.
- Z = 25, r_m = 0.0375 m -> pitch = 2*pi*r_m/Z = 9.4248 mm (`packing_driver = "z"`), solidity 1.036.
- Resolution: defaults from `configs/geom_points.json`: n_around 56, n_radial 12, stretch (wall growth) 1.25.
  The mesher raised the O-ring to n_around 60 (S/N=16, E=14, W=14).
- Job used: `demo_points_job.json`.

## Result

- Cells: 1728 hexahedra, one pitch, one blade, 2D slab 1 mm thick (z).
- First cell on the wall: 8.12e-6 m (derived, see Limits). O-grid thickness d_o = 0.30 mm (capped by 0.28*g_min, g_min = 1.071 mm).

| Patch | Type | nFaces |
|---|---|---|
| inlet | patch | 28 |
| outlet | patch | 28 |
| bottom | cyclic (translational, neighbour top, separation (0 0.009424778 0)) | 44 |
| top | cyclic (translational, neighbour bottom) | 44 |
| frontAndBack | empty | 3456 |
| blade0 | wall | 60 |

Periodic = the bottom/top translational cyclic pair, one pitch apart.

## checkMesh (-meshQuality), verbatim verdict: `Failed 1 mesh checks.`

- Max non-orthogonality 82.690135, average 13.283428.
- Faces > 70 deg: 10. The standard geometry section prints "Non-orthogonality check OK". The failure comes from
  the -meshQuality "faces in error" list (`non-orthogonality > 70 degrees : 10`).
- Max skewness 2.1712533 OK. Face pyramids OK. Cell volumes OK (min 1.147e-12 m3). Max aspect ratio 41.1.
  Topology OK, 1 region. Coupled point match OK (avg 3.9e-11).
- A separate plain `checkMesh` (without -meshQuality) was not run because of the one-checkMesh rule.
- run_job note: "checkMesh failed 1 quality checks; Cell volumes OK — foam allowed (Track A)".

## Images

- `mesh_preview.png`: polyMesh wireframe. The one-pitch mesh is drawn stacked 3 times for viewing only.

## Label

PREDICTED. This is a mesh demonstration only. The gas state is copied from `configs/geom_points.json` and is unsigned. No flow was solved.

## Limits

- 2D slab (one cell in z, `frontAndBack` empty). No tip gap, no radius change or 3D effects. The cascade is at mean radius.
- No explicit first-cell knob exists in the mesher. For `profile_points` blades the mesher uses its AABB O-grid
  branch: d_o = min(0.45 mm, 0.22*cyclic clearance, 6% chord), capped at 0.28*g_min, and n_radial layers
  inside it grow at the stretch ratio. `cfd.yplus_target` / `oh_shock.y1_wall_m` are read only on the
  impulse-bucket cavity branch, not this one. `--yplus-target` is exposed but has no effect on this path.
- The mesh is coarse: 60 wall faces around the blade.
