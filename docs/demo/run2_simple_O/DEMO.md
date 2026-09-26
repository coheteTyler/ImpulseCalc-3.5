# Demo: blade surface points -> labelled 2D cascade mesh (run 2, live knobs)

Status: **FAIL**. `checkMesh -meshQuality` reported `Failed 1 mesh checks.` See `FAIL.json`.
Stop rule applied: one mesh write, one checkMesh. No remesh, no further knob changes, no mesher edits.
Run 1 used the stale `configs/geom_points.json` cfd block and is archived in `run1_stale_knobs/`
(1728 cells, also `Failed 1 mesh checks.`).

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
- The script joins them into one closed loop of 290 distinct points, identical to the original loop (max deviation 1.8e-18 m).
- Scale 1e-3. Chord (LE->TE) 9.767 mm. Z = 25, r_m = 0.0375 m -> pitch 9.4248 mm, solidity 1.036.
- Gas and engine come from `configs/geom_points.json`. The cfd block is `LIVE_CFD`, embedded in the script and copied
  from `output/geom_tests/knobs_preview/knobs_preview.json['cfd']`. `--cfd-from` can load it from a file instead.
- Job used: `demo_points_job.json`.

## Knobs vs live (knobs_preview.json cfd)

| knob | live | demo run 2 | run 1 (stale) |
|---|---|---|---|
| mesh | body_fitted_OH | body_fitted_OH | body_fitted_OH |
| n_around | 56 | 56 | 56 |
| n_radial | 20 | 20 | 12 |
| n_inlet / n_outlet / n_cyclic | 10 / 28 / 16 | 10 / 28 / 16 | 10 / 14 / 16 |
| n_pitch_fill | 40 | 40 | 7 |
| n_pitchwise_throat | 40 | 40 | (default 40) |
| stretch / inlet_stretch / growth | 1.12 / 1.12 / 1.25 | 1.12 / 1.12 / 1.25 | 1.25 / (1.12) / (1.25) |
| x_up_c / x_dn_c | 0.95 / 2.5 | 0.95 / 2.5 | 1.5 / 2.0 |
| outlet_p | waveTransmissive | waveTransmissive | inletOutlet |
| le_cluster / n_le / te_angular_min | 2.5 / 14 / 10 | 2.5 / 14 / 10 | defaults |
| dump_rx / dump_dx_last_m / wake_cx / x_dense_c | 1.18 / 0.0014 / 1.0 / 0.25 | same | defaults |
| yplus_target / u_tau_frac_w1 | 1.0 / 0.05 | 1.0 / 0.05 | defaults |
| smoke_end_s | 1.05e-5 | dropped (mesh only) | 1.05e-5 |

Knob diff vs live, checked programmatically: only `smoke_end_s`, which was dropped on purpose.
The geometry differs. Live is `profile_family impulse_bucket` with Z=29 and pitch 8.114 mm. The demo is
`profile_points` with Z=25 and pitch 9.425 mm.

## Result

- Cells: 6428 hexahedra (the live knobs_preview case has 15148). One pitch, one blade, 2D slab 1 mm thick.
- First cell on the wall: 4.84e-6 m (derived). O-grid thickness d_o = 0.30 mm (capped by 0.28*g_min, g_min 1.071 mm).
  The O-ring has 60 cells around the blade.

| Patch | Type | nFaces |
|---|---|---|
| inlet | patch | 94 |
| outlet | patch | 94 |
| bottom | cyclic (translational, neighbour top, separation (0 0.009424778 0)) | 58 |
| top | cyclic (translational, neighbour bottom) | 58 |
| frontAndBack | empty | 12856 |
| blade0 | wall | 60 |

Periodic = the bottom/top translational cyclic pair, one pitch apart.

## checkMesh (-meshQuality), verbatim verdict: `Failed 1 mesh checks.`

- Non-orthogonality max 83.132689, average 8.0202354. Faces > 70 deg: 10.
- The standard geometry section prints "Non-orthogonality check OK". The failure comes from the -meshQuality
  "faces in error" list (`non-orthogonality > 70 degrees : 10`).
- Max skewness 2.1712533 OK. Face pyramids OK. Cell volumes OK (min 6.846e-13 m3). Max aspect ratio 160.5.
  Topology OK, 1 region. Coupled point match OK.
- The 10 faces sit at the LE and TE tips and along the concave (inner) surface O-grid. Locations are in `FAIL.json` and
  circled in `mesh_zoom_passage.png`.

## Why the knob fix did not change the verdict

Run 1 and run 2 have the same 10 faces and the same max skewness (2.1712533), so the near-wall O-ring did not change.
For `profile_family profile_points` the mesher takes its padded-AABB O-grid branch. The live
`impulse_bucket` case takes the `build_body_fitted_oh_shock` branch (y+ inflation, pitchwise throat
resolution). The cfd knobs cannot switch this; the branch is selected by `profile_family` in `mesh.py`.
The live knobs_preview report also lists `failed_checks: 1` (non-ortho max 79.09), so live is not a clean reference either.

## Images

- `mesh_preview.png`: polyMesh wireframe. The one-pitch mesh is drawn stacked 3 times for viewing only.
- `mesh_zoom_passage.png`: z=0 faces from the polyMesh, with the 10 faces over 70 deg circled. The left panel
  (full passage) is correct. In the LE/TE panels the view centre is offset from the tips, and the blue wall highlight did not
  render. This was a single attempt, not redone.

## Label

PREDICTED. This is a mesh demonstration only. The gas state is unsigned and no flow was solved.

## Limits

- 2D slab (one cell in z, `frontAndBack` empty). No tip gap, no radius change or 3D effects. The cascade is at mean radius.
- No explicit first-cell knob. On the profile_points branch the first cell is derived from d_o, n_radial and stretch.
  `--yplus-target` is written but only used on the impulse-bucket branch.
- 60 wall faces around the blade.
