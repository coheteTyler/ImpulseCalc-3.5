# Demo: blade surface points -> labelled 2D cascade mesh

Status (run 3): **checkMesh -meshQuality: `Mesh OK.`** Max non-orthogonality 62.166707 deg, 0 faces > 70 deg.
The demo case was written once and checked once after the code change.

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
  They are the `configs/geom_points.json` `profile_points` split at LE (min x) and TE (max x), in mm.
- The script joins them into one closed loop of 290 distinct points, identical to the original loop (max deviation 1.8e-18 m).
- Scale 1e-3. Chord (LE->TE) 9.767 mm. Z = 25, r_m = 0.0375 m -> pitch 9.4248 mm, solidity 1.036.
- Gas and engine come from `configs/geom_points.json`. The cfd block is `LIVE_CFD`, embedded in the script and copied from
  `output/geom_tests/knobs_preview/knobs_preview.json['cfd']`. Knob diff vs live: only `smoke_end_s`, dropped on purpose
  (see `run2_simple_O/DEMO.md` for the knob table).
- Job used: `demo_points_job.json`.

## Mesh

- 15148 hexahedra (the live knobs_preview case also has 15148; same knobs and topology counts, different metal and pitch).
  One pitch, one blade, 2D slab 1 mm thick.
- Path: high-def body_fitted_OH (Goldman O–H), the same path as the live impulse_bucket case.
- Wall first cell y1 = 6.063e-7 m, from yplus_target = 1 (`oh_shock.y1_wall_m`: u_tau = 0.05*W1 = 47.5 m/s,
  mu 4.5e-5, rho 1.5625). first_cell/y1 = 1.00.
- n_radial 20, wall growth 1.12, O-collar d_o 43.7 um. O ring has 201 cells around the blade, n_pitchwise_throat 40.
  TE angular cells in 20 deg: 43.

| Patch | Type | nFaces |
|---|---|---|
| inlet | patch | 94 |
| outlet | patch | 116 |
| bottom | cyclic (translational, neighbour top, separation (0 0.009424778 0)) | 122 |
| top | cyclic (translational, neighbour bottom) | 122 |
| frontAndBack | empty | 30296 |
| blade0 | wall | 202 |

Periodic = the bottom/top translational cyclic pair, exactly one pitch apart.

## checkMesh (-meshQuality), verbatim verdict: `Mesh OK.`

- Non-orthogonality max 62.166707, average 15.493637. Faces > 70 deg: 0.
- Max skewness 1.4343016 OK. Face pyramids OK. Cell volumes OK (min 6.163e-15 m3).
  Max aspect ratio 482.6 OK (y+≈1 wall layer).
- Topology OK, 1 region. Coupled point match OK (avg 3.924e-11).
- Faces in error: non-orthogonality > 70: 0; concavity > 80: 0; skewness > 50: 0.

## Code change behind run 3 (impulsecalc3/mesh.py, +8 / -11 lines)

- (A) `write_polymesh`: `use_cavity` now includes `profile_family == "profile_points"`. The `profile_has_cavity` gate is
  kept, so a cup-shaped point loop gets the high-def O–H path: y1 from y+, n_radial 15–25, n_pitchwise_throat.
- (B) `build_offset_oh`, nw H-block (above the inlet stem, between the west H-block and the top cyclic):
  - The west (inlet) edge now uses the same packed y-nodes as the east edge (the west edge of the north H-block, first
    Δy ≈ dn_o). This mirrors `h_sw`.
  - The 80-iteration Laplacian smoothing of that block was removed; it is now plain TFI.
- Root cause of the > 70 deg faces:
  - The east edge of the nw block is packed to a first Δy ≈ 5–12 um, while its west edge was uniform (≈ 78 um).
  - The Laplacian smoother pulled the interior nodes towards equal spacing against that packed edge. This sheared the
    thin bottom rows sideways.
  - The result was 28 faces (demo on this path) and 41 faces (live) at 70–79 deg, all inside the nw block along
    y ≈ pNW, from the inlet to the LE stem.
- Blade O-ring, H-blocks, dump, cyclic x-nodes, wall first cell and all-hex topology are unchanged. There are no physics
  or BC changes.

## Offline check (pure Python, same writer, no OpenFOAM)

The offline non-orthogonality script reproduced checkMesh exactly on the existing meshes (run 2: max 83.13, 10 faces;
live: max 79.09, 41 faces) before being used on the change.

| mesh | code | max non-ortho | faces > 70 |
|---|---|---|---|
| demo | HEAD (AABB path, run 2) | 83.133 | 10 |
| demo | + (A) high-def path | 76.023 | 28 (all nw block) |
| demo | + (A) + (B) | 62.167 | 0 |
| live knobs_preview job | HEAD | 79.094 | 41 (all nw block) |
| live knobs_preview job | + (A) + (B) | 63.873 | 0 |

The offline live result is in memory only; the live case was not rewritten. Run 3's checkMesh max (62.166707)
matches the offline demo number.

## Run history

| run | inputs / code | cells | checkMesh -meshQuality |
|---|---|---|---|
| run1 (`run1_stale_knobs/`) | stale geom_points cfd block, AABB O path | 1728 | Failed 1 mesh checks (10 faces > 70, max 82.69) |
| run2 (`run2_simple_O/`) | live cfd knobs, AABB O path (profile_points not routed) | 6428 | Failed 1 mesh checks (10 faces > 70, max 83.13) |
| run3 (this folder) | live cfd knobs + mesh.py (A)+(B) | 15148 | **Mesh OK** (max 62.17, 0 faces > 70) |

## Images

- `mesh_preview.png`: polyMesh wireframe from the writer. The one-pitch mesh is drawn stacked for viewing.
- `mesh_zoom_passage.png`: z=0 faces read from `constant/polyMesh`. Panels:
  - the passage with grey periodic images (the blade and its cyclic neighbour);
  - LE (min x) region;
  - TE (max x) region;
  - LE wall layers.

## Label

PREDICTED. This is a mesh demonstration only. The gas state is unsigned and no flow was solved.

## Limits

- 2D slab (one cell in z, `frontAndBack` empty). No tip gap, no radius change or 3D effects. The cascade is at mean radius.
- No explicit first-cell knob. y1 follows `yplus_target` / `u_tau_frac_w1` (estimated u_tau, not a solved y+).
- The high-def path needs a cup-shaped (down-opening) point loop. Other point loops fall back to the AABB O path.

## Tests

- `pytest tests/` was run on isolated copies of the repo with OpenFOAM/Docker disabled, so the foam-dependent tests were skipped.
- Before (HEAD mesh.py): 17 failed, 112 passed, 5 skipped.
- After: 17 failed, 112 passed, 5 skipped.
- The 17 failures are the same tests before and after (pre-existing: app.html ids, HOH mesher, knob polygons, and others).
