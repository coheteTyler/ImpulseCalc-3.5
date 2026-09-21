# MESH_FIX_PLAN — knobs_preview / body_fitted_OH

**Label: PREDICTED.** Plan only. No remesh, no mesh edit, no solve, no η-code change in this step.

Case: `output/geom_tests/knobs_preview/openfoam_cases/knobs_preview`  
Mesh kind: `body_fitted_OH` · cells = 12934 · live residual **8.9%** at TE+0.2c  
Artifacts read: `log.checkMesh`, `README.txt`, `freezeB_promote.json`, `classic_cascade_promote.json`, `efficiency_wy_scan.json`, `efficiency_wy_compare_1p2e4.json`, `efficiency_PREDICTED.csv`, polyMesh (`points/faces/owner/neighbour/boundary`), `postProcessing/yPlus1`, `viewer/ofio.py` cell centres + yPlus parse.

---

## Verdict / diagnosis

**Vertical asymptote at the TE+0.2c station is the O–H seam / abrupt cell-size jump, not under-converged time.**

- Work residual at code station TE+0.2c: **0.08885 (8.9%)** (`efficiency_wy_compare_1p2e4.json` / `efficiency_PREDICTED.csv`). Wy3 lags force-implied (Wy3=−330 vs Wy3_force_implied=−448 m/s; gap≈118 m/s). `work_match_band=hard_fail`. Extend to t=1.2e−4 s moved Wy3 toward force but did **not** clear 8%.
- `efficiency_wy_scan.json`: best_offset residual ~0.14% at 0.5c offset — code station still >8%. Scan note: residual still >8% at code station; no remesh.
- More rhoCentralFoam time will not fix an 8.9% station residual whose sample line sits immediately downstream of a **~8.4:1** O–H size cliff at the TE collar (x≈0.00988 m). That cliff is a mesh defect (vertical interface column), visible as a vertical asymptote in field cuts near the sample x.

**Primary cause:** O-east collar ↔ H-dump / south–north H join at `x_TE_col≈0.00988–0.00999 m` with graded-jump failure (target ≤2:1). Secondary: TE O circumferential budget under ~40–50 in the near-TE collar window.

---

## Embedded mesh-reader results

Reader: `viewer/ofio.py` (points/faces/owner/neighbour, cell_centres) + boundary parse + face-adjacent half-width Δn across x-facing internal faces + `parse_yplus` / `yPlus1` FO table. Units as name (symbol).

| Quantity | Value | Notes |
|---|---|---|
| Cells (n_cells) | 12934 | all hex; matches meta / classic_cascade_promote |
| Blade wall faces (n_blade) | 148 | patch `blade0`; README n_around≈147 (JSON knob n_around=56 stale) |
| O radial layers (n_radial) | 20 | README / cfd |
| TE tip (x_TE, y_TE) | 0.009893 m, −0.001840 m | max-x on blade wall loop |
| **TE O-grid wall count SS / PS** | **17 / 15** (sum **32**) | faces with x > x_TE − 0.15 c along wall from TE tip; 0.25c window → 25 / 20 (sum 45). README OH sector n_east=18. **Below ~40–50 collar target** if read as near-TE budget |
| **O–H Δn ratio nearest TE** | **8.40 : 1** | x-facing faces at fx≈0.00988 m, TE-wake y band; half-widths Δn_coarse=0.0215 mm, Δn_fine=0.0026 mm → full Δx≈0.0430 mm / 0.0051 mm. Fine side matches README dn_o≈5.22 μm / dump first_Δx. **FLAG ≫ 2:1** |
| O–H max Δn ratio near TE band | 9.6–12.2 : 1 | north cyclic / cavity-adjacent vertical joins; same seam x |
| O–H V^{1/2} ratio (prism proxy at seam) | ~2.9 : 1 | same face; still >2 |
| Sample-line (TE+0.2c) x-facing Δn ratio | 1.18 : 1 | dump stretch only; asymptote is **upstream seam**, not dump packing |
| **Max aspect ratio wake strip** | AABB-xy **103.5** (TE→TE+0.5c); checkMesh global **541.82** | wake strip n_cells≈2404; neighbor-spacing AR spikes at seam (pathology of 8.4:1 jump) |
| checkMesh non-ortho max / avg | 74.60° / 15.60° | 14 faces >70° → set `nonOrthoFaces`; Mesh OK (failed_checks=0 this log) |
| checkMesh skew max | 2.208 | OK |
| **y+ on blade** | min **0.00989**, max **1.801**, avg **0.313** | `postProcessing/yPlus1` at t=1.2e−4 s (FO; volField yPlus was zeroed by bare postProcess). **Band used: laminar wall-resolve y+ ≲ 1–5.** Status: **in band** (max 1.8) |
| Domain height (H_y) | 0.008113917 m | bbox y ∈ [−0.004056958, +0.004056958] m |
| Geometry pitch (pitch_m) | 0.008113917 m | knobs_preview.json; freezeB pitch_period_m same |
| **H_y / pitch_m** | **1.000000** | cyclic period OK — **no pitch mismatch** |
| Dump packing (README) | n_dump=41, L=25.06 mm, first_Δx=0.0051 mm, last_Δx=3.826 mm | x_TE_col=0.00998649 m → x_out=0.03504 m |
| Soft H↔O note (README) | firstΔy_n / dn_o = **8.00** | already admits north join 8:1; measured seam confirms ~8.4:1 on south TE column too |
| Live residual context | 8.9% hard_fail; Wy3 lag; sample_on_wall=True; plateau=False | η dark; PREDICTED |

---

## Block layout sketch (TE vs TE+0.2c sample)

```
        y_top cyclic ─────────────────────────────
              H north fill (README firstΔy_n/dn_o=8)
                    │
     inlet H    ┌───┴─── O collar (n_radial=20) ──┐   H dump (n_dump=41)
     stems  ──► │  wall j=0 ············· TE tip  │══╗
                │         impulse U-cavity         │  ║  ← O–H SEAM (vertical)
                │              SS↑  TE·  ↓PS       │  ║     x≈0.00988–0.00999 m
                └──────────────────────────────────┘  ║     Δx ratio ≈ 8.4:1  FLAG
                         open-O east silhouette ──────╝
                                                      │
                         sample TE+0.2c ──────────────●  x=0.01189 m
                         (in dump H; Δx stretch≈1.18) │
                                                      ▼
                                                   outlet
        y_bot cyclic ─────────────────────────────

  LE cluster: min wall Δs≈0.024 mm at LE; le_cluster=2.5 (README).
  Seam column spans ≈ full south pitch strip at x_TE_col (measured
  high-R faces from y≈−4.0 mm up through TE wake); sample line is
  ~0.19c downstream of seam — field asymptote is the seam imprint.
```

---

## Fix order (execute later — not now)

1. **Smooth O–H transition** at TE collar / east silhouette. Target graded jump **≤2:1** in face-adjacent Δn (and V^{1/2}). Kill the vertical size cliff at x_TE_col (match dump first_Δx to O outer Δn on **both** south and north joins; README already flags north 8:1). Soften south TE column the same way.
2. **Bump TE O cells** if near-TE collar budget stays **<~40–50** (live SS+PS≈32 in 0.15c window; n_east=18). Raise circumferential density around TE tips without starving LE.
3. **Correct y+** only if off-band. Live max y+=1.80 ∈ laminar band ≲1–5 → **no change required** unless remesh coarsens wall first cell.
4. **LE clustering quick-check** after TE/OH edits: keep streamwise LE Δs bias (le_cluster=2.5); confirm no new non-ortho hole at LE stems (past builds: independent outer resample → O non-ortho).
5. **Periodicity** only if pitch mismatch. Live H_y/pitch=1.000 — **skip unless remesh breaks cyclic x-node share**.

---

## Gates before re-solve

Fail or regress → **stop, report, no solve.**

| Gate | Criterion |
|---|---|
| checkMesh | `failed_checks=0`; no new severe non-ortho cluster at O–H interface; no aspect regression vs live (global AR max 541.8 / wake AABB 103.5 as baseline — improve, don’t worsen at seam) |
| O–H jump | face-adjacent Δn ratio at TE collar seam **≤2:1** (re-run same polyMesh inspect) |
| Normals | Re-run existing normal-vector / face-pyramid orientation check (past builds flipped normals on wake quads — `mesh.py` wake `flip_neg` guard). **Report pass/fail + max deviation explicitly.** Do not solve if fail |
| TE render | Re-render TE confirming clean graded O→H transition (no chalk-line vertical cliff) |
| Sample line | TE+0.2c sample still in **valid interior cells**, not on boundary / wall; keep station off the seam x |
| Residual expectation | Plan does not claim η; only that mesh gates pass before any foam |

---

## Out of scope (this plan)

- No solver run / no extend / no endTime tweak as a fix for 8.9%.
- No η-code edits; no tip-loss (2D slab).
- No HOH zipper / Gmsh / hybrid_OH_tri swap unless a later plan says so.
- All efficiency numbers remain **PREDICTED**.

---

## Falsify

If after smoothing the O–H jump to ≤2:1 (and TE cell bump if applied) the vertical asymptote / 8.9%-class station residual **remains**, the cause is **not** the O–H seam — **re-triage first** (BC, station definition, force–Wy closure, sample path) before another mesh pass or any solve-for-η.

---

ready to execute on your go.
