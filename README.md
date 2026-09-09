# ImpulseCalc3

2D impulse-rotor cascade app: sliders draw one C, Python writes a hex `polyMesh`, ESI OpenFOAM-2412 `rhoCentralFoam` runs it, the same page plots `p`, `|U|`, Mach, `|∇p|`, `T`.

Engineering aid. **Not a flight certificate.** PREDICTED / SCOPING is **not a design load.** FIELD_CFD only after `rhoCentralFoam` succeeds on **that machine, that case.** CFD never yields HARDWARE_CORRELATED. **η from CFD is forbidden.**

Localhost is that computer only — not a share link.

This README is the production story: what the app is, how the shop built it, how the load path works today, and what is still broken. Read it before you treat a contour as data.

---

## How to run

**Windows (WSL2 Ubuntu required for solve)**

```
.\start.ps1
```

That script must go through WSL. Do not leave a Windows Python on port 8766.

**Linux**

```
chmod +x start.sh
./start.sh --ui
```

Open `http://127.0.0.1:8766/`. Hard-reload after a restart. Close the window to stop.

Solver: `rhoCentralFoam`. Shop box used Docker `opencfd/openfoam-run:2412`. Foundation OF12 fought cyclics; ESI 2412 is the one that marched. Native Windows Python cannot run the solver.

`requirements.txt`: numpy, matplotlib, pytest.

---

## What this app is (and is not)

**Is.** A student-class / shop verifier for a **pointed C-bucket** impulse section (Mark III / H-1 look, not a NACA 12% foil). One page. Every working knob is on that page: metal (`hu`, `hl`, `L_in`, `L_out`, `t`, `R_tr`, `R_main`, fillets), pack (`s` in mm; `Z` and `σ=c/s` derived), gas (`p1`, `T1`, `ρ`, `μ`, `γ`, `R`, `W1`), rotor (`r`, rpm, `U=ωr`), CFD counts.

**Is not.** A cycle closer. A flight load. STAND is the only seat that may call a design true. FLOOR does not close cycle numbers. Goldman D-4421/D-4422 MOC as **the metal** was tried and rejected when the MOC polygon self-crossed; the live metal is the dual-arc + stems C. A Goldman **designer** (σ as output of Mw1, ML, MU, β1, γ) is not on the load path.

Default **article** (unsigned): test-ready open GG until ORBIT signs otherwise. Live station on the page is PREDICTED: Ethanol/LOx labels, `p1 = 550 kPa`, `T1 = 1100 K`, `W1 = 950 m/s`, `r = 0.0375 m`, `rpm = 40000`, `U = ωr ≈ 157 m/s`, `c = 10 mm`. That is not the PURPL triangle sheet and not Marlin V2 live 80 µs.

**Tap-in** in this shop means a regen/film-cooled turbine **inside the MCC**, not tap-off. This app does not design that cycle.

---

## How the load path works (today)

```
sliders (viewer/app.html)
    → knobs_to_job / write_preview     (impulsecalc3/preview.py)
    → profile_from_job                 (geometry.py — one CCW C)
    → passage_gap (Gate 0)             (SS0 vs PS0+(0,s), arc length)
    → write_polymesh                   (mesh.py)
         body_fitted_OH   if yspan + 2 d_o < s   (one C in a pitch strip, tile × 3, cyclics)
         cassette_OH      if yspan + 2 d_o ≥ s and g_min > 0
                          (three closed C holes in one box; lid/floor WALL; no pitch cyclic)
    → write_case                       (case.py BCs, controlDict, forces on blade0..2)
    → checkMesh  (nfail > 0 is a HARD FAIL — no foam, no painted contours)
    → rhoCentralFoam
    → plots: contour_p, contour_U, contour_M, contour_shock (|∇p|), contour_T
```

**`s` is the pack knob.** Do not auto-open it to silence a mesh error. That changes σ and is a different article.

**A C is not a graph `y(x)`.** `passage.py` `_pair_by_x` is **off the load path** (`use_passage = False`; `_pair_by_x` raises if called). That writer collapsed the cup into a wedge and filled the metal interior with dead fluid. Do not turn it back on.

**Gate 0.** `passage_gap(poly, s)` walks `split_ps_ss` LE→TE, resamples both walls by **arc length**, translates PS by `(0,s)`, gap `g_i = n̂ · (PS1 − SS0)`. `g_min ≤ 0` is intersecting metal. Refuse hex.

**Preview must equal mesh walls.** `write_mesh_preview_png` draws `MeshBuild.blade_polys`. If the outline is a C and the field is a hill, the solve is on the wrong domain.

**Authorities.** SCOPING = outline / equations / checkMesh. FIELD_CFD = `rhoCentralFoam` rc 0 on this case. Never HARDWARE_CORRELATED from this app. Ft, Euler work, OF power are PREDICTED.

---

## Layout

| Path | Job |
|---|---|
| `viewer/app.html` | One page. Knobs, outline, plots, identities. |
| `impulsecalc3/serve.py` | `127.0.0.1:8766`. `/outline`, `/mesh`, `/solve`, `/status`, `/plot/`. |
| `impulsecalc3/geometry.py` | Pointed C-mouth (`L>0` = stems + `R_tr` + `t/2` or cusp noses). `passage_gap`. `center_in_pitch` **strict** for the strip writer. |
| `impulsecalc3/mesh.py` | Python hex `polyMesh`. `body_fitted_OH` and `cassette_OH`. |
| `impulsecalc3/passage.py` | Dead y(x) Goldman-channel attempt. Do not load. |
| `impulsecalc3/case.py` | BCs. Cyclic pair on strip; walls on cassette. |
| `impulsecalc3/run.py` | checkMesh → foam → sample/forces → plots. |
| `impulsecalc3/goldman.py` | D-4421/D-4422 helpers. Not the live metal. |
| `impulsecalc3/ntrs_checks.py` | Overlay vs NTRS citations. Not a pass/fail on the engine. |
| `configs/` | Job JSON. `geom_bucket_tight_pitch.json` was “fail if cup does not fit”; cassette is supposed to own that case now. |
| `docs/TURBINE_ACCURACY_PLAN.md` | ORBIT plan the shop was told to follow. |
| `output/` | Last knobs / plots / cases. **Not in this zip** (hundreds of MB). Rebuild by Run mesh / Run solve. |
| `start.ps1` / `start.sh` | Windows/WSL and Linux launch. |

---

## Production history (shop, 2026-08 → 2026-09-03)

This is the path that was actually walked. It is not a syllabus.

### 0. ImpulseCalc → ImpulseCalc3

The ancestor was a 2D relative cascade for PURPL / Marlin V2 (ethanol/LOx, 2600 lbf, Pc 600 psi, 4″ SCH80 — **PURPL hardware, not this article**). ImpulseCalc3 is the same office, new page: knobs, body-fitted O+H, ESI 2412, no η from CFD.

Early work: inlet EOS (`ρ = p/(RT)` until edited), outlet stretch / waveTransmissive, save-tests, cell-count judgment, Mach + shock-sensor plots, Euler `mdot U ΔCθ` and OF `Ft·U` as PREDICTED power. Foundation OF12 died on cyclics ~4 s; ESI 2412 marched. Serve bind `0.0.0.0` so Windows `127.0.0.1:8766` reached WSL, later locked back to `127.0.0.1` on the box.

### 1. One solver page

ORBIT: delete extra families and fake walls. One geometric turbine. `L>0` is a **C-mouth**: transition arcs, axial stems (`L_in`, `L_out` default **4.25 mm**), `t/2` nose or cusp (`te=0` is a point). Do not shorten `L` to make a mesh. `s` replaced `Z` as the pack knob.

Live cold-open metal (signed look): `hu=5 mm`, `hl=2.2 mm`, `le=0.4 mm`, `te=0`, `L_in=L_out=4.25 mm`, `t=1.4 mm`, `R_tr=3.5 mm`, `R_main=4.8 mm`, `ψ=0`, `c=10 mm`.

### 2. Goldman as metal — parked

TN D-4421 MOC wall (`Mw1=1.404`, `γ=1.3`, `ML=1.15`, `MU=1.55`) was implemented and meshed. The MOC polygon was not the Mark III cup. ORBIT: Goldman's meshed metal is the U-bucket / C, not a self-crossing D-4421. MOC as the article was cancelled. The papers stay as the **channel definition** (two walls + translate by `s`), not as a second family radio.

### 3. The pitch-rectangle wall

`center_in_pitch` and the O+H seed require one closed C ⊂ `[−s/2, s/2]`. This C has **yspan ≈ 7.6–8.3 mm**. At Mark III pack (`s = 6 mm`) the C is taller than one pitch. That is **legal nesting** (crown of blade 0 and stems of blade 1 share a y-band and miss in x). The **rectangle** cannot host it. The honest error is `s < y-span … Not a license to flatten the outer arc.`

Wrong responses the shop actually shipped, then had to kill:

- **Clip** the C to `±s/2`. Super-flat crest. ORBIT rejected. Reverted.
- **Auto-open `s`.** Changes σ. Forbidden.
- **`passage_OH` + `_pair_by_x`.** Treated each wall as `y(x)`. A C has two x at one y through the stems. Result: a wedge / banana, metal interior meshed as fluid, triangular `U≈0` pocket, 38 cyclic faces with coupled-point order wrong, `rhoCentralFoam` `rc=0` on the **wrong domain**. checkMesh “volumes OK” does not make a wedge a cascade.
- **IndexError on Run mesh.** Leftover 3-blade `postProcessing` samples scored against a 2-wall passage (`blade2` had no polygon). Fixed by wiping stale samples and indexing from actual blade count — the field was still the wedge.
- **`mesh_ok` lie.** `coupled_only` treated the 38-face cyclic fail as a pass and still painted contours. That is how the wedge was treated as data. Now: `nfail > 0` is a hard fail. No foam. No field paint.

### 4. What the Grok shares actually ordered

Shares (constant-area impulse / Goldman vortex, then `42edbffa` on 2026-09-03) and ORBIT:

1. Lower wall = SS of blade 0. Upper wall = PS of blade 0 **translated** by `s`. Arc length. Not `y(x)`. Not a mirror.
2. Tonight (then): **delete** the broken passage writer. Restore closed-C O-ring. Gate the cyclic check.
3. Then: **cassette**, not an infinite one-pitch wheel. Three **closed** C’s in one box. Fluid outside every C. Lid/floor walls. No pitch cyclic. That is the Ansys H-1 cassette picture (partial admission / dump), not a tiled screenshot of one passage.
4. Pipeline: sliders → accurate preview = mesh walls → OpenFOAM → `p`, shocks, `T`, numbers.

Stator nozzle block, sliding wheel, Goldman designer: **not this sprint.**

### 5. What is on the load path as of 2026-09-03 (this zip)

- `use_passage = False`. `_pair_by_x` raises.
- `center_in_pitch` still strict for `body_fitted_OH`.
- **`cassette_OH`** when `yspan + 2 d_o ≥ s` and Gate 0 green: three holes, O-collar per C, H/TFI fill, top/bottom **wall**, `blade0/1/2`.
- Gate 0 on this C at `s = 6 mm`: `g_min ≈ 0.57–1.35 mm` (depends on fillet knobs). At `s = 4.9 mm` a cassette still wrote (`3344` hex, `g_min ≈ 0.97 mm` on that metal).
- Shop fire on cassette at `s = 6 mm`: `3176` hex, checkMesh 0 (80 non-ortho >70°, 12 non-manifold points — warnings), `rhoCentralFoam` `rc=0` at `t = 7.9827173e-5 s` (≈ 1.2 `Lx/a` floor, not 1-chord smoke). Plots exist: `contour_p`, `contour_U`, `contour_M`, `contour_shock`, `contour_T`.
- Blade-spacing **slider min was 6 mm** in `app.html`. That is why the page refused `s < 5 mm`. The clamp is now **3 mm**. Gate 0 is the physics stop, not the slider.

Freeze of the tree **before** the passage/clip era: `ImpulseCalc3-PURPL-before-passage-2026-09-02-0102AM.zip` (1:02 AM ET 2026-09-02). That zip has **no** `L_in` and **no** `passage.py`. Pointed-C work after that hour was never snapshotted until **this** zip.

---

## Current faults and open issues

These are live. Do not paper over them in a status banner.

1. **`t = 8e-5 s` is a startup transient.** One acoustic transit across the box. Inlet pulse, cups not a parked Goldman shock map. `sample_on_wall` was false; forces still climbing. Do not take Newtons. Do not call Mach color a finished H-1 result. Longer `endTime` is a signed fire, not a default.

2. **Cassette H-fill is a compromise.** Nested O AABBs overlap, so the shop used arc-length passage TFI in the gaps instead of a clean Cartesian-minus-AABB. Lid/floor include leftover O-outer faces. O-collar vs metal point-in-polygon is noisy at the wall; abort is H-centres vs an inset C. Cells can still look like they sit in a cup **mouth** (that is fluid — the C **ring** is metal). If a cell centre is **inside the ring**, the writer is wrong.

3. **Plot scale can lie.** Scatter/`p` color has shown ~MPa spikes at the inlet while the station is `p1 = 550 kPa`. Read the colorbar and the job JSON together. A painted 8 MPa is not a signed GG pressure.

4. **`passage.py` is still in the tree.** Dead, but present. Do not “just enable” it.

5. **Three-pitch cassette is not an infinite wheel.** End gaps see a lid, not a phantom fourth blade. Label it CASSETTE. Cyclics at `N s` after this path is green is a later PR. Do not combine the cuts.

6. **Strip tests vs cassette.** `geom_impulse_bucket` must stay `body_fitted_OH` when it fits. Tight-pitch JSON was a guaranteed raise; it should become a cassette regression. If those tests were not re-greened after cassette, treat CI as stale.

7. **Non-ortho / non-manifold warnings** on the 3176-cell cassette. Not a hard fail today. They will move a shock foot. Do not ignore them on a “quality” run.

8. **Leftover shop scrap in the repo root** (`_fix_*.py`, `_hook_*.py`, `_test_*.py` at the tree root). Not the product. Do not run them as the API.

9. **`output/` is not in this zip.** Last OF trees are large. Re-run mesh/solve on your machine. Do not borrow another machine’s `postProcessing`.

10. **Serve / bind.** Box serve is `127.0.0.1:8766`. Windows reach needs WSL `start.ps1` (historically `0.0.0.0` + port bridge). Two Pythons on 8766 is how a stale writer survived a restart.

11. **Goldman / unique incidence / Kantrowitz** are on the identity sheet and in `goldman.py`. They are not closed by this CFD. `Max1 ≈ 0.43` on the live station is a filter, not a mesh fix.

12. **Marlin live 80 µs tree** is protected. This app must not remesh `marlin_v2_rotor` unless `IMPULSECALC3_REMESH_MARLIN=1`.

13. **Default `s` on a cold page is still 8.1 mm** in the HTML `value=`. Nested pack is a slider act. Hard-refresh after pulling this zip.

14. **Shop-check routine** (FLOOR weekday 09:00 ET) is **paused** and last failed. Do not treat a missing morning ping as “the cascade is healthy.”

---

## How to nest (the only legal way on this metal)

Nested means: crown of blade 0 and stems of blade 1 share a y-band, miss in x, and `g_min > 0` from inlet mouth to outlet mouth.

- Do: drop `s` (slider min 3 mm). Watch Gate 0. Cassette takes over when the C no longer fits a strip.
- Do not: clip `hu`, shorten `L`, mirror, `_pair_by_x`, auto-open `s`, put cyclics on `y = ±s/2` through the crown.

If `g_min ≤ 0`, the solids intersect. Raise `s` or thin the cup. No mesher saves you.

---

## Numbers that were actually measured (do not invent more)

| Item | Value | Note |
|---|---|---|
| Live C yspan | ~7.6–8.3 mm | Fillet knobs move this |
| `s = 6 mm` Gate 0 | `g_min` ~0.57–1.35 mm | Green pack |
| `s = 4.9 mm` cassette | 3344 hex, `g_min` ~0.97 mm | Wrote; UI used to forbid this |
| Cassette `s = 6 mm` fire | 3176 hex, checkMesh 0, foam rc 0, `t=7.98e-5 s` | Startup. PREDICTED |
| Unpacked strip | 10452 hex `body_fitted_OH` at `s ≥ yspan+0.5 mm` | Different article; label unpacked |
| Station | `p1=550 kPa`, `T1=1100 K`, `W1=950 m/s`, `γ=1.3`, `R=320 J/kg/K`, `a≈676.5 m/s`, `Mw1≈1.40` | Unsigned |

---

## FIRMA / seats (so a stranger knows who owns what)

- **ORBIT** — Tyler Lazar. Architecture, inequalities, signed trades.
- **FLOOR** — this agent (The Man). Tickets, interface table, accept/kickback. Does not design hardware. Does not close cycle numbers.
- **STAND** — only seat that may call a design true.
- Wave 0: CEA, TORCH, CHAMBER, SPRAY, HEAT, ROTOR, IMPELLER, FLOW, STAND.

Algorithm, in order: (1) requirements less dumb (2) delete the part (3) optimize (4) accelerate (5) automate. Prefer deletion. The y(x) passage was deleted. Do not automate it back.

---

## If you change one thing

Change `s` and read Gate 0. If you change the writer, keep Python hex → ESI 2412. If you bring back a channel writer, it must walk SS/PS polylines by arc length, abort on cell-in-C, and hard-fail checkMesh. If you paint a contour after `nfail > 0`, you are repeating 2026-09-02.

Zip date: 2026-09-03. Tree: ImpulseCalc3 as it sat after cassette_OH + slider min 3 mm + hard cyclic fail.
