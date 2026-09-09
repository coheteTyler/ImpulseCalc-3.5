# ImpulseCalc3 — more accurate turbine sections (rendition 2)

Article: 2D cascade verifier. PREDICTED until STAND. No η from CFD. Unsigned gas.

## Why the current metal is the accuracy hole

Today’s outline is a circular-arc camber with NACA thickness and circular LE/TE caps (`geometry.py`). That is a knob-friendly foil, not a supersonic impulse passage. CFD on the wrong wall cannot save a loss model. Accuracy starts by drawing the passage Goldman actually designed.

Do not jump to 3D, Gmsh, or a second solver until this 2D section exists.

## Target metal (Goldman vortex-flow impulse)

Source of method: Goldman, NASA TN D-4422, NTRS 19680010807 (and the MOC program line in NTRS 19720005326).

The passage has three parts:

1. Inlet transition arcs (method of characteristics) — take uniform parallel inlet flow and turn it into vortex flow. Lower arc drops Mach from inlet Mw1 toward a chosen concave-surface Mach. Upper arc raises Mach toward a chosen convex-surface Mach.
2. Concentric circular arcs — vortex turn at constant surface Mach on each wall.
3. Outlet transition arcs (MOC) — reverse: vortex back to uniform parallel exit.

Impulse means |W| in ≈ |W| out in the design intent. The OF |W2|/|W1| we already print is the check, not η.

Inputs (unsigned until TORCH/CEA sign gas): Mw1, Mw2, γ, β1, β2, concave Mach, convex Mach.

Bézier is a construction handle for those MOC polylines so a human can tug a control point without rewriting characteristics. It is not a second family.

Stator walls later: same MOC idea for a nozzle (uniform → vortex or a simple Prandtl-Meyer fan), then the rotor section. Same strip, two rows, not a 3D wheel.

Boundary-layer correction (displacement thickness added to the isentropic wall) is rendition 2b, after the isentropic Goldman outline meshes and a solve plateaus. NTRS 19720005326.

## Mesh

Keep Python → polyMesh O+H. The Goldman polygon replaces `closed_profile` / bucket as family `goldman_impulse`. Offset-O wraps the new outline. Do not add Gmsh to get accuracy.

Dump stays 6 chords, waveTransmissive, fieldInf = p1.

## Accept (STAND)

Unchanged: mesh_ok_strict, wall p = O(p1), t_end ≥ max(5 c/W1, 1.2 Lx/a), |dFt/Ft| last chord < 0.05, Ft·U after plateau only, no η from CFD, no G4.

## Order (Algorithm)

1. Requirements: 2D Goldman rotor section in the existing strip. Not 3D. Not a new cycle.
2. Delete: NACA-on-camber as the live default once Goldman meshes. Keep foil as a debug family.
3. Then optimize the MOC point count. Then accelerate. Then automate a section table.

## Tickets

- G-01 ROTOR: Goldman outline writer (transition + circular + transition) from Mw, γ, β. Artifact: closed polygon. Stop: self-intersect or open.
- G-02 FLOOR: family `goldman_impulse` in knobs + inlet modal already feeds Mw1. Residual PREDICTED.
- G-03 STAND: fire the new outline on the same 2412 rhoCentralFoam path. Same gates.
- G-04 CEA: which of (p,T,R) vs Mw1 is the Goldman input lock. Do not silent-close.
- G-05 later: stator MOC row, BL displacement, 3D.

