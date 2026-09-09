# ImpulseCalc 3.5 — Beginner install guide

Get to **Inputs → Update → profile** on one machine path first. Mesh and OpenFOAM come later.

**What this app is:** a local engineering aid for a 2D impulse-rotor cascade.
**What it is not:** a flight certificate. Numbers stay **PREDICTED / SCOPING** until a trusted wall-force plateau and later hardware correlation. This guide never treats CFD efficiency (η) or an unsigned spouting velocity (C₀) as design truth.

---

## Happy path (do this first)

Use **WSL Ubuntu** (Windows) or **native Linux**. That is the path this guide leads with.

1. Clone and enter the repo:

```bash
git clone https://github.com/coheteTyler/ImpulseCalc-3.5.git
cd ImpulseCalc-3.5
```

2. Create and activate a Python environment, then install packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. Start the UI:

```bash
chmod +x start.sh
./start.sh --ui
```

On Windows, from PowerShell in the repo folder you can instead run `.\start.ps1` — that script starts the app **inside WSL**. Do not leave a pure Windows Python on port 8766 if you plan to solve later.

4. Open `http://127.0.0.1:8766/` (port **8766**). If the page looks stale, hard-reload once (`Ctrl+Shift+R`).

5. Click **Inputs**. Edit values. Click **Update**. Confirm the central profile strip and cascade outline change on the main page.

That is day one. Stop here until that works.

---

## 1. What you need

| Item | UI + Inputs | Mesh / solve |
|------|-------------|--------------|
| Git | Yes | Yes |
| Python 3.10+ (in WSL or Linux) | Yes | Yes |
| Packages in `requirements.txt` | Yes | Yes |
| WSL2 Ubuntu or native Linux | Yes (recommended) | Yes |
| OpenFOAM ESI **2412** (`rhoCentralFoam`) | No | Yes |

Pure Windows Python can sometimes open the page, but it is not the supported path. Prefer WSL.

---

## 2. Inputs → Update → profile

1. On the main page, click **Inputs**. A second window opens (`/filters.html`) so you can put it on another monitor.
2. Left tabs filter which fields you see:
   - **Inlet gas** — post-stator working fluid (no stator metal in this article). Visible labels include inlet total pressure, inlet total temperature, mass flow, γ, cp, molar mass, relative inlet angle (β1), relative inlet speed (W1).
   - **Size** — mean diameter, chord, blade count, solidity, admission, wheel speed, target shaft power (SCOPING).
   - **Metal** — rotor fillets and profile knobs (for example upper/lower sagitta, LE/TE fillet, inlet/outlet straight length).
3. Optional: the **Smart filter** dropdown narrows the list further (for example "Horizontal size" shows diameter, blade count, solidity, and related size fields). Skip it if the tabs are enough.
4. Edit values, then click **Update** at the top.
5. Back on the main page, the **central profile** strip updates and the outline refreshes.

If the popup is blocked, allow popups for `127.0.0.1`, or open `http://127.0.0.1:8766/filters.html` directly.

**Authority:** the profile is **PREDICTED / SCOPING** — useful for sizing and mesh prep, not a signed load.

---

## 3. Mesh and solve (optional)

Only after Inputs → Update works.

1. Install OpenFOAM ESI 2412 (or use Docker image `opencfd/openfoam-run:2412` if that is how your machine is set up).
2. Confirm `rhoCentralFoam` is on PATH in the same environment that runs the app.
3. In the UI: Update profile → **Run mesh** → wait for a clean mesh.
4. **Run solve** and pick an `endTime` from the menu (shortest option still lets flow cross the passage).
5. Watch the job bar. Leave shaft power **dark** until wall force plateaus.

**Gates (read before you trust numbers)**

- `checkMesh` must report **Failed 0** before you trust a solve.
- Mid-run Mach or pressure-gradient pictures are transient only.
- Accept PREDICTED force × tip speed only after `sample_on_wall` / Ft is flat — not while it is still climbing.
- Do not invent CFD η as a cycle knob.

---

## 4. If something breaks

| Symptom | Likely fix |
|---------|------------|
| `python3: command not found` | Install Python 3.10+ in WSL Ubuntu or Linux |
| `pip` fails on `requirements.txt` | Activate `.venv`, then `python -m pip install -r requirements.txt` |
| Page will not load | Confirm the start script is still running; open exactly `http://127.0.0.1:8766/` |
| Port already in use | Stop the old ImpulseCalc process holding `8766` |
| Inputs popup blank / blocked | Allow popups; open `/filters.html` directly |
| Mesh / solve buttons fail | OpenFOAM not on PATH inside WSL; UI-only still works |
| Clone auth errors | Log into GitHub in the browser, use HTTPS clone, or set up SSH keys |

---

## 5. What stays on your machine

```
ImpulseCalc-3.5/
  INSTALL.md          <- you are here
  README.md           <- deeper shop story / load path
  requirements.txt
  start.sh / start.ps1
  viewer/             <- main page + Inputs popup
  configs/            <- example job JSON
  output/             <- local runs (created when you run)
```

---

## 6. Install-day labels

- **PREDICTED / SCOPING** — early analysis and outline intent. Not FIELD_CFD. Not a design load.
- **Power dark** — do not treat mid-fire screenshots or a climbing wall force as shaft power.

---

## 7. After install works

1. In **Size**, change mean diameter and chord, hit **Update**, confirm the outline moves.
2. In **Inlet gas**, change relative inlet speed (W1) and relative inlet angle (β1), Update again.
3. Only then attempt mesh on WSL/Linux.
4. Read `README.md` for the longer load path and known limits.

If install still fails, note your OS (Windows+WSL vs Linux), the exact command you ran, and the full error text.
