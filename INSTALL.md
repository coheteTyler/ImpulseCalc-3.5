# ImpulseCalc 3.5 — Beginner install guide

This guide gets ImpulseCalc 3.5 running on your computer from a clean install.
No prior OpenFOAM experience required for the **Inputs → Update → profile** path.
Solver (mesh + CFD) needs Linux or WSL2.

**What this app is:** a local engineering aid for a 2D impulse-rotor cascade (metal + gas knobs → outline → optional OpenFOAM).
**What it is not:** a flight certificate. Numbers stay **PREDICTED / SCOPING** until a trusted wall-force plateau and later hardware correlation. This guide never treats CFD efficiency (η) or an unsigned spouting velocity (C₀) as design truth.

---

## 0. What you will have when done

1. The app open in your browser at `http://127.0.0.1:8766/`
2. An **Inputs** popup (filter groups: Inlet gas / Size / Metal)
3. An **Update** button that writes a central profile and refreshes the main page outline
4. Optional: mesh + solve on Linux/WSL if OpenFOAM ESI 2412 is installed

---

## 1. What you need

| Item | Required for UI + Inputs? | Required for mesh/solve? |
|------|---------------------------|--------------------------|
| Git | Yes (to download) | Yes |
| Python 3.10+ | Yes | Yes |
| pip packages in `requirements.txt` | Yes | Yes |
| Windows 10/11 + WSL2 Ubuntu **or** native Linux | UI can run on Linux/WSL | Yes (solver) |
| OpenFOAM ESI **2412** (`rhoCentralFoam`) | No | Yes |

**Windows note:** Do not leave a pure Windows Python holding port 8766 if you plan to solve. Use the provided `start.ps1` so the app runs **inside WSL**.

---

## 2. Download the repo

Open a terminal (PowerShell on Windows, or any shell on Linux/macOS) and run:

```bash
git clone https://github.com/<YOUR_GITHUB_USER>/ImpulseCalc-3.5.git
cd ImpulseCalc-3.5
```

Replace `<YOUR_GITHUB_USER>` with the account that owns the repo (shown on the GitHub page URL).

If you already have a zip download from GitHub:

1. Unzip it.
2. Open a terminal **in that folder** (`cd` into it).

---

## 3. Create a Python environment (recommended)

Keeps ImpulseCalc packages from colliding with other projects.

```bash
python3 -m venv .venv
```

Activate it:

**Linux / macOS / WSL**

```bash
source .venv/bin/activate
```

**Windows PowerShell (only if you are not using WSL for the UI)**

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

You should see `numpy`, `matplotlib`, and `pytest` install without errors.

---

## 4. Start the app (UI only)

### Linux or WSL Ubuntu

```bash
chmod +x start.sh
./start.sh --ui
```

### Windows (must go through WSL)

From PowerShell **in the repo folder**:

```powershell
.\start.ps1
```

That script is written to run inside WSL. If WSL is not installed yet, install **Ubuntu** from Microsoft Store, open Ubuntu once, then retry.

### Open the page

In Chrome or Edge, go to:

```
http://127.0.0.1:8766/
```

Hard-reload once (`Ctrl+Shift+R`) if the page looks stale after a restart.

You should see the ImpulseCalc page with metal knobs and a cascade outline. Close the terminal window (or stop the process) to shut the server down.

---

## 5. Use Inputs → Update → profile (new in 3.5)

1. On the main page, click **Inputs**.  
   A second window opens (`/filters.html`) so you can put it on another monitor.
2. Use the left tabs:
   - **Inlet gas** — uniform post-stator working fluid (no stator metal in this article)
   - **Size** — diameter, chord, blade count, admission, speed
   - **Metal** — rotor profile / fillets (ImpulseCalc mm knobs + RTS-style ratios)
3. Optional: pick a **Smart filter** (for example “Horizontal size”) to show only related knobs.
4. Edit values, then click **Update** at the top.
5. Back on the main page, the **central profile** strip updates, and the outline refreshes from the mapped knobs.

If the popup is blocked, allow popups for `127.0.0.1` and click **Inputs** again.

**Authority labels:** the profile is **PREDICTED / SCOPING**. That means useful for sizing and mesh prep, not a signed load.

---

## 6. Mesh and solve (optional, Linux/WSL + OpenFOAM)

Only after the UI path works.

1. Install **OpenFOAM ESI 2412** (or use the shop Docker image `opencfd/openfoam-run:2412` if that is how your machine is set up).
2. Confirm `rhoCentralFoam` is on your PATH inside the same environment that runs the app.
3. In the UI: set knobs / Update profile → **Run mesh** → wait for a clean mesh.
4. Gate: `checkMesh` must report **Failed 0** before you trust a solve.
5. **Run solve** and pick an `endTime` from the menu (the shortest option is still long enough for flow to cross the passage — not a one-chord smoke test).
6. Watch the job bar. Shaft power stays **dark** until wall force (`sample_on_wall` / Ft) **plateaus**. Mid-run Mach or |∇p| pictures are transient only.

**Hard rules (do not skip):**

- Do not solve on a mesh that fails `checkMesh`.
- Do not invent CFD η as a cycle knob.
- Do not treat mid-fire screenshots as settled power.

---

## 7. Quick checks if something breaks

| Symptom | Likely fix |
|---------|------------|
| `python3: command not found` | Install Python 3.10+; on Windows use WSL Ubuntu |
| `pip` fails on `requirements.txt` | Activate `.venv`, then `python -m pip install -r requirements.txt` |
| Page won’t load | Confirm the start script is still running; open exactly `http://127.0.0.1:8766/` |
| Port already in use | Stop the old ImpulseCalc process, or find what holds `8766` and close it |
| Inputs popup blank / blocked | Allow popups; open `http://127.0.0.1:8766/filters.html` directly |
| Mesh/solve buttons fail | OpenFOAM not on PATH inside WSL; UI-only still works |
| `gh` / git auth errors when cloning | Log into GitHub in the browser, use HTTPS clone, or set up SSH keys |

---

## 8. Folder map (orientation)

```
ImpulseCalc-3.5/
  INSTALL.md              ← you are here
  README.md               ← deeper shop story / load path
  requirements.txt
  start.sh / start.ps1
  impulsecalc3/           ← Python package (geometry, mesh, serve, filters)
    rts_filters.py        ← RTS-keyed Inlet gas / Size / Metal schema
    serve.py              ← local UI server + /api/profile/*
  viewer/
    app.html              ← main page
    filters.html          ← Inputs popup
  configs/                ← example job JSON
  output/                 ← local runs (created on your machine)
```

---

## 9. Safety labels (read once)

- **PREDICTED** — analysis without hardware correlation.
- **SCOPING** — 0D / outline / early mesh intent, not FIELD_CFD.
- **FIELD_CFD** — only after a successful solve on that machine and that case, with forces only after wall plateau.
- **η from CFD** — forbidden as a design / cycle input in this project.
- **Unsigned C₀** — do not paste guestimate spouting velocities into the guide or the profile as truth.

When in doubt: use Inputs → Update for geometry and gas setup; leave power dark until Foamy / your foam run reports a flat Ft.

---

## 10. Next steps after install

1. Change **Size → mean diameter** and **chord**, hit **Update**, confirm the outline moves.
2. Change **Inlet gas → W1 / β1**, Update again.
3. Only then attempt mesh on WSL/Linux.
4. Read `README.md` for the longer load-path and known limits.

If install still fails, note your OS (Windows+WSL vs Linux), the exact command you ran, and the full error text — that is enough for someone to debug with you.
