"""Mesh preview from the real writer. Knobs → JSON → polyMesh + PNG.

No synthetic Cp. No eta. Never writes the live Marlin 80 µs tree.
Knobs remesh the outline without opening OpenFOAM. Building a case is not CFD.
Authority of a preview PNG is always SCOPING.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

from .case import job_tree_held_by_other, write_case
from .geometry import apply_constant_passage_width, apply_metal_bounds, fit_pitch_to_metal, passage_gap, polygon_signed_area, profile_from_job, safe_profile_from_job, spec_from_job
from .job import (
    ROOT,
    FORMAT,
    is_live_marlin_job,
    pitch_m,
    validate_job,
)
from .meanline import compute_meanline
from .ntrs_checks import evaluate as ntrs_evaluate
from viewer.plots import triangle_png
from .times import compute_times
from .job import domain_x

TEMPLATE_CUP = ROOT / "configs" / "geom_impulse_bucket.json"
TEMPLATE_FOIL = ROOT / "configs" / "geom_foil.json"
TEMPLATE_POINTS = ROOT / "configs" / "geom_points.json"
APP_OUTPUT = "output/geom_tests/knobs_preview"
APP_NAME = "knobs_preview"

AUTHORITY_SCOPING = "SCOPING"
AUTHORITY_FIELD_CFD = "FIELD_CFD"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _family_key(raw: Any) -> str:
    s = str(raw or "").strip().lower()
    if s in ("foil", "airfoil", "naca", "circular_arc", "circular_arc_camber_metal_angles"):
        return "foil"
    if s in ("points", "profile_points"):
        return "points"
    if s in ("goldman", "goldman_vortex", "vortex_impulse"):
        return "goldman"
    return "cup"


def template_path_for(family: str) -> Path:
    fam = _family_key(family)
    if fam == "foil":
        return TEMPLATE_FOIL if TEMPLATE_FOIL.is_file() else (ROOT / "configs" / "marlin_v2_rotor.json")
    if fam == "points":
        return TEMPLATE_POINTS
    return TEMPLATE_CUP


def _template(family: str = "cup") -> dict[str, Any]:
    return _read_json(template_path_for(family))


def triangles_from_meanline(ml: Any) -> dict[str, Any]:
    """SCOPING / PREDICTED velocity-triangle readouts. Meanline, not OpenFOAM. Not η."""
    return {
        "c1_m_s": float(ml.c1_m_s),
        "c2_m_s": float(ml.c2_m_s),
        "w1_m_s": float(ml.w1_m_s),
        "w2_m_s": float(ml.w2_m_s),
        "alpha1_abs_deg": float(ml.alpha1_abs_deg),
        "alpha2_abs_deg": float(ml.alpha2_abs_deg),
        "beta1_flow_deg": float(ml.beta1_flow_deg),
        "beta2_flow_deg": float(ml.beta2_flow_deg),
        "u_m_s": float(ml.u_m_s),
        "euler_work_j_kg": float(ml.euler_work_j_kg),
        "Mw1": float(ml.Mw1),
        "Max1": float(ml.Max1),
        "Re_c": float(ml.Re_c),
        "psi": float(ml.psi),
        "phi": float(ml.phi),
        "U_over_C1": float(ml.U_over_C1),
        "unique_incidence_active": bool(ml.unique_incidence_active),
        "mdot_open_kg_s": float(ml.mdot_open_kg_s),
        "mdot_gg_kg_s": ml.mdot_gg_kg_s,
        "blockage_implied": ml.blockage_implied,
        "predicted": True,
    }


def knobs_to_job(knobs: dict[str, Any] | None = None, *, template: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a geometry-test job from knobs. Gas/engine copied from the family template
    so the metal is the variable — not a second Pc/W1 article.
    """
    k = dict(knobs or {})
    if isinstance(k.get("geometry"), dict):
        k = {**k, **k["geometry"]}
    # One solver: pointed-tip impulse bucket (default) or circular-arc foil. Not MOC.
    fam = _family_key(k.get("family") or k.get("profile_family") or "cup")
    base = copy.deepcopy(template or _template(fam))
    base["format"] = FORMAT
    base["name"] = APP_NAME
    base["geometry_test"] = True
    base["predicted"] = True
    base["output_dir"] = APP_OUTPUT
    base["article"] = (
        "KNOBS PREVIEW — geometry test. Engineering aid, not a flight certificate. "
        "Not Marlin V2 live tree. Gas unsigned. PREDICTED is not a design load."
    )
    g = base.setdefault("geometry", {})
    for src, dst in (
        ("chord_m", "chord_m"),
        ("c", "chord_m"),
        ("chord", "chord_m"),
        ("solidity", "solidity"),
        ("sigma", "solidity"),
        ("n_blades_machine", "n_blades_machine"),
        ("Z", "n_blades_machine"),
        ("mean_radius_m", "mean_radius_m"),
        ("rm", "mean_radius_m"),
        ("r", "mean_radius_m"),
        ("rotor_tip_radius_m", "rotor_tip_radius_m"),
        ("rt", "rotor_tip_radius_m"),
        ("hub_radius_m", "hub_radius_m"),
        ("rh", "hub_radius_m"),
        ("span_m", "span_m"),
        ("span", "span_m"),
        ("beta1_flow_deg", "beta1_flow_deg"),
        ("beta1", "beta1_flow_deg"),
        ("beta2_flow_deg", "beta2_flow_deg"),
        ("beta2", "beta2_flow_deg"),
        ("upper_sagitta_c", "upper_sagitta_c"),
        ("upper_h", "upper_sagitta_c"),
        ("hu_c", "upper_sagitta_c"),
        ("hu", "upper_sagitta_c"),
        ("lower_sagitta_c", "lower_sagitta_c"),
        ("lower_h", "lower_sagitta_c"),
        ("hl_c", "lower_sagitta_c"),
        ("hl", "lower_sagitta_c"),
        ("thickness_c", "thickness_c"),
        ("t_c", "thickness_c"),
        ("le_fillet_r_c", "le_fillet_r_c"),
        ("le", "le_fillet_r_c"),
        ("le_radius_c", "le_radius_c"),
        ("te_fillet_r_c", "te_fillet_r_c"),
        ("te", "te_fillet_r_c"),
        ("te_radius_c", "te_radius_c"),
        ("stagger_deg", "stagger_deg"),
        ("stagger", "stagger_deg"),
        ("incidence_deg", "incidence_deg"),
        ("deviation_deg", "deviation_deg"),
        ("n_profile_points", "n_profile_points"),
        ("packing_driver", "packing_driver"),
        ("profile_points", "profile_points"),
        ("points", "profile_points"),
        ("upper_sagitta_m", "upper_sagitta_m"),
        ("lower_sagitta_m", "lower_sagitta_m"),
        ("le_fillet_r_m", "le_fillet_r_m"),
        ("te_fillet_r_m", "te_fillet_r_m"),
        ("hu_mm", "hu_mm"),
        ("hl_mm", "hl_mm"),
        ("le_mm", "le_mm"),
        ("te_mm", "te_mm"),
    ):
        if src in k and k[src] not in (None, ""):
            g[dst] = k[src]
    # millimetres → metres (absolute metal; chord is not a zoom)
    if k.get("hu_mm") not in (None, ""):
        g["upper_sagitta_m"] = float(k["hu_mm"]) * 1e-3
    elif any(k.get(s) not in (None, "") for s in ("hu_c", "hu", "upper_sagitta_c", "upper_h")):
        g.pop("upper_sagitta_m", None)
        g.pop("hu_mm", None)
        g.pop("upper_sagitta_mm", None)
    if k.get("hl_mm") not in (None, ""):
        g["lower_sagitta_m"] = float(k["hl_mm"]) * 1e-3
    elif any(k.get(s) not in (None, "") for s in ("hl_c", "hl", "lower_sagitta_c", "lower_h")):
        g.pop("lower_sagitta_m", None)
        g.pop("hl_mm", None)
        g.pop("lower_sagitta_mm", None)
    if k.get("le_mm") not in (None, ""):
        g["le_fillet_r_m"] = float(k["le_mm"]) * 1e-3
        g.pop("le_radius_c", None)
        g.pop("le_fillet_r_c", None)
    if k.get("te_mm") not in (None, ""):
        g["te_fillet_r_m"] = float(k["te_mm"]) * 1e-3
        g.pop("te_radius_c", None)
        g.pop("te_fillet_r_c", None)
    if k.get("lin_mm") not in (None, ""):
        g["lin_m"] = float(k["lin_mm"]) * 1e-3
    if k.get("lout_mm") not in (None, ""):
        g["lout_m"] = float(k["lout_mm"]) * 1e-3
    if k.get("r_tr_mm") not in (None, ""):
        g["r_tr_m"] = float(k["r_tr_mm"]) * 1e-3
    if k.get("r_main_mm") not in (None, ""):
        g["r_main_m"] = float(k["r_main_mm"]) * 1e-3
    if k.get("t_mm") not in (None, ""):
        g["t_m"] = float(k["t_mm"]) * 1e-3
    if k.get("psi_tr_deg") not in (None, ""):
        g["psi_tr_deg"] = float(k["psi_tr_deg"])
    if k.get("beta1_metal_deg") not in (None, ""):
        g["beta1_metal_deg"] = float(k["beta1_metal_deg"])
    if k.get("beta2_metal_deg") not in (None, ""):
        g["beta2_metal_deg"] = float(k["beta2_metal_deg"])
    if "beta" in k and k["beta"] not in (None, "") and "beta1" not in k and "beta1_flow_deg" not in k:
        b = float(k["beta"])
        g["beta1_flow_deg"] = b
        g["beta2_flow_deg"] = -b
    if g.get("le_radius_c") not in (None, ""):
        g["le_fillet_r_c"] = g["le_radius_c"]
    elif g.get("le_fillet_r_c") not in (None, ""):
        g.setdefault("le_radius_c", g["le_fillet_r_c"])
    if g.get("te_radius_c") not in (None, ""):
        g["te_fillet_r_c"] = g["te_radius_c"]
    elif g.get("te_fillet_r_c") not in (None, ""):
        g.setdefault("te_radius_c", g["te_fillet_r_c"])
    if fam == "foil":
        g["profile_family"] = "circular_arc_camber_metal_angles"
    else:
        g["profile_family"] = "impulse_bucket"
    g.pop("profile_points", None)
    g.pop("goldman", None)
    g["n_blades_cascade"] = 3  # mesh/validate lock; outline PNG stacks 5 for photo frame
    if k.get("stagger_deg") in (None, "") and k.get("stagger") in (None, ""):
        b1 = float(g.get("beta1_flow_deg") or 72.0)
        b2 = float(g.get("beta2_flow_deg") or -72.0)
        inc = float(g.get("incidence_deg") or 0.0)
        dev = float(g.get("deviation_deg") or 0.0)
        g["stagger_deg"] = 0.5 * ((b1 - inc) + (b2 + dev))
        g["beta1_metal_deg"] = b1 - inc
        g["beta2_metal_deg"] = b2 + dev
    if "packing_driver" in k:
        g["packing_driver"] = k["packing_driver"]
    if k.get("s_mm") not in (None, ""):
        g["pitch_m"] = float(k["s_mm"]) * 1e-3
        g["packing_driver"] = "pitch"
    if k.get("passage_depth_mm") not in (None, ""):
        g["passage_depth_m"] = float(k["passage_depth_mm"]) * 1e-3
    if k.get("pitch_m") not in (None, ""):
        g["pitch_m"] = float(k["pitch_m"])
        g.setdefault("packing_driver", "pitch")
    drv = str(g.get("packing_driver") or "")
    if drv in ("pitch", "spacing", "s_mm"):
        g["packing_driver"] = "pitch"
    base["geometry"] = g
    # Constant passage width: Goldman concentric walls (hl slaved; pitch if g_pass set).
    _cpw = k.get("constant_passage_width")
    if _cpw in (True, 1, "1", "true", "True", "yes", "on"):
        try:
            base, _cpw_rep = apply_constant_passage_width(base)
            g = base["geometry"]
            base["_constant_passage"] = {
                "enabled": True,
                "ok": bool(_cpw_rep.get("ok")),
                "hl_mm": _cpw_rep.get("hl_mm"),
                "pitch_mm": _cpw_rep.get("pitch_mm"),
                "slave": _cpw_rep.get("slave"),
                "passage_depth_mm": _cpw_rep.get("passage_depth_mm"),
                "g_mid_range_mm": (
                    None
                    if not _cpw_rep.get("after")
                    else float(_cpw_rep["after"].get("g_mid_range_m", 0) * 1e3)
                ),
                "g_mid_mean_mm": (
                    None
                    if not _cpw_rep.get("after")
                    else float(_cpw_rep["after"].get("g_mid_mean_m", 0) * 1e3)
                ),
                "note": _cpw_rep.get("note"),
            }
        except Exception as exc:
            base["_constant_passage"] = {"enabled": True, "ok": False, "error": str(exc)}
    gas = base.setdefault("gas", {})
    for src, dst in (
        ("w1_m_s", "w1_m_s"),
        ("W1", "w1_m_s"),
        ("w1", "w1_m_s"),
        ("blade_speed_u_m_s", "blade_speed_u_m_s"),
        ("U", "blade_speed_u_m_s"),
        ("u", "blade_speed_u_m_s"),
        ("p1_pa", "p1_pa"),
        ("p1", "p1_pa"),
        ("t1_k", "t1_k"),
        ("T1", "t1_k"),
        ("t1", "t1_k"),
        ("rho1_kg_m3", "rho1_kg_m3"),
        ("rho1", "rho1_kg_m3"),
        ("mu_pa_s", "mu_pa_s"),
        ("mu", "mu_pa_s"),
        ("gamma", "gamma"),
        ("r_specific_j_kg_k", "r_specific_j_kg_k"),
        ("R", "r_specific_j_kg_k"),
        ("r_specific", "r_specific_j_kg_k"),
    ):
        if src in k and k[src] not in (None, ""):
            gas[dst] = float(k[src])
    held = bool(k.get("rho_held"))
    gas["rho_held"] = held
    if not held:
        from .job import perfect_gas_rho_kg_m3
        p1 = float(gas.get("p1_pa") or 0)
        r = float(gas.get("r_specific_j_kg_k") or 0)
        t1 = float(gas.get("t1_k") or 0)
        if p1 and r and t1:
            gas["rho1_kg_m3"] = perfect_gas_rho_kg_m3(p1, r, t1)
    rpm = None
    for key in ("rpm", "N_rpm", "shaft_rpm"):
        if key in k and k[key] not in (None, ""):
            rpm = float(k[key])
            break
    rm = float(g.get("mean_radius_m") or 0.0)
    if rpm is not None and rpm > 0 and rm > 0:
        import math
        gas["blade_speed_u_m_s"] = rpm * 2.0 * math.pi / 60.0 * rm
        gas["rpm"] = rpm
    if "mdot_engine_kg_s" in k and k["mdot_engine_kg_s"] not in (None, ""):
        gas["mdot_engine_kg_s"] = float(k["mdot_engine_kg_s"])
    if "mdot_gg_kg_s" in k and k["mdot_gg_kg_s"] not in (None, ""):
        gas["mdot_gg_kg_s"] = float(k["mdot_gg_kg_s"])
    if "OF" in k and k["OF"] not in (None, ""):
        base.setdefault("engine", {})["OF"] = float(k["OF"])
    if k.get("t_end_s") not in (None, ""):
        base.setdefault("cfd", {})["end_time_s"] = float(k["t_end_s"])
    # App / knobs jobs: six-chord dump after TE, denser outlet, waveTransmissive fieldInf=p1.
    # Templates may still list x_dn_c=2 / n_outlet=14; do not rewrite marlin_v2_rotor.json.
    # OF / mdot copy engine knobs only — they must not rewrite p1.
    cfd = base.setdefault("cfd", {})
    for src, dst, cast in (
        ("n_around", "n_around", int),
        ("n_radial", "n_radial", int),
        ("n_inlet", "n_inlet", int),
        ("n_outlet", "n_outlet", int),
        ("n_cyclic", "n_cyclic", int),
        ("x_up_c", "x_up_c", float),
        ("x_dn_c", "x_dn_c", float),
        ("stretch", "stretch", float),
        ("inlet_stretch", "inlet_stretch", float),
        ("le_cluster", "le_cluster", float),
        ("n_le", "n_le", int),
    ):
        if k.get(src) not in (None, ""):
            cfd[dst] = cast(k[src])
    if k.get("x_up_c") in (None, ""):
        cfd["x_up_c"] = 1.5
    if k.get("x_dn_c") in (None, ""):
        cfd["x_dn_c"] = 6.0
    if k.get("n_outlet") in (None, ""):
        cfd["n_outlet"] = 28
    cfd["outlet_p"] = "waveTransmissive"
    gas["predicted"] = True
    gas["note"] = (
        "PREDICTED station knobs. Not signed GG-to-rotor. "
        "U from rpm*rm when rpm set. Not FIELD_CFD."
    )
    base["gas"] = gas
    base["geometry"] = g
    return validate_job(base, source="knobs")



def _plot_poly_preview(dest_png, case_png, poly, job, note=""):
    """Cascade outline: pitch stack vertical, axial flow left→right, fins/TE down.

    Matches ORBIT photo frame (not a sideways port of pitch along x).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    from .job import pitch_m as _pitch_m
    try:
        pitch = float(_pitch_m(job))
    except Exception:
        pitch = 0.0088
    g = job.get("geometry") or {}
    # Mesh stays at 3; outline shows 5 so the pitch column reads like the wheel photo.
    n_show = 5
    # Portrait-leaning: pitch column reads vertical like the wheel photo.
    fig, ax = plt.subplots(figsize=(5.2, 7.2), dpi=120)
    metal_face = "#c5c8ce"
    metal_edge = "#2b2d31"
    try:
        gp = passage_gap(poly, pitch)
        ss, ps1 = gp["ss"], gp["ps1"]
        ax.fill(
            list(ss[:, 0] * 1000) + list(ps1[::-1, 0] * 1000),
            list(ss[:, 1] * 1000) + list(ps1[::-1, 1] * 1000),
            color="#6b5e2e", alpha=0.40, lw=0, zorder=0,
            label=f"gap g_min={gp['g_min']*1e3:.2f} mm",
        )
    except Exception:
        gp = None
    xs0 = [p[0] for p in poly]
    ys0 = [p[1] for p in poly]
    x_lo, x_hi = min(xs0), max(xs0)
    # Fins down: LE/TE tips at lower y of each C (dual-arc crown at +y).
    for k in range(n_show):
        xs = [p[0] * 1000 for p in poly]
        ys = [(p[1] + k * pitch) * 1000 for p in poly]
        ax.fill(xs, ys, color=metal_face, ec=metal_edge, lw=1.1, zorder=2,
                label=("metal" if k == 0 else None))
    y_bot = (min(ys0) - 0.15 * pitch) * 1000
    y_top = (max(ys0) + (n_show - 1) * pitch + 0.15 * pitch) * 1000
    # Crop dump: metal + ~0.4c inlet/outlet only — dump length sideways-ports the eye.
    c_mm = max((x_hi - x_lo) * 1000.0, 1.0)
    ax.set_xlim(x_lo * 1000 - 0.45 * c_mm, x_hi * 1000 + 0.55 * c_mm)
    ax.set_ylim(y_bot, y_top)
    ax.set_aspect("equal")
    ax.set_xlabel("x axial [mm]  → flow")
    ax.set_ylabel("y pitch [mm]  ↑ stack")
    # Do not dump mesh exception text into the title (sideways-ports the eye).
    ax.set_title("Cascade outline — upright C, fins down, flow →", fontsize=10)
    # Flow path: β from axial (+x). θ from vertical upright to W1 = 90° − |β1|.
    import math as _math
    gmeta = job.get("geometry") or {}
    try:
        b1 = float(gmeta.get("beta1_flow_deg") if gmeta.get("beta1_flow_deg") not in (None, "") else 65.0)
    except Exception:
        b1 = 65.0
    try:
        b2 = float(gmeta.get("beta2_flow_deg") if gmeta.get("beta2_flow_deg") not in (None, "") else -65.0)
    except Exception:
        b2 = -65.0
    y_mid = 0.5 * (y_bot + y_top)
    L = 0.35 * c_mm
    # Inlet W1 at LE, angle β1 from +x (CCW)
    x_le = x_lo * 1000
    r1 = _math.radians(b1)
    ax.annotate(
        "",
        xy=(x_le + L * _math.cos(r1), y_mid + L * _math.sin(r1)),
        xytext=(x_le, y_mid),
        arrowprops=dict(arrowstyle="->", color="#7c5cff", lw=1.6),
        zorder=3,
    )
    ax.text(x_le - 0.08 * c_mm, y_mid + 0.06 * (y_top - y_bot), r"$W_1$", color="#7c5cff", fontsize=8)
    # θ arc marker: vertical upright to W1
    theta_v = 90.0 - abs(b1)
    ax.text(
        x_le - 0.02 * c_mm,
        y_mid + 0.22 * L,
        rf"$\theta\approx{theta_v:.0f}^\circ$ (from vertical)",
        color="#e8e6f2",
        fontsize=7,
    )
    # Exit W2 at TE
    x_te = x_hi * 1000
    r2 = _math.radians(b2)
    ax.annotate(
        "",
        xy=(x_te + L * _math.cos(r2), y_mid + L * _math.sin(r2)),
        xytext=(x_te, y_mid),
        arrowprops=dict(arrowstyle="->", color="#a78bfa", lw=1.6),
        zorder=3,
    )
    ax.text(x_te + 0.02 * c_mm, y_mid - 0.08 * (y_top - y_bot), rf"$W_2\ (\beta_2={b2:.0f}^\circ)$", color="#a78bfa", fontsize=7)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    ax.set_facecolor("#12121a")
    fig.patch.set_facecolor("#0b0b0f")
    for spine in ax.spines.values():
        spine.set_color("#9b97b0")
    ax.tick_params(colors="#e8e6f2")
    ax.xaxis.label.set_color("#e8e6f2")
    ax.yaxis.label.set_color("#e8e6f2")
    ax.title.set_color("#e8e6f2")
    fig.tight_layout()
    dest_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest_png, facecolor="#0b0b0f", edgecolor="none")
    case_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(case_png, facecolor="#0b0b0f", edgecolor="none")
    plt.close(fig)

def write_preview(job: dict[str, Any], dest: Path | None = None) -> dict[str, Any]:
    """Call the real case/mesh writer. Returns polygon stats + PNG path. No solve. No OF.

    Building a case is not running CFD. Authority is SCOPING.
    """
    job = validate_job(copy.deepcopy(job))
    root = ROOT
    out_root = Path(job.get("output_dir", APP_OUTPUT))
    if not out_root.is_absolute():
        out_root = (root / out_root).resolve()
    if is_live_marlin_job(job, out_root):
        raise RuntimeError("preview refuses to write the live Marlin 80 us tree")
    if str(job.get("name")) == "marlin_v2_rotor":
        job["name"] = APP_NAME
        job["geometry_test"] = True
        job["output_dir"] = APP_OUTPUT
        out_root = (root / APP_OUTPUT).resolve()
    case_dir = dest or (out_root / "openfoam_cases" / str(job["name"]))
    case_dir = Path(case_dir)
    if job_tree_held_by_other(case_dir):
        raise RuntimeError(
            f"refusing to rewrite {case_dir}: mesh/solve holds this tree"
        )
    job, geom_notes, geom_bounds = apply_metal_bounds(job)
    ml = compute_meanline(job)
    spec = spec_from_job(job)
    spec.beta1_metal_deg = float(ml.beta1_metal_deg)
    spec.beta2_metal_deg = float(ml.beta2_metal_deg)
    g, gas = job["geometry"], job["gas"]
    xin, xout = domain_x(job)
    times = compute_times(
        chord_m=float(g["chord_m"]),
        w1_m_s=float(gas["w1_m_s"]),
        gamma=float(gas["gamma"]),
        r_specific=float(gas["r_specific_j_kg_k"]),
        t1_k=float(gas["t1_k"]),
        x_in_m=xin,
        x_out_m=xout,
        n_chords_min=float(job["cfd"].get("n_chords_min", 10.0)),
        n_lx_a_min=float(job["cfd"].get("n_lx_a_min", 1.2)),
    )
    poly, more_notes = safe_profile_from_job(job, spec)
    pack_notes = fit_pitch_to_metal(job, poly)
    geom_notes = list(geom_notes) + list(more_notes) + list(pack_notes)
    mesh = None
    t_end = times.t_end_floor_s
    # Outline and mesh wire must not share one file — tabs need both.
    outline_case = case_dir / "outline_preview.png"
    outline_root = out_root / "outline_preview.png"
    mesh_case = case_dir / "mesh_preview.png"
    mesh_root = out_root / "mesh_preview.png"
    out_root.mkdir(parents=True, exist_ok=True)
    gap = passage_gap(poly, pitch_m(job))
    geom_notes.append(f"passage_gap g_min={float(gap['g_min'])*1e3:.3f} mm (arc-length n_hat, not y(x))")
    try:
        if float(gap["g_min"]) <= 0.0:
            raise RuntimeError(
                f"INTERSECTING METAL: passage_gap g_min={float(gap['g_min'])*1e3:.4f} mm <= 0. "
                "Refuse mesh/solve."
            )
        mesh, t_end = write_case(case_dir, job, ml, times, spec)
        # Day-one outline is the flow-path cascade (W1/W2/θ), not the mesh wire crop.
        # write_case already wrote body-fitted wire to mesh_preview.png — leave it.
        _plot_poly_preview(outline_root, outline_case, poly, job, note="")
        if mesh_case.is_file():
            shutil.copy2(mesh_case, mesh_root)
    except (ValueError, RuntimeError, IndexError) as exc:
        # Live outline must still show metal. Pitch overflow is a packing fail,
        # not a license to flatten the stem onto a chord cut.
        _plot_poly_preview(outline_root, outline_case, poly, job, note=str(exc))
        mesh = None
    ys = [p[1] for p in poly]
    xs = [p[0] for p in poly]
    job_path = out_root / f"{job['name']}.json"
    out_root.mkdir(parents=True, exist_ok=True)
    job_path.write_text(json.dumps(job, indent=2, default=str) + "\n", encoding="utf-8")
    plot_dir = out_root / "plots"
    tri = triangle_png(plot_dir / "triangles.png", ml)
    return {
        "triangles": triangles_from_meanline(ml),
        "ntrs_checks": ntrs_evaluate(ml, job),
        "loss_scoping": ml.loss_scoping,
        "euler_work_j_kg": ml.euler_work_j_kg,
        "power_w": ml.power_w,
        "u_m_s": ml.u_m_s,
        "Mw1": ml.Mw1,
        "Max1": ml.Max1,
        "Re_c": ml.Re_c,
        "psi": ml.psi,
        "U_over_C1": ml.U_over_C1,
        "unique_incidence_active": ml.unique_incidence_active,
        "goldman": None,
        "hu_mm": (g.get("upper_sagitta_m") or 0) * 1e3 if g.get("upper_sagitta_m") not in (None, "") else None,
        "hl_mm": (g.get("lower_sagitta_m") or 0) * 1e3 if g.get("lower_sagitta_m") not in (None, "") else None,
        "le_mm": (g.get("le_fillet_r_m") or 0) * 1e3 if g.get("le_fillet_r_m") not in (None, "") else None,
        "te_mm": (g.get("te_fillet_r_m") or 0) * 1e3 if g.get("te_fillet_r_m") not in (None, "") else None,
        "Mw1_OF": None,
        "of_station": None,
        "of_power_w": None,
        "ok": True,
        "predicted": True,
        "success": False,
        "authority": AUTHORITY_SCOPING,
        "cfd_ran": False,
        "hardware_correlated": False,
        "name": job["name"],
        "family": g.get("profile_family"),
        "pitch_m": pitch_m(job),
        "solidity": float(g["solidity"]),
        "n_blades_machine": int(g["n_blades_machine"]),
        "chord_m": float(g["chord_m"]),
        "span_m": float(g["span_m"]),
        "mean_radius_m": float(g["mean_radius_m"]),
        "stagger_deg": float(g.get("stagger_deg") or ml.stagger_deg),
        "beta1_flow_deg": float(g["beta1_flow_deg"]),
        "beta2_flow_deg": float(g["beta2_flow_deg"]),
        "upper_sagitta_c": g.get("upper_sagitta_c"),
        "lower_sagitta_c": g.get("lower_sagitta_c"),
        "thickness_c": g.get("thickness_c"),
        "n_cells": None if mesh is None else mesh.n_cells,
        "n_points": None if mesh is None else mesh.n_points,
        "min_area_2d": None if mesh is None else mesh.min_area_2d,
        "mesh_kind": "outline_only" if mesh is None else mesh.mesh_kind,
        "poly_area_m2": polygon_signed_area(poly),
        "poly_ymax_m": max(ys) if ys else None,
        "poly_ymin_m": min(ys) if ys else None,
        "poly_xmax_m": max(xs) if xs else None,
        "png": (
            str(outline_case if outline_case.is_file() else outline_root)
            if (outline_case.is_file() or outline_root.is_file())
            else None
        ),
        "outline_png": (
            str(outline_case if outline_case.is_file() else outline_root)
            if (outline_case.is_file() or outline_root.is_file())
            else None
        ),
        "mesh_png": (
            str(mesh_case if mesh_case.is_file() else mesh_root)
            if (mesh_case.is_file() or mesh_root.is_file())
            else None
        ),
        "triangles_png": str(tri) if tri else None,
        "case_dir": str(case_dir),
        "job_json": str(job_path),
        "t_end_s": t_end,
        "eta_from_cfd": None,
        "notes": [] if mesh is None else mesh.check_notes,
        "disclaimer": "Engineering aid, not a flight certificate. SCOPING outline is not FIELD_CFD.",
        "geom_warnings": geom_notes,
        "geom_bounds": geom_bounds,
        "geom_ok": not geom_notes,
    }


def outline_from_knobs(knobs: dict[str, Any] | None = None, dest: Path | None = None) -> dict[str, Any]:
    """Knob change → real geometry/mesh writer → PNG. Does not open OpenFOAM."""
    job = knobs_to_job(knobs)
    return write_preview(job, dest=dest)


def authority_from_report(report: dict[str, Any] | None) -> str:
    """FIELD_CFD only after rhoCentralFoam succeeded on this machine this case.

    Never HARDWARE_CORRELATED. predicted stays true (unsigned station).
    """
    if not report:
        return AUTHORITY_SCOPING
    flags = report.get("flags") or {}
    solve = report.get("solve") or {}
    name = str(report.get("name") or "")
    if name == "marlin_v2_rotor":
        # UI never borrows the live Marlin label.
        return AUTHORITY_SCOPING
    mesh_ok = flags.get("mesh_ok") is True
    cfd_ran = bool(flags.get("solve_ok")) and solve.get("rc") == 0 and not solve.get("fatal")
    if cfd_ran and mesh_ok:
        return AUTHORITY_FIELD_CFD
    return AUTHORITY_SCOPING
