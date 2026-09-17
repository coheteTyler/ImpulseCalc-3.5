"""ImpulseCalc 3.5 — RTS-keyed smart filters (local scaffolding).

No stator geometry in the article. Inlet gas = uniform post-stator working fluid.
Filters group RTS-style knobs so Update writes one central profile JSON.
Authority: PREDICTED / SCOPING until FIELD_CFD flats Ft.
"""

from __future__ import annotations

from typing import Any

# Color tokens match viewer legend (gas / size-pack / metal).
FILTER_GROUPS: dict[str, dict[str, Any]] = {
    "inlet_gas": {
        "id": "inlet_gas",
        "label": "Inlet gas",
        "hint": "Uniform post-stator working fluid. No stator metal in this article.",
        "color": "#1d5a8a",
        "css": "sys-gas",
        "subgroups": {
            "state": {
                "label": "Thermodynamic state",
                "keys": [
                    {"rts": "cycle.p_01", "id": "p1_Pa", "label": "Inlet total pressure", "sym": "p01", "unit": "Pa", "default": 1.5e6},
                    {"rts": "cycle.T_01", "id": "T1_K", "label": "Inlet total temperature", "sym": "T01", "unit": "K", "default": 875.0},
                    {"rts": "cycle.mdot", "id": "mdot_kg_s", "label": "Mass flow", "sym": "mdot", "unit": "kg/s", "default": 0.4},
                ],
            },
            "eos": {
                "label": "Gas model",
                "keys": [
                    {"rts": "cycle.gamma", "id": "gamma", "label": "Ratio of specific heats", "sym": "γ", "unit": "—", "default": 1.37},
                    {"rts": "cycle.c_p", "id": "cp_J_kgK", "label": "Specific heat", "sym": "cp", "unit": "J/(kg·K)", "default": 8100.0},
                    {"rts": "cycle.Mmol", "id": "Mmol_kg_kmol", "label": "Molar mass", "sym": "M", "unit": "kg/kmol", "default": 3.83},
                ],
            },
            "relative_inlet": {
                "label": "Relative inlet (post-stator)",
                "keys": [
                    {"rts": "cycle.beta_1", "id": "beta1_deg", "label": "Relative inlet from axial", "sym": "β1", "unit": "deg", "default": 65.0, "help": "θ from vertical upright to W1 is 90°−|β1|"},
                    {"rts": "(derived)", "id": "W1_m_s", "label": "Relative inlet speed", "sym": "W1", "unit": "m/s", "default": 1200.0},
                    {"rts": "cycle.eta_n", "id": "eta_n", "label": "Nozzle efficiency (SCOPING)", "sym": "ηn", "unit": "—", "default": 0.90},
                ],
            },
        },
    },
    "size": {
        "id": "size",
        "label": "Size",
        "hint": "Horizontal sizing (pitch / diameter / count) vs vertical (chord). Smart filter pulls related knobs.",
        "color": "#5a4a8a",
        "css": "sys-pack",
        "subgroups": {
            "vertical": {
                "label": "Vertical (chord)",
                "intent": "chord",
                "keys": [
                    {"rts": "rotorValues.chord", "id": "chord_mm", "label": "Chord", "sym": "c", "unit": "mm", "default": 10.0},
                ],
            },
            "horizontal": {
                "label": "Horizontal (pitch / diameter)",
                "intent": "horizontal",
                "keys": [
                    {"rts": "cycle.d_m", "id": "dm_mm", "label": "Mean diameter", "sym": "dm", "unit": "mm", "default": 75.0},
                    {"rts": "rotorValues.N", "id": "Z", "label": "Blade count (machine)", "sym": "N", "unit": "—", "default": 25},
                    {"rts": "(derived σ=c/s)", "id": "solidity", "label": "Solidity (derived c/s)", "sym": "σ", "unit": "—", "default": 1.4, "derived": True, "readonly": True},
                    {"rts": "cycle.degree_of_admission", "id": "epsilon", "label": "Admission", "sym": "ε", "unit": "—", "default": 1.0},
                ],
            },
            "speed": {
                "label": "Wheel speed",
                "keys": [
                    {"rts": "cycle.omega", "id": "omega_rps", "label": "Rotational speed", "sym": "ω", "unit": "rev/s", "default": 666.67},
                    {"rts": "cycle.P", "id": "P_W", "label": "Target shaft power (SCOPING)", "sym": "P", "unit": "W", "default": 125e3},
                ],
            },
        },
    },
    "metal": {
        "id": "metal",
        "label": "Metal",
        "hint": "Pritchard 11-param rotor metal. LE/TE fillets via le_mm/te_mm only (ratios demoted). No Lin/Lout stems. Stator omitted.",
        "color": "#2a7a68",
        "css": "sys-metal",
        "subgroups": {
            "pritchard": {
                "label": "Pritchard 11-param",
                "keys": [
                    {"rts": "cycle.beta_1", "id": "beta1_deg", "label": "Inlet blade angle βi", "sym": "βi", "unit": "deg", "default": 65.0},
                    {"rts": "cycle.beta_2", "id": "beta2_deg", "label": "Exit blade angle βo (impulse: βo ≈ −βi)", "sym": "βo", "unit": "deg", "default": -65.0, "hint": "impulse: βo ≈ −βi"},
                    {"rts": "rotorValues.unguided_turning", "id": "unguided_turning_deg", "label": "Unguided turning", "sym": "θu", "unit": "deg", "default": 8.0},
                    {"rts": "cycle.gamma_ri", "id": "epsilon_i_deg", "label": "Inlet half-wedge εi", "sym": "εi", "unit": "deg", "default": 10.0},
                    {"rts": "rotorValues.t_ri", "id": "throat_mm", "label": "Geometric throat", "sym": "o", "unit": "mm", "default": 2.356, "hint": "Default ≈ 0.25·pitch at R=37.5 mm, Z=25"},
                    {"rts": "(ic3)", "id": "throat_pitch_ratio", "label": "Throat / pitch (if throat_mm empty)", "sym": "o/s", "unit": "—", "default": 0.25},
                    {"rts": "(ic3)", "id": "cx_mm", "label": "Axial chord cx", "sym": "cx", "unit": "mm", "default": 10.0},
                    {"rts": "(ic3)", "id": "ct_mm", "label": "Tangential chord ct (stagger)", "sym": "ct", "unit": "mm", "default": 0.0},
                    {"rts": "cycle.reaction", "id": "reaction", "label": "Degree of reaction", "sym": "R", "unit": "—", "default": 0.0},
                ],
            },
            "fillets": {
                "label": "LE / TE fillets (absolute — only drivers)",
                "keys": [
                    {"rts": "(ic3)", "id": "le_mm", "label": "LE radius le_r", "sym": "le_r", "unit": "mm", "default": 0.4, "hint": "Only LE fillet driver. Default 0.04·chord."},
                    {"rts": "(ic3)", "id": "te_mm", "label": "TE radius te_r", "sym": "te_r", "unit": "mm", "default": 0.4, "hint": "Only TE fillet driver. Default 0.04·chord. te_mm=0 no longer wins over a ratio — omit te_mm to use demoted r_te_ratio."},
                    {"rts": "rotorValues.r_le_ratio", "id": "r_le_ratio", "label": "LE radius / chord (demoted)", "sym": "rLE/c", "unit": "—", "default": 0.04, "hint": "Demoted: used only if le_mm absent."},
                    {"rts": "rotorValues.r_te_ratio", "id": "r_te_ratio", "label": "TE radius / chord (demoted)", "sym": "rTE/c", "unit": "—", "default": 0.04, "hint": "Demoted: used only if te_mm absent. Cannot zero TE when te_mm is set."},
                ],
            },
            "legacy_bucket": {
                "label": "Legacy dual-arc bucket (not default)",
                "keys": [
                    {"rts": "(ic3)", "id": "hu_mm", "label": "Upper sagitta (legacy)", "sym": "hu", "unit": "mm", "default": None},
                    {"rts": "(ic3)", "id": "hl_mm", "label": "Lower sagitta (legacy)", "sym": "hl", "unit": "mm", "default": None},
                    {"rts": "(ic3)", "id": "lin_mm", "label": "Inlet straight (ignored for Pritchard)", "sym": "Lin", "unit": "mm", "default": 0.0},
                    {"rts": "(ic3)", "id": "lout_mm", "label": "Outlet straight (ignored for Pritchard)", "sym": "Lout", "unit": "mm", "default": 0.0},
                    {"rts": "(ic3)", "id": "passage_depth_mm", "label": "Passage depth (fillet-foot clearance)", "sym": "g_pass", "unit": "mm", "default": None},
                    {"rts": "(ic3)", "id": "constant_passage_width", "label": "Constant passage width", "sym": "g≈const", "unit": "flag", "default": False},
                ],
            },
        },
    },
}

# Smart
# Smart intents: selecting an intent surfaces related keys even if not one RTS var.
SMART_INTENTS: dict[str, list[str]] = {
    "inlet": ["p1_Pa", "T1_K", "mdot_kg_s", "gamma", "cp_J_kgK", "Mmol_kg_kmol", "beta1_deg", "W1_m_s", "eta_n"],
    "horizontal": ["dm_mm", "Z", "solidity", "epsilon", "chord_mm"],
    "vertical": ["chord_mm", "hu_mm", "hl_mm", "solidity"],
    "size": ["dm_mm", "chord_mm", "Z", "solidity", "epsilon", "omega_rps"],
    "metal": ["beta1_deg", "beta2_deg", "unguided_turning_deg", "epsilon_i_deg", "throat_mm", "throat_pitch_ratio", "cx_mm", "ct_mm", "le_mm", "te_mm", "chord_mm", "Z", "r_le_ratio", "r_te_ratio", "reaction"],
    "flow_rate": ["mdot_kg_s", "p1_Pa", "T1_K", "W1_m_s", "dm_mm", "epsilon"],
}


def all_keys() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for g in FILTER_GROUPS.values():
        for sg_id, sg in g["subgroups"].items():
            for k in sg["keys"]:
                row = dict(k)
                row["group"] = g["id"]
                row["css"] = g["css"]
                row["color"] = g["color"]
                row["subgroup"] = sg.get("label") or sg_id
                out.append(row)
    return out


def schema_payload() -> dict[str, Any]:
    """JSON for /api/filters/schema."""
    prof = defaults_profile()
    return {
        "format": "impulsecalc35_filter_schema_v1",
        "groups": {
            gid: {"id": g["id"], "label": g["label"], "hint": g["hint"], "color": g["color"], "css": g["css"]}
            for gid, g in FILTER_GROUPS.items()
        },
        "intents": SMART_INTENTS,
        "keys": all_keys(),
        "defaults": prof["values"],
    }



def sanitize_impulse_betas(profile: dict[str, Any]) -> dict[str, Any]:
    """Keep impulse article from re-poisoning stagger via stale β2=+16 saved profiles."""
    if not isinstance(profile, dict):
        return profile
    v = profile.setdefault("values", {})
    try:
        b1 = float(v.get("beta1_deg") if v.get("beta1_deg") not in (None, "") else 65.0)
    except (TypeError, ValueError):
        b1 = 65.0
    try:
        b2 = float(v.get("beta2_deg") if v.get("beta2_deg") not in (None, "") else -abs(b1))
    except (TypeError, ValueError):
        b2 = -abs(b1)
    # Poisoned chrome default was +16 with β1=+65 → bogus ~40° stagger.
    if abs(b2 - 16.0) < 1e-6 and abs(b1 - 65.0) < 1e-6:
        b2 = -abs(b1)
    # Impulse law until ORBIT signs otherwise: β2 ≈ −β1 when |β2| looks unsigned/wrong-sign.
    if b2 > 0 and b1 > 0 and abs(abs(b2) - abs(b1)) > 20.0:
        b2 = -abs(b1)
    v["beta1_deg"] = b1
    v["beta2_deg"] = b2
    return profile

def defaults_profile() -> dict[str, Any]:
    """Central profile written by Update. Flat id→value plus metadata."""
    vals = {k["id"]: k["default"] for k in all_keys()}
    return {
        "format": "impulsecalc35_profile_v1",
        "authority": "SCOPING",
        "predicted": True,
        "article": "ImpulseCalc 3.5 filter profile — PREDICTED. No stator metal; inlet gas is uniform post-stator.",
        "values": vals,
        "rts_map": {k["id"]: k["rts"] for k in all_keys()},
    }


def filter_keys(intent: str | None = None, group: str | None = None) -> list[dict[str, Any]]:
    keys = all_keys()
    if group:
        keys = [k for k in keys if k["group"] == group]
    if intent:
        want = set(SMART_INTENTS.get(intent, []))
        if want:
            keys = [k for k in keys if k["id"] in want]
    return keys


def profile_to_ic3_knobs(profile: dict[str, Any]) -> dict[str, Any]:
    """Map central profile → ImpulseCalc3 knobs dict (mm + gas).

    TE fillet path: values.te_mm → knobs.te_mm → geometry.te_fillet_r_m
    (preview.knobs_to_job). values.r_te_ratio → te_fillet_r_c when te_mm absent.
    Absolute te_mm wins when set and >0; te_mm=0 is treated as unset for Pritchard
    (map r_te_ratio·chord) so a zero cannot kill TE. Default family: pritchard_11.
    """
    v = profile.get("values") or profile
    knobs: dict[str, Any] = {}
    for src, dst in (
        ("hu_mm", "hu_mm"),
        ("hl_mm", "hl_mm"),
        ("le_mm", "le_mm"),
        ("te_mm", "te_mm"),
        ("passage_depth_mm", "passage_depth_mm"),
        ("s_mm", "s_mm"),
        ("lin_mm", "lin_mm"),
        ("lout_mm", "lout_mm"),
        ("chord_mm", "chord_mm"),
        ("Z", "Z"),
        ("solidity", "solidity"),
        ("beta1_deg", "beta1"),
        ("beta2_deg", "beta2"),
        ("dm_mm", "dm_mm"),
        ("W1_m_s", "W1"),
        ("p1_Pa", "p1"),
        ("T1_K", "T1"),
        ("gamma", "gamma"),
        ("cp_J_kgK", "cp"),
    ):
        if src in v and v[src] not in (None, ""):
            knobs[dst] = v[src]
    # Absolute fillet aliases (m or mm) → mm. ≤0 treated as unset.
    if "te_mm" not in knobs:
        if v.get("te_fillet_r_m") not in (None, "") and float(v["te_fillet_r_m"]) > 0:
            knobs["te_mm"] = float(v["te_fillet_r_m"]) * 1e3
        elif v.get("te_fillet_mm") not in (None, "") and float(v["te_fillet_mm"]) > 0:
            knobs["te_mm"] = float(v["te_fillet_mm"])
    if "le_mm" not in knobs:
        if v.get("le_fillet_r_m") not in (None, "") and float(v["le_fillet_r_m"]) > 0:
            knobs["le_mm"] = float(v["le_fillet_r_m"]) * 1e3
        elif v.get("le_fillet_mm") not in (None, "") and float(v["le_fillet_mm"]) > 0:
            knobs["le_mm"] = float(v["le_fillet_mm"])
    if knobs.get("te_mm") is not None and float(knobs["te_mm"]) <= 0:
        knobs.pop("te_mm", None)
    if knobs.get("le_mm") is not None and float(knobs["le_mm"]) <= 0:
        knobs.pop("le_mm", None)

    # Pritchard cx/ct independents (stagger = atan(ct/cx); chord = hypot).
    for src, dst in (
        ("unguided_turning_deg", "unguided_turning_deg"),
        ("epsilon_i_deg", "epsilon_i_deg"),
        ("throat_mm", "throat_mm"),
        ("throat_pitch_ratio", "throat_pitch_ratio"),
        ("cx_mm", "cx_mm"),
        ("ct_mm", "ct_mm"),
    ):
        if src in v and v[src] not in (None, ""):
            knobs[dst] = v[src]
    cx_mm = float(knobs["cx_mm"]) if knobs.get("cx_mm") not in (None, "") else None
    ct_mm = float(knobs["ct_mm"]) if knobs.get("ct_mm") not in (None, "") else 0.0
    if cx_mm is not None:
        chord_mm = (cx_mm ** 2 + ct_mm ** 2) ** 0.5
        knobs["cx_mm"] = cx_mm
        knobs["ct_mm"] = ct_mm
        knobs["chord_mm"] = chord_mm
        knobs["chord_m"] = chord_mm * 1e-3
        knobs["cx_m"] = cx_mm * 1e-3
        knobs["ct_m"] = ct_mm * 1e-3
    elif "chord_mm" in v and v["chord_mm"] not in (None, ""):
        knobs["chord_m"] = float(v["chord_mm"]) * 1e-3
        knobs["chord_mm"] = float(v["chord_mm"])
        chord_mm = float(v["chord_mm"])
    else:
        chord_mm = float(knobs.get("chord_mm") or 10.0)
        knobs["chord_mm"] = chord_mm
        knobs["chord_m"] = chord_mm * 1e-3

    # RTS ratios → absolute le_r/te_r using /chord (Pritchard length reference).
    if "le_mm" not in knobs:
        ratio = v.get("r_le_ratio", v.get("le_fillet_r_c", v.get("le_radius_c")))
        if ratio not in (None, ""):
            knobs["le_mm"] = float(ratio) * chord_mm
        else:
            knobs["le_mm"] = 0.04 * chord_mm
    if "te_mm" not in knobs:
        ratio = v.get("r_te_ratio", v.get("te_fillet_r_c", v.get("te_radius_c")))
        if ratio not in (None, ""):
            knobs["te_mm"] = float(ratio) * chord_mm
        else:
            knobs["te_mm"] = 0.04 * chord_mm
    # Surface demoted ratios for readout only (never drive when abs present).
    knobs["r_le_ratio"] = float(knobs["le_mm"]) / max(chord_mm, 1e-9)
    knobs["r_te_ratio"] = float(knobs["te_mm"]) / max(chord_mm, 1e-9)

    if "dm_mm" in v and v["dm_mm"] not in (None, ""):
        knobs["mean_radius_m"] = float(v["dm_mm"]) * 5e-4  # dm/2
        knobs["dm_mm"] = float(v["dm_mm"])

    # FIELD-path family. Stems off (Goldman L_in/L_out are not Pritchard radii).
    knobs["family"] = "pritchard_11"
    knobs["profile_family"] = "pritchard_11"
    knobs["lin_mm"] = 0.0
    knobs["lout_mm"] = 0.0
    return knobs


def _mm_from_m(v: Any) -> float | None:
    if v in (None, ""):
        return None
    return float(v) * 1e3


def design_to_profile_values(design: dict[str, Any]) -> dict[str, Any]:
    """Flatten a full job JSON, knobs dict, or profile.values into filter ids."""
    if not isinstance(design, dict):
        raise ValueError("design must be a JSON object")
    # Already a profile envelope
    if isinstance(design.get("values"), dict) and (
        design.get("format") in (None, "impulsecalc35_profile_v1") or "rts_map" in design
    ):
        return dict(design["values"])
    # Nested geometry job (default_design.json shape)
    g = design.get("geometry") if isinstance(design.get("geometry"), dict) else {}
    gas = design.get("gas") if isinstance(design.get("gas"), dict) else {}
    src = {**design, **g, **gas}
    # If caller passed knobs/profile flat values already
    if not g and any(k in design for k in ("hu_mm", "te_mm", "chord_mm", "beta1_deg")):
        src = design
    out: dict[str, Any] = {}
    # size
    if src.get("chord_m") not in (None, "") or src.get("chord_mm") not in (None, ""):
        out["chord_mm"] = float(src["chord_mm"]) if src.get("chord_mm") not in (None, "") else float(src["chord_m"]) * 1e3
    if src.get("mean_radius_m") not in (None, "") or src.get("dm_mm") not in (None, ""):
        if src.get("dm_mm") not in (None, ""):
            out["dm_mm"] = float(src["dm_mm"])
        else:
            out["dm_mm"] = float(src["mean_radius_m"]) * 2e3
    z = src.get("n_blades_machine", src.get("Z"))
    if z not in (None, ""):
        out["Z"] = int(z)
    if src.get("solidity") not in (None, ""):
        out["solidity"] = float(src["solidity"])
    # metal mm
    for dst, keys in (
        ("hu_mm", ("hu_mm", "upper_sagitta_m", "upper_sagitta_mm")),
        ("hl_mm", ("hl_mm", "lower_sagitta_m", "lower_sagitta_mm")),
        ("le_mm", ("le_mm", "le_fillet_r_m", "le_fillet_mm")),
        ("te_mm", ("te_mm", "te_fillet_r_m", "te_fillet_mm")),
        ("lin_mm", ("lin_mm", "lin_m")),
        ("lout_mm", ("lout_mm", "lout_m")),
    ):
        for k in keys:
            if src.get(k) not in (None, ""):
                val = float(src[k])
                out[dst] = val * 1e3 if k.endswith("_m") and not k.endswith("_mm") else val
                break
    # ratios
    for dst, keys in (
        ("r_le_ratio", ("r_le_ratio", "le_fillet_r_c", "le_radius_c")),
        ("r_te_ratio", ("r_te_ratio", "te_fillet_r_c", "te_radius_c")),
    ):
        for k in keys:
            if src.get(k) not in (None, ""):
                out[dst] = float(src[k])
                break
    # angles / gas
    b1 = src.get("beta1_deg", src.get("beta1_flow_deg", src.get("beta1")))
    b2 = src.get("beta2_deg", src.get("beta2_flow_deg", src.get("beta2")))
    if b1 not in (None, ""):
        out["beta1_deg"] = float(b1)
    if b2 not in (None, ""):
        out["beta2_deg"] = float(b2)
    for dst, keys in (
        ("W1_m_s", ("W1_m_s", "w1_m_s", "W1", "w1")),
        ("p1_Pa", ("p1_Pa", "p1_pa", "p1")),
        ("T1_K", ("T1_K", "t1_k", "T1", "t1")),
        ("gamma", ("gamma",)),
    ):
        for k in keys:
            if src.get(k) not in (None, ""):
                out[dst] = float(src[k])
                break
    for dst, keys in (
        ("unguided_turning_deg", ("unguided_turning_deg", "unguided_turning")),
        ("epsilon_i_deg", ("epsilon_i_deg", "inlet_half_wedge_deg", "epsilon_i")),
        ("throat_mm", ("throat_mm", "throat_m")),
        ("throat_pitch_ratio", ("throat_pitch_ratio", "throat_ratio")),
        ("cx_mm", ("cx_mm", "cx_m", "axial_chord_m")),
        ("ct_mm", ("ct_mm", "ct_m", "tangential_chord_m")),
    ):
        for k in keys:
            if src.get(k) not in (None, ""):
                val = float(src[k])
                if k.endswith("_m") and not k.endswith("_mm"):
                    out[dst] = val * 1e3
                else:
                    out[dst] = val
                break
    if src.get("pitch_m") not in (None, ""):
        out["s_mm"] = float(src["pitch_m"]) * 1e3
    elif src.get("s_mm") not in (None, ""):
        out["s_mm"] = float(src["s_mm"])
    return out


def merge_design_into_profile(profile: dict[str, Any], design: dict[str, Any]) -> dict[str, Any]:
    """Write imported design numbers into profile.values (locked source of truth)."""
    base = dict(profile) if isinstance(profile, dict) else defaults_profile()
    vals = dict(base.get("values") or {})
    vals.update(design_to_profile_values(design))
    base["values"] = vals
    return sanitize_impulse_betas(base)


def apply_constant_passage_to_profile(profile: dict) -> tuple[dict, dict]:
    """If constant_passage_width: Goldman concentric walls, ΔR = g_pass.

    Auto fills passage_depth_mm from fillet-foot→blade-below min distance.
    hl is slaved (Cl = Cu − pitch ŷ). Pitch too when a g_pass target is set.
    """
    from .geometry import apply_constant_passage_width, measure_fillet_foot_clearance
    from .preview import knobs_to_job

    if not isinstance(profile, dict):
        return profile, {"ok": False, "reason": "no profile"}
    v = profile.setdefault("values", {})
    flag = v.get("constant_passage_width")
    on = flag in (True, 1, "1", "true", "True", "yes", "on")
    v["constant_passage_width"] = bool(on)
    if not on:
        return profile, {"ok": True, "enabled": False}

    knobs = profile_to_ic3_knobs(profile)
    knobs["family"] = "pritchard_11"
    knobs["constant_passage_width"] = False
    if v.get("passage_depth_mm") not in (None, ""):
        knobs["passage_depth_mm"] = float(v["passage_depth_mm"])
    job = knobs_to_job(knobs)
    if v.get("passage_depth_mm") not in (None, ""):
        job.setdefault("geometry", {})["passage_depth_m"] = float(v["passage_depth_mm"]) * 1e-3
    job, report = apply_constant_passage_width(job)
    if report.get("ok"):
        if report.get("passage_depth_mm") is not None:
            v["passage_depth_mm"] = float(report["passage_depth_mm"])
        if report.get("pitch_mm") is not None:
            # surface pitch via solidity if possible — store s_mm helper
            v["s_mm"] = float(report["pitch_mm"])
        if report.get("hl_mm") is not None:
            v["hl_mm"] = float(report["hl_mm"])
    report["enabled"] = True
    return profile, report


def auto_passage_depth_for_profile(profile: dict) -> tuple[dict, dict]:
    """Measure fillet-foot clearance → write values.passage_depth_mm. No pitch move."""
    from .geometry import measure_fillet_foot_clearance
    from .preview import knobs_to_job

    if not isinstance(profile, dict):
        return profile, {"ok": False, "reason": "no profile"}
    v = profile.setdefault("values", {})
    knobs = profile_to_ic3_knobs(profile)
    knobs["family"] = "pritchard_11"
    knobs["constant_passage_width"] = False
    job = knobs_to_job(knobs)
    meas = measure_fillet_foot_clearance(job)
    if meas.get("ok"):
        v["passage_depth_mm"] = float(meas["passage_depth_mm"])
    return profile, meas
