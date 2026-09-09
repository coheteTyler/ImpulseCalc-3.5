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
                    {"rts": "rotorValues.chord", "id": "chord_mm", "label": "Chord", "sym": "c", "unit": "mm", "default": 18.0},
                ],
            },
            "horizontal": {
                "label": "Horizontal (pitch / diameter)",
                "intent": "horizontal",
                "keys": [
                    {"rts": "cycle.d_m", "id": "dm_mm", "label": "Mean diameter", "sym": "dm", "unit": "mm", "default": 88.9},
                    {"rts": "rotorValues.N", "id": "Z", "label": "Blade count (machine)", "sym": "Z", "unit": "—", "default": 36},
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
        "hint": "Rotor metal only. Stator omitted — inlet gas carries post-nozzle state.",
        "color": "#2a7a68",
        "css": "sys-metal",
        "subgroups": {
            "profile": {
                "label": "Profile / fillets",
                "keys": [
                    {"rts": "rotorValues.r_le_ratio", "id": "r_le_ratio", "label": "LE radius / chord", "sym": "rLE/c", "unit": "—", "default": 0.04},
                    {"rts": "rotorValues.r_te_ratio", "id": "r_te_ratio", "label": "TE radius / chord", "sym": "rTE/c", "unit": "—", "default": 0.04},
                    {"rts": "rotorValues.unguided_turning", "id": "unguided_turning_deg", "label": "Unguided turning", "sym": "θu", "unit": "deg", "default": 8.0},
                    {"rts": "rotorValues.t_ri", "id": "t_ri_mm", "label": "Inlet gap", "sym": "tri", "unit": "mm", "default": 3.0},
                    {"rts": "rotorValues.gamma_turning_ri", "id": "gamma_turning_ri_deg", "label": "Inlet turning wedge", "sym": "γri", "unit": "deg", "default": -10.0},
                    {"rts": "cycle.gamma_ri", "id": "gamma_ri_deg", "label": "Rotor inlet wedge", "sym": "γri,wedge", "unit": "deg", "default": 8.0},
                    {"rts": "cycle.beta_2", "id": "beta2_deg", "label": "Relative exit from axial", "sym": "β2", "unit": "deg", "default": -65.0},
                    {"rts": "cycle.reaction", "id": "reaction", "label": "Degree of reaction", "sym": "R", "unit": "—", "default": 0.0},
                ],
            },
            "ic3_mm": {
                "label": "ImpulseCalc3 absolute metal (mm)",
                "keys": [
                    {"rts": "(ic3)", "id": "hu_mm", "label": "Upper sagitta", "sym": "hu", "unit": "mm", "default": 6.5},
                    {"rts": "(ic3)", "id": "hl_mm", "label": "Lower sagitta", "sym": "hl", "unit": "mm", "default": 1.5},
                    {"rts": "(ic3)", "id": "le_mm", "label": "LE fillet", "sym": "rLE", "unit": "mm", "default": 0.4},
                    {"rts": "(ic3)", "id": "te_mm", "label": "TE fillet", "sym": "rTE", "unit": "mm", "default": 0.0},
                    {"rts": "(ic3)", "id": "lin_mm", "label": "Inlet straight", "sym": "Lin", "unit": "mm", "default": 0.0},
                    {"rts": "(ic3)", "id": "lout_mm", "label": "Outlet straight", "sym": "Lout", "unit": "mm", "default": 0.0},
                ],
            },
        },
    },
}

# Smart intents: selecting an intent surfaces related keys even if not one RTS var.
SMART_INTENTS: dict[str, list[str]] = {
    "inlet": ["p1_Pa", "T1_K", "mdot_kg_s", "gamma", "cp_J_kgK", "Mmol_kg_kmol", "beta1_deg", "W1_m_s", "eta_n"],
    "horizontal": ["dm_mm", "Z", "solidity", "epsilon", "chord_mm"],
    "vertical": ["chord_mm", "hu_mm", "hl_mm", "solidity"],
    "size": ["dm_mm", "chord_mm", "Z", "solidity", "epsilon", "omega_rps"],
    "metal": ["hu_mm", "hl_mm", "le_mm", "te_mm", "lin_mm", "lout_mm", "r_le_ratio", "r_te_ratio", "unguided_turning_deg", "beta2_deg", "reaction"],
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
    """Map central profile → ImpulseCalc3 knobs dict (mm + gas)."""
    v = profile.get("values") or profile
    knobs: dict[str, Any] = {}
    for src, dst in (
        ("hu_mm", "hu_mm"),
        ("hl_mm", "hl_mm"),
        ("le_mm", "le_mm"),
        ("te_mm", "te_mm"),
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
    if "chord_mm" in v and v["chord_mm"] not in (None, ""):
        knobs["chord_m"] = float(v["chord_mm"]) * 1e-3
    if "dm_mm" in v and v["dm_mm"] not in (None, ""):
        knobs["mean_radius_m"] = float(v["dm_mm"]) * 5e-4  # dm/2
    return knobs
