"""Off-nominal vs cited NASA/NTRS impulse-turbine practice. SCOPING / PREDICTED.

Hard-coded citations only. Not a FIRMA spec. Not catalog η. Literature never
upgrades to FIELD_CFD even when a solve exists.
"""

from __future__ import annotations

import math
from typing import Any

from .job import pitch_m

AUTHORITY_SCOPING = "SCOPING"

STATUSES = frozenset({"NOMINAL", "HIGH", "LOW", "N/A", "INFO"})

# CITED BOUNDS — do not scrape NTRS at runtime; do not invent report numbers.
LITERATURE: dict[str, dict[str, Any]] = {
    "turning_140": {
        "ntrs_id": "19940018586",
        "report": "NTRS 19940018586 (N94-23059) — traditional rotor turning 140°",
        "bound": 140.0,
    },
    "huber_160_context": {
        "ntrs_id": "19920066198",
        "report": "Huber NTRS 19920066198 (AIAA 92-3221) — ~160° unconventional high-turning rocket turbopump blading (context, not a fail bar)",
        "bound": 160.0,
    },
    "zweifel_08": {
        "ntrs_id": "19740025363",
        "report": "NASA-SP-290-VOL-2 Stewart & Glassman Blade Design (NTRS 19740025363 / 19940006731) — min profile loss at Zweifel = 0.8",
        "bound": 0.8,
        "also": "19940006731",
    },
    "goldman_mw": {
        "ntrs_id": "19680010807",
        "report": "Goldman NASA-TN-D-4422, NTRS 19680010807 — impulse blade sections Mach 1.5 to 5.0",
        "bound_sonic": 1.0,
        "bound_goldman": 1.5,
    },
    "ko_shock": {
        "ntrs_id": "19890012364",
        "report": "Kacker–Okapuu 1982 shock term, cited in NTRS 19890012364 (SSME HPFTP loss)",
    },
}

NTRS_URL = "https://ntrs.nasa.gov/citations/{id}"

# Incompressible rotor Zweifel loading (Dixon / NASA-SP-290 Blade Design family):
#   Z_w  (symbol Zw) = 2 (s / c_x) cos²(β2) (tan β1 − tan β2)
# β from axial (this cascade's signed flow β), s = pitch, c_x = c · cos(stagger).
# Stagger from meanline. Not a catalog η knob.


def ntrs_url(ntrs_id: str) -> str:
    return NTRS_URL.format(id=ntrs_id)


def _finding(
    *,
    id: str,
    title: str,
    value: Any,
    bound: Any,
    unit: str,
    pct_vs_bound: float | None,
    status: str,
    ntrs_id: str,
    report: str,
    note: str | None = None,
) -> dict[str, Any]:
    if not ntrs_id:
        raise ValueError(f"missing ntrs_id for finding {id!r}")
    if status not in STATUSES:
        raise ValueError(f"bad status {status!r} for {id}")
    out: dict[str, Any] = {
        "id": id,
        "title": title,
        "value": value,
        "bound": bound,
        "unit": unit,
        "pct_vs_bound": pct_vs_bound,
        "status": status,
        "ntrs_id": ntrs_id,
        "report": report,
        "url": ntrs_url(ntrs_id),
        "predicted": True,
        "authority": AUTHORITY_SCOPING,
    }
    if note:
        out["note"] = note
    return out


def turning_deg(beta1_flow_deg: float, beta2_flow_deg: float) -> float:
    return abs(float(beta1_flow_deg) - float(beta2_flow_deg))


def zweifel_incompressible(
    *,
    pitch_s_m: float,
    chord_m: float,
    stagger_deg: float,
    beta1_flow_deg: float,
    beta2_flow_deg: float,
) -> tuple[float, float]:
    """Return (Zw, cx). Formula in module comment (Zw)."""
    b1 = math.radians(float(beta1_flow_deg))
    b2 = math.radians(float(beta2_flow_deg))
    stagger = math.radians(float(stagger_deg))
    cx = float(chord_m) * math.cos(stagger)
    s = float(pitch_s_m)
    if abs(cx) < 1e-16:
        return float("nan"), cx
    # Zw = 2 (s / cx) cos²(β2) (tan β1 − tan β2)
    zw = 2.0 * (s / cx) * (math.cos(b2) ** 2) * (math.tan(b1) - math.tan(b2))
    return zw, cx


def evaluate(
    meanline: Any,
    job: dict[str, Any] | None = None,
    cfd_flags: dict[str, Any] | None = None,
    of_station: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Literature table for THIS cascade vs cited NTRS. SCOPING / PREDICTED.

    of_station extras (OF Mw1 vs Goldman/sonic, OF |W2|/|W1|) do not upgrade
    literature rows to FIELD_CFD. No invented NTRS ids.
    """
    findings: list[dict[str, Any]] = []
    b1 = float(meanline.beta1_flow_deg)
    b2 = float(meanline.beta2_flow_deg)
    turn = turning_deg(b1, b2)

    lit140 = LITERATURE["turning_140"]
    bound140 = float(lit140["bound"])
    pct140 = (turn / bound140 - 1.0) * 100.0
    findings.append(
        _finding(
            id="rotor_turning",
            title="Rotor turning |β1_flow − β2_flow|",
            value=turn,
            bound=bound140,
            unit="deg",
            pct_vs_bound=pct140,
            status="HIGH" if turn > bound140 else "NOMINAL",
            ntrs_id=str(lit140["ntrs_id"]),
            report=str(lit140["report"]),
            note="Off-nominal HIGH if turning > 140°. % above = (turning/140 − 1)×100.",
        )
    )

    lit160 = LITERATURE["huber_160_context"]
    findings.append(
        _finding(
            id="huber_high_turning_context",
            title="Huber ~160° high-turning context (not a fail bar)",
            value=turn,
            bound=float(lit160["bound"]),
            unit="deg",
            pct_vs_bound=(turn / float(lit160["bound"]) - 1.0) * 100.0,
            status="INFO",
            ntrs_id=str(lit160["ntrs_id"]),
            report=str(lit160["report"]),
            note="Context only. Unconventional high-turning rocket turbopump blading ~160°. Not a second fail bar.",
        )
    )

    litz = LITERATURE["zweifel_08"]
    if job is not None:
        g = job["geometry"]
        s = pitch_m(job)
        c = float(g["chord_m"])
        stagger = float(getattr(meanline, "stagger_deg", None) or g.get("stagger_deg") or 0.0)
        zw, cx = zweifel_incompressible(
            pitch_s_m=s,
            chord_m=c,
            stagger_deg=stagger,
            beta1_flow_deg=b1,
            beta2_flow_deg=b2,
        )
        z0 = float(litz["bound"])
        rel = abs(zw - z0) / z0 if z0 else float("nan")
        if rel >= 0.25:
            zstat = "HIGH" if zw > z0 else "LOW"
        else:
            zstat = "NOMINAL"
        findings.append(
            _finding(
                id="zweifel_rotor",
                title="Incompressible rotor Zweifel Zw",
                value=zw,
                bound=z0,
                unit="—",
                pct_vs_bound=(zw / z0 - 1.0) * 100.0 if z0 else None,
                status=zstat,
                ntrs_id=str(litz["ntrs_id"]),
                report=str(litz["report"]),
                note=(
                    f"Zw = 2 (s/cx) cos²(β2) (tan β1 − tan β2); s={s:.6g} m, cx={cx:.6g} m, "
                    "stagger from meanline. Off-nominal if |Zw−0.8|/0.8 ≥ 25%."
                ),
            )
        )

    litg = LITERATURE["goldman_mw"]
    mw1 = float(meanline.Mw1)
    sonic = float(litg["bound_sonic"])
    gbound = float(litg["bound_goldman"])
    if mw1 >= gbound:
        mstat = "HIGH"
        mtitle = "Relative inlet Mach Mw1 (Goldman chart 1.5–5.0)"
        mnote = (
            "HIGH vs TN D-4422 published sweep (Mw1 ≥ 1.5). Meanline Mw1 = W1/a. "
            "Method can still run. Not a 25% shock-formation criterion."
        )
    elif mw1 >= sonic:
        mstat = "INFO"
        mtitle = "Relative inlet Mach Mw1 (method valid, off the 1968 chart)"
        mnote = (
            "INFO: 1.0 ≤ Mw1 < 1.5. Goldman construction is valid; off TN D-4422's "
            "published 1.5–5.0 sweep. Do not read thickness/solidity off the 1968 plots. "
            "Not NOMINAL. Not a fail. Meanline Mw1 = W1/a."
        )
    else:
        mstat = "NOMINAL"
        mtitle = "Relative inlet Mach Mw1 (subsonic)"
        mnote = "NOMINAL subsonic (Mw1 < 1). Meanline Mw1 = W1/a."
    findings.append(
        _finding(
            id="mw1_inlet",
            title=mtitle,
            value=mw1,
            bound=sonic,
            unit="—",
            pct_vs_bound=(mw1 / sonic - 1.0) * 100.0,
            status=mstat,
            ntrs_id=str(litg["ntrs_id"]),
            report=str(litg["report"]),
            note=mnote,
        )
    )
    findings.append(
        _finding(
            id="mw1_goldman_range",
            title="Goldman impulse-section Mach range 1.5–5.0",
            value=mw1,
            bound=gbound,
            unit="—",
            pct_vs_bound=(mw1 / gbound - 1.0) * 100.0,
            status="HIGH" if mw1 >= gbound else ("INFO" if mw1 >= sonic else "NOMINAL"),
            ntrs_id=str(litg["ntrs_id"]),
            report=str(litg["report"]),
            note=(
                "TN D-4422 published impulse-section sweep is Mach 1.5–5.0. "
                "1.0 ≤ Mw1 < 1.5: method valid, off the 1968 chart (INFO, not NOMINAL). "
                "Do not read thickness/solidity off those plots. Meanline only if CFD has not run."
            ),
        )
    )

    litko = LITERATURE["ko_shock"]
    findings.append(
        _finding(
            id="ko_shock_term_context",
            title="Kacker–Okapuu shock term (cited; no in-hand 25% criterion)",
            value=mw1,
            bound=None,
            unit="—",
            pct_vs_bound=None,
            status="INFO",
            ntrs_id=str(litko["ntrs_id"]),
            report=str(litko["report"]),
            note="Shock term exists in the cited SSME HPFTP loss report. No 25% shock-formation design number in-hand — not invented.",
        )
    )

    # |W2|/|W1| locked by meanline — skip a fake NASA find; one N/A row, no invented NTRS id.
    findings.append(
        {
            "id": "impulse_w2_over_w1",
            "title": "|W2|/|W1| impulse ratio",
            "value": abs(float(meanline.w2_m_s) / float(meanline.w1_m_s)) if float(meanline.w1_m_s) else None,
            "bound": 1.0,
            "unit": "—",
            "pct_vs_bound": None,
            "status": "N/A",
            "ntrs_id": None,
            "report": "ImpulseCalc3 meanline — |W2|=|W1| by construction",
            "url": None,
            "predicted": True,
            "authority": AUTHORITY_SCOPING,
            "note": "N/A (locked by meanline). Not a NASA find.",
        }
    )

    of = of_station if isinstance(of_station, dict) else {}
    mw_of = of.get("Mw1_OF")
    if mw_of is not None:
        try:
            mw_of_f = float(mw_of)
        except (TypeError, ValueError):
            mw_of_f = None
        else:
            if mw_of_f >= gbound:
                ostat, otitle, onote = (
                    "HIGH",
                    "OF relative inlet Mach Mw1 (Goldman-supersonic-impulse range)",
                    "HIGH Goldman-supersonic-impulse range (OF Mw1 ≥ 1.5). Value from this-case OpenFOAM station. Literature bound unchanged. Not a 25% shock-formation criterion.",
                )
            elif mw_of_f >= sonic:
                ostat, otitle, onote = (
                    "HIGH",
                    "OF relative inlet Mach Mw1 (transonic / shock-risk)",
                    "HIGH transonic/shock-risk (OF Mw1 ≥ 1). Value from this-case OpenFOAM station. Literature bound unchanged. Not a 25% shock-formation criterion.",
                )
            else:
                ostat, otitle, onote = (
                    "NOMINAL",
                    "OF relative inlet Mach Mw1 (subsonic)",
                    "NOMINAL subsonic (OF Mw1 < 1). Value from this-case OpenFOAM station.",
                )
            findings.append(
                _finding(
                    id="mw1_of_inlet",
                    title=otitle,
                    value=mw_of_f,
                    bound=sonic,
                    unit="—",
                    pct_vs_bound=(mw_of_f / sonic - 1.0) * 100.0,
                    status=ostat,
                    ntrs_id=str(litg["ntrs_id"]),
                    report=str(litg["report"]),
                    note=onote,
                )
            )
            findings.append(
                _finding(
                    id="mw1_of_goldman_range",
                    title="OF Goldman impulse-section Mach range 1.5–5.0",
                    value=mw_of_f,
                    bound=gbound,
                    unit="—",
                    pct_vs_bound=(mw_of_f / gbound - 1.0) * 100.0,
                    status="HIGH" if mw_of_f >= gbound else "NOMINAL",
                    ntrs_id=str(litg["ntrs_id"]),
                    report=str(litg["report"]),
                    note="OF Mw1 vs Goldman 1.5 bound. Same NTRS 19680010807. Literature rows stay SCOPING / PREDICTED.",
                )
            )
    ratio_of = of.get("w2_over_w1")
    if ratio_of is not None:
        try:
            ratio_f = float(ratio_of)
        except (TypeError, ValueError):
            ratio_f = None
        else:
            findings.append(
                {
                    "id": "of_w2_over_w1",
                    "title": "OF |W2|/|W1| vs meanline 1.0",
                    "value": ratio_f,
                    "bound": 1.0,
                    "unit": "—",
                    "pct_vs_bound": (ratio_f - 1.0) * 100.0,
                    "status": "INFO",
                    "ntrs_id": None,
                    "report": "this-case OpenFOAM station — not a NASA id. Meanline locks |W2|/|W1|=1; OF will not.",
                    "url": None,
                    "predicted": True,
                    "authority": AUTHORITY_SCOPING,
                    "note": of.get("probe") or "INFO. Not a NASA find. No invented NTRS id.",
                }
            )

    if cfd_flags:
        for key, title in (("mesh_ok", "CFD flag mesh_ok"), ("force_plateau", "CFD flag force_plateau")):
            if key not in cfd_flags:
                continue
            ok = bool(cfd_flags.get(key))
            findings.append(
                {
                    "id": f"cfd_{key}",
                    "title": title,
                    "value": ok,
                    "bound": True,
                    "unit": "—",
                    "pct_vs_bound": None,
                    "status": "NOMINAL" if ok else "HIGH",
                    "ntrs_id": None,
                    "report": "this-case CFD flag — not literature; does not upgrade NTRS rows to FIELD_CFD",
                    "url": None,
                    "predicted": True,
                    "authority": AUTHORITY_SCOPING,
                    "note": "Extra row when a solve exists. Literature rows stay SCOPING / PREDICTED.",
                }
            )
    return findings
