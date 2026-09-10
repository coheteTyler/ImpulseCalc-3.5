"""Local one-page UI on port 8766. This machine only — not a share link.

Outline remeshes without OpenFOAM (SCOPING). Mesh and solve are background jobs.
Missing OF degrades honestly. Does not import ImpulseCalc or LPRE.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .case import acquire_job_tree, clear_job_trees, release_job_tree
from .job import ROOT
from .ofenv import openfoam_available
from .preview import (
    APP_NAME,
    APP_OUTPUT,
    AUTHORITY_FIELD_CFD,
    AUTHORITY_SCOPING,
    authority_from_report,
    knobs_to_job,
    outline_from_knobs,
    write_preview,
)

HOST = "127.0.0.1"
PORT = 8766
APP_HTML = ROOT / "viewer" / "app.html"
MAIN35_HTML = ROOT / "viewer" / "main35.html"
FILTERS_HTML = ROOT / "viewer" / "filters.html"
PREVIEW_CASE = ROOT / APP_OUTPUT / "openfoam_cases" / APP_NAME
PREVIEW_OUT = ROOT / APP_OUTPUT
# Outline (metal cascade) and mesh wire must stay on distinct paths.
LAST_OUTLINE_PNG = PREVIEW_CASE / "outline_preview.png"
LAST_MESH_PNG = PREVIEW_CASE / "mesh_preview.png"
# Legacy alias: older code / tests may still refer to LAST_PNG as the browser outline.
LAST_PNG = LAST_OUTLINE_PNG
LAST_JOB = ROOT / APP_OUTPUT / f"{APP_NAME}.json"
PROFILE_PATH = ROOT / APP_OUTPUT / "ic35_profile.json"
LAST_REPORT = ROOT / APP_OUTPUT / f"{APP_NAME}_report.json"
PLOT_NAMES = frozenset({"force_history.png", "contour_p.png", "contour_U.png", "contour_stream.png", "contour_M.png", "contour_shock.png", "contour_T.png", "wall_cp_blade0.png", "triangles.png"})
PLOT_DIRS = (
    ROOT / APP_OUTPUT / "plots",
    ROOT / APP_OUTPUT / "viewer_plots",
)


def resolve_plot(name: str) -> Path | None:
    """Serve a named field-plot PNG from this-job plots dirs. No path traversal."""
    base = Path(name).name
    if base != name or base not in PLOT_NAMES:
        return None
    for d in PLOT_DIRS:
        cand = d / base
        if cand.is_file():
            return cand
    return None


def resolve_outline_png() -> Path | None:
    """Last cascade-metal outline preview (not the mesh wire)."""
    for cand in (
        LAST_OUTLINE_PNG,
        PREVIEW_OUT / "outline_preview.png",
        PREVIEW_CASE / "outline_preview.png",
    ):
        if cand.is_file():
            return cand
    return None


def resolve_mesh_png() -> Path | None:
    """Last body-fitted mesh wire preview (not the cascade outline)."""
    for cand in (
        LAST_MESH_PNG,
        PREVIEW_OUT / "mesh_preview.png",
        PREVIEW_CASE / "mesh_preview.png",
        ROOT / APP_OUTPUT / "plots" / "mesh_preview.png",
        ROOT / APP_OUTPUT / "viewer_plots" / "mesh_preview.png",
    ):
        if cand.is_file():
            return cand
    return None


def list_field_plots() -> list[dict[str, str]]:
    """Existing CFD field PNGs for the Fields tab."""
    out: list[dict[str, str]] = []
    for name in sorted(PLOT_NAMES):
        if resolve_plot(name) is not None:
            out.append({"name": name, "url": f"/plot/{name}"})
    return out


_RUNNING = frozenset({"meshing", "solving"})

_lock = threading.Lock()
_IDLE: dict[str, Any] = {
    "phase": "idle",  # idle | meshing | solving | error | done-ok | done-fail
    "kind": None,
    "authority": AUTHORITY_SCOPING,
    "predicted": True,
    "success": False,
    "error": None,
    "of_present": False,
    "result": None,
    "Ft_N": None,
    "Fd_N": None,
    "cfd_ran": False,
    "mesh_ok": None,
    "n_cells": None,
    "failed_checks": None,
    "checkMesh": None,
    "t0": None,
    "elapsed_s": 0.0,
    "heartbeat": "idle",
    "sim_time_s": None,
    "end_time_s": None,
    "n_time_dirs": None,
}
_state: dict[str, Any] = dict(_IDLE)


def reset_state() -> None:
    """Test helper — idle status and drop job-tree holders."""
    with _lock:
        _state.clear()
        _state.update(_IDLE)
        _state["of_present"] = openfoam_available()
    clear_job_trees()


def _public(d: dict[str, Any]) -> dict[str, Any]:
    drop = {"eta_from_cfd", "eta_design_proxy", "eta"}
    out = {k: v for k, v in d.items() if k not in drop}
    out["predicted"] = True
    out["hardware_correlated"] = False
    out.setdefault("authority", AUTHORITY_SCOPING)
    return out


def _safe_error(exc: BaseException) -> str:
    raw = str(exc)
    if isinstance(exc, FileNotFoundError) and "log." in raw:
        return (
            "checkMesh/rhoCentralFoam never started or case was wiped mid-run "
            f"(missing log). {type(exc).__name__}: {raw}"
        )[:800]
    return f"{type(exc).__name__}: {exc}"[:800]


def _fmt_elapsed(seconds: float) -> str:
    s = max(0, int(seconds))
    m, sec = divmod(s, 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"


def _of_progress(case_dir: Path) -> dict[str, Any]:
    """Real CFD sim time from time dirs / log. Omit when unknown — do not synthesize."""
    out: dict[str, Any] = {}
    cdir = Path(case_dir)
    if not cdir.is_dir():
        return out
    cd = cdir / "system" / "controlDict"
    if cd.is_file():
        try:
            m = re.search(r"endTime\s+([0-9.eE+-]+)", cd.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            m = None
        if m:
            try:
                out["end_time_s"] = float(m.group(1))
            except ValueError:
                pass
    latest = 0.0
    n = 0
    try:
        kids = list(cdir.iterdir())
    except OSError:
        return out
    for p in kids:
        if not p.is_dir():
            continue
        try:
            t = float(p.name)
        except ValueError:
            continue
        if t > 0 and ((p / "p").is_file() or (p / "U").is_file()):
            n += 1
            if t > latest:
                latest = t
    if latest <= 0:
        lp = cdir / "log.rhoCentralFoam"
        if lp.is_file():
            try:
                txt = lp.read_text(encoding="utf-8", errors="replace")
            except OSError:
                txt = ""
            hits = re.findall(r"(?m)^Time\s*=\s*([0-9.eE+-]+)", txt)
            if hits:
                try:
                    latest = float(hits[-1])
                except ValueError:
                    latest = 0.0
    if latest > 0:
        out["sim_time_s"] = latest
    if n:
        out["n_time_dirs"] = n
    return out


def _foam_log_tail(case_dir: Path) -> dict[str, Any]:
    """Disk truth for a live or finished rhoCentralFoam after RAM reset."""
    lp = Path(case_dir) / "log.rhoCentralFoam"
    out: dict[str, Any] = {"exists": False, "ended": False, "clock_s": None, "fresh": False}
    if not lp.is_file():
        return out
    out["exists"] = True
    try:
        age = time.time() - lp.stat().st_mtime
        out["fresh"] = age < 90.0
        with lp.open("rb") as f:
            f.seek(0, 2)
            n = f.tell()
            f.seek(max(0, n - 16384))
            txt = f.read().decode("utf-8", "replace")
    except OSError:
        return out
    if re.search(r"(?m)^End\s*$", txt):
        out["ended"] = True
    hits = re.findall(r"ClockTime\s*=\s*([0-9.]+)", txt)
    if hits:
        try:
            out["clock_s"] = float(hits[-1])
        except ValueError:
            pass
    return out


def _recover_phase_from_disk(snap: dict[str, Any], case_dir: Path) -> None:
    """Serve restart must not paint idle over dumps / a live foam."""
    if str(snap.get("phase") or "idle") not in ("idle", ""):
        return
    lg = _foam_log_tail(case_dir)
    stime, etime = snap.get("sim_time_s"), snap.get("end_time_s")
    complete = bool(
        lg.get("ended")
        and stime is not None
        and etime is not None
        and float(etime) > 0
        and float(stime) >= 0.98 * float(etime)
    )
    clock = lg.get("clock_s")
    if lg.get("exists") and not lg.get("ended") and lg.get("fresh"):
        snap["phase"] = "solving"
        snap["kind"] = snap.get("kind") or "solve"
        if clock is not None:
            snap["elapsed_s"] = float(clock)
        return
    if complete:
        snap["phase"] = "done-ok"
        if clock is not None:
            snap["elapsed_s"] = float(clock)
        return
    if lg.get("exists") and (snap.get("n_time_dirs") or stime):
        snap["phase"] = "done-fail"
        snap["error"] = snap.get("error") or (
            "solve dropped from RAM (serve restart). Dumps on disk still shown."
        )
        if clock is not None:
            snap["elapsed_s"] = float(clock)


def _heartbeat_line(st: dict[str, Any]) -> str:
    phase = str(st.get("phase") or "idle")
    elapsed = float(st.get("elapsed_s") or 0.0)
    clock = _fmt_elapsed(elapsed)
    extra = ""
    stime, etime, nd = st.get("sim_time_s"), st.get("end_time_s"), st.get("n_time_dirs")
    if stime is not None and etime is not None:
        extra = f" · CFD t={stime:.4g}/{etime:.4g} s ({nd or 0} dumps)"
    elif nd:
        extra = f" · {nd} dumps"
    if phase in _RUNNING:
        return f"elapsed {clock} · still running · not stuck{extra}"
    if phase == "done-ok":
        return f"done after {clock}{extra}"
    if phase in ("done-fail", "error"):
        err = st.get("error") or "failed"
        return f"failed after {clock} — {err}"
    if extra:
        return f"last CFD{extra}"
    return "idle"


def status_payload() -> dict[str, Any]:
    _rehydrate_state_from_disk()
    with _lock:
        snap = dict(_state)
        t0 = snap.get("t0")
        phase = snap.get("phase")
        if phase in _RUNNING and t0:
            snap["elapsed_s"] = max(0.0, time.time() - float(t0))
        elif snap.get("elapsed_s") is None:
            snap["elapsed_s"] = 0.0
    ofp = _of_progress(PREVIEW_CASE)
    if "sim_time_s" in ofp:
        snap["sim_time_s"] = ofp["sim_time_s"]
    if "end_time_s" in ofp:
        snap["end_time_s"] = ofp["end_time_s"]
    if "n_time_dirs" in ofp:
        snap["n_time_dirs"] = ofp["n_time_dirs"]
    _recover_phase_from_disk(snap, PREVIEW_CASE)
    snap["of_present"] = openfoam_available()
    snap["predicted"] = True
    snap["hardware_correlated"] = False
    snap["heartbeat"] = _heartbeat_line(snap)
    if not _status_forces_live(snap):
        _clear_status_forces(snap)
    res = snap.get("result")
    if isinstance(res, dict) and res.get("ntrs_checks") is not None:
        snap["ntrs_checks"] = res["ntrs_checks"]
    if isinstance(res, dict):
        for k in (
            "euler_work_j_kg",
            "power_w",
            "u_m_s",
            "Mw1",
            "Mw1_OF",
            "of_power_w",
            "of_station",
            "loss_scoping",
        ):
            if snap.get(k) is None and res.get(k) is not None:
                snap[k] = res[k]
    snap["eta_from_cfd"] = None
    # Mesh summary from last mesh job / report (independent of CFD fields).
    res = snap.get("result") if isinstance(snap.get("result"), dict) else {}
    if snap.get("mesh_ok") is None and isinstance(res.get("flags"), dict):
        snap["mesh_ok"] = res["flags"].get("mesh_ok")
    if snap.get("checkMesh") is None and res.get("checkMesh") is not None:
        snap["checkMesh"] = res.get("checkMesh")
    if snap.get("failed_checks") is None and snap.get("checkMesh") is not None:
        snap["failed_checks"] = snap.get("checkMesh")
    if snap.get("n_cells") is None and isinstance(res.get("mesh"), dict):
        snap["n_cells"] = res["mesh"].get("n_cells")
    snap["outline_available"] = resolve_outline_png() is not None
    snap["mesh_available"] = resolve_mesh_png() is not None
    snap["outline_png"] = "/outline.png"
    snap["mesh_png"] = "/mesh.png"
    snap["field_plots"] = list_field_plots() if snap.get("cfd_ran") else []
    for k in (
        "phase",
        "elapsed_s",
        "heartbeat",
        "of_present",
        "predicted",
        "authority",
        "success",
        "kind",
        "cfd_ran",
        "mesh_ok",
        "n_cells",
        "failed_checks",
        "checkMesh",
        "Ft_N",
        "Fd_N",
        "error",
        "euler_work_j_kg",
        "power_w",
        "u_m_s",
        "Mw1",
        "Mw1_OF",
        "of_power_w",
        "loss_scoping",
    ):
        snap.setdefault(k, None)
    return snap


def _forces_from_report(rep: dict[str, Any] | None) -> tuple[Any, Any]:
    if not rep:
        return None, None
    blades = (rep.get("forces") or {}).get("blades") or []
    if not blades:
        return None, None
    b0 = blades[0]
    return b0.get("Ft_N"), b0.get("Fd_N")


def _loss_from_report(rep: dict[str, Any] | None, *, include_of: bool) -> dict[str, Any]:
    """Meanline PREDICTED loss fields; OF extras only after a real solve."""
    out: dict[str, Any] = {
        "euler_work_j_kg": None,
        "power_w": None,
        "u_m_s": None,
        "Mw1": None,
        "Mw1_OF": None,
        "of_power_w": None,
        "of_station": None,
        "loss_scoping": None,
        "eta_from_cfd": None,
    }
    if not isinstance(rep, dict):
        return out
    ml = rep.get("meanline") or {}
    if not isinstance(ml, dict):
        ml = {}

    def _pick(*keys: str) -> Any:
        for src in (rep, ml):
            for k in keys:
                v = src.get(k)
                if v is not None:
                    return v
        return None

    out["euler_work_j_kg"] = _pick("euler_work_j_kg")
    out["power_w"] = _pick("power_w")
    out["u_m_s"] = _pick("u_m_s")
    out["Mw1"] = _pick("Mw1")
    out["loss_scoping"] = rep.get("loss_scoping") or ml.get("loss_scoping")
    out["eta_from_cfd"] = None
    if include_of:
        of_st = rep.get("of_station")
        out["of_station"] = of_st
        mw_of = rep.get("Mw1_OF")
        if mw_of is None and isinstance(of_st, dict):
            mw_of = of_st.get("Mw1_OF")
        out["Mw1_OF"] = mw_of
        ft, _fd = _forces_from_report(rep)
        ofp = rep.get("of_power_w")
        u = out.get("u_m_s")
        if ofp is None and ft is not None and u is not None:
            try:
                ofp = float(ft) * float(u)
            except (TypeError, ValueError):
                ofp = None
        out["of_power_w"] = ofp
    return out


def _meanline_loss_from_job_file() -> dict[str, Any]:
    """PREDICTED meanline from last knobs_preview.json. Never η-from-CFD."""
    if not LAST_JOB.is_file():
        return {}
    try:
        job = json.loads(LAST_JOB.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(job, dict):
        return {}
    from .meanline import compute_meanline

    ml = compute_meanline(job)
    return {
        "euler_work_j_kg": ml.euler_work_j_kg,
        "power_w": ml.power_w,
        "u_m_s": ml.u_m_s,
        "Mw1": ml.Mw1,
        "loss_scoping": ml.loss_scoping,
        "eta_from_cfd": None,
        "_job": job,
        "_ml": ml,
    }


def _foam_case_complete(case_dir: Path) -> bool:
    lg = _foam_log_tail(case_dir)
    ofp = _of_progress(case_dir)
    stime, etime = ofp.get("sim_time_s"), ofp.get("end_time_s")
    return bool(
        lg.get("ended")
        and stime is not None
        and etime is not None
        and float(etime) > 0
        and float(stime) >= 0.98 * float(etime)
    )


def _load_last_report() -> dict[str, Any] | None:
    if not LAST_REPORT.is_file():
        return None
    try:
        rep = json.loads(LAST_REPORT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return rep if isinstance(rep, dict) else None


def _rehydrate_state_from_disk() -> None:
    """After serve restart: last job JSON + OF Ft so Losses is not dashes."""
    with _lock:
        if _state.get("phase") in _RUNNING:
            return
        res = _state.get("result")
        have_result = isinstance(res, dict)
        have_of = have_result and res.get("Mw1_OF") is not None
        cfd_ram = bool(_state.get("cfd_ran"))
        kind = _state.get("kind")
    if kind in ("mesh", "skip_solve"):
        return
    complete = _foam_case_complete(PREVIEW_CASE)
    if have_result:
        if cfd_ram and (not have_of) and complete:
            try:
                from .post import extract_of_stations
                from .meanline import compute_meanline
                from .ntrs_checks import evaluate as ntrs_evaluate

                job = None
                ml = None
                if LAST_JOB.is_file():
                    job = json.loads(LAST_JOB.read_text(encoding="utf-8"))
                    ml = compute_meanline(job)
                of_st = extract_of_stations(PREVIEW_CASE, job)
            except Exception:
                of_st = None
                ml = None
                job = None
            with _lock:
                r = _state.get("result")
                if not isinstance(r, dict):
                    return
                r = dict(r)
                r["of_station"] = of_st
                if isinstance(of_st, dict):
                    r["Mw1_OF"] = of_st.get("Mw1_OF")
                if r.get("of_power_w") is None and r.get("Ft_N") is not None and r.get("u_m_s") is not None:
                    try:
                        r["of_power_w"] = float(r["Ft_N"]) * float(r["u_m_s"])
                    except (TypeError, ValueError):
                        pass
                if ml is not None and isinstance(of_st, dict) and of_st.get("Mw1_OF") is not None:
                    r["ntrs_checks"] = ntrs_evaluate(
                        ml, job, cfd_flags=r.get("flags"), of_station=of_st
                    )
                _state["result"] = r
                _state["of_station"] = of_st
                _state["Mw1_OF"] = r.get("Mw1_OF")
                _state["of_power_w"] = r.get("of_power_w")
        return
    job_pack = _meanline_loss_from_job_file()
    job = job_pack.pop("_job", None) if job_pack else None
    ml = job_pack.pop("_ml", None) if job_pack else None
    rep = _load_last_report()
    ran = bool(complete and _report_cfd_this_click(rep))
    loss = _loss_from_report(rep, include_of=ran)
    for k, v in (job_pack or {}).items():
        if loss.get(k) is None and v is not None:
            loss[k] = v
    of_st = loss.get("of_station") if ran else None
    if ran and (not isinstance(of_st, dict) or of_st.get("Mw1_OF") is None):
        try:
            from .post import extract_of_stations

            of_st = extract_of_stations(PREVIEW_CASE, job)
        except Exception:
            of_st = None
        loss["of_station"] = of_st
        if isinstance(of_st, dict):
            loss["Mw1_OF"] = of_st.get("Mw1_OF")
    ft = fd = None
    if ran and rep:
        ft, fd = _forces_from_report(rep)
        u = loss.get("u_m_s")
        if loss.get("of_power_w") is None and ft is not None and u is not None:
            try:
                loss["of_power_w"] = float(ft) * float(u)
            except (TypeError, ValueError):
                pass
    ntrs = (rep or {}).get("ntrs_checks") if isinstance(rep, dict) else None
    if ran and ml is not None:
        from .ntrs_checks import evaluate as ntrs_evaluate

        flags = (rep or {}).get("flags") if isinstance(rep, dict) else None
        ntrs = ntrs_evaluate(ml, job, cfd_flags=flags, of_station=of_st if isinstance(of_st, dict) else None)
    elif (not ntrs) and ml is not None:
        from .ntrs_checks import evaluate as ntrs_evaluate

        ntrs = ntrs_evaluate(ml, job)
    auth = authority_from_report(rep) if ran else AUTHORITY_SCOPING
    with _lock:
        if _state.get("phase") in _RUNNING:
            return
        result = {
            "success": bool((rep or {}).get("success")) if ran else False,
            "predicted": True,
            "authority": auth,
            "flags": (rep or {}).get("flags") if isinstance(rep, dict) else None,
            "errors": (rep or {}).get("errors") if isinstance(rep, dict) else None,
            "Ft_N": ft,
            "Fd_N": fd,
            "name": (rep or {}).get("name") if isinstance(rep, dict) else None,
            "checkMesh": ((rep or {}).get("checkMesh") or {}).get("failed_checks") if isinstance(rep, dict) else None,
            "solve_ok": ((rep or {}).get("flags") or {}).get("solve_ok") if isinstance(rep, dict) else None,
            "hardware_correlated": False,
            "ntrs_checks": ntrs,
            **loss,
            "eta_from_cfd": None,
        }
        _state["result"] = result
        for k, v in loss.items():
            _state[k] = v
        _state["Ft_N"] = ft
        _state["Fd_N"] = fd
        _state["eta_from_cfd"] = None
        if ran:
            _state["cfd_ran"] = True
            _state["kind"] = _state.get("kind") or "solve"
            _state["authority"] = auth
            _state["success"] = bool((rep or {}).get("success")) and auth == AUTHORITY_FIELD_CFD
            if str(_state.get("phase") or "idle") in ("idle", ""):
                _state["phase"] = "done-ok"
            lg = _foam_log_tail(PREVIEW_CASE)
            if lg.get("clock_s") is not None and not _state.get("elapsed_s"):
                _state["elapsed_s"] = float(lg["clock_s"])


def _report_cfd_this_click(rep: dict[str, Any] | None) -> bool:
    """True only after rhoCentralFoam this click — not skip_solve leftover postProcessing."""
    if not rep:
        return False
    solve = rep.get("solve") or {}
    note = str(solve.get("note") or "").lower()
    if "skip_solve" in note or "not run" in note:
        return False
    return solve.get("rc") is not None


def _status_forces_live(st: dict[str, Any]) -> bool:
    """Ft/Fd may appear only after a real rhoCentralFoam this click."""
    if st.get("kind") in ("mesh", "skip_solve"):
        return False
    return bool(st.get("cfd_ran"))


def _clear_status_forces(st: dict[str, Any]) -> None:
    st["Ft_N"] = None
    st["Fd_N"] = None
    st["of_power_w"] = None
    st["Mw1_OF"] = None
    st["of_station"] = None
    res = st.get("result")
    if isinstance(res, dict):
        copied = dict(res)
        copied["Ft_N"] = None
        copied["Fd_N"] = None
        copied["of_power_w"] = None
        copied["Mw1_OF"] = None
        copied["of_station"] = None
        st["result"] = copied


def _apply_report(rep: dict[str, Any], *, cfd_this_click: bool = True) -> None:
    """Copy report into status. Mesh-only / skip_solve never copies leftover Ft/Fd."""
    ran = bool(cfd_this_click) and _report_cfd_this_click(rep)
    ft, fd = _forces_from_report(rep) if ran else (None, None)
    auth = authority_from_report(rep)
    loss = _loss_from_report(rep, include_of=ran)
    _state["result"] = {
        "success": bool(rep.get("success")),
        "predicted": True,
        "authority": auth,
        "flags": rep.get("flags"),
        "errors": rep.get("errors"),
        "Ft_N": ft,
        "Fd_N": fd,
        "name": rep.get("name"),
        "checkMesh": (rep.get("checkMesh") or {}).get("failed_checks"),
        "solve_ok": (rep.get("flags") or {}).get("solve_ok"),
        "hardware_correlated": False,
        "ntrs_checks": rep.get("ntrs_checks"),
        **loss,
        "eta_from_cfd": None,
    }
    _state["authority"] = auth
    _state["success"] = bool(rep.get("success")) and auth == AUTHORITY_FIELD_CFD
    _state["predicted"] = True
    _state["cfd_ran"] = ran
    _state["Ft_N"] = ft
    _state["Fd_N"] = fd
    for k, v in loss.items():
        _state[k] = v
    _state["eta_from_cfd"] = None
    _state["error"] = None if not (rep.get("errors")) else "; ".join(str(e) for e in (rep.get("errors") or [])[:4])


def _begin_job(phase: str, kind: str) -> bool:
    with _lock:
        if _state["phase"] in _RUNNING:
            return False
        _state["phase"] = phase
        _state["kind"] = kind
        _state["t0"] = time.time()
        _state["elapsed_s"] = 0.0
        _state["error"] = None
        _state["heartbeat"] = "elapsed 0:00 · still running · not stuck"
        _state["sim_time_s"] = None
        _state["end_time_s"] = None
        _state["n_time_dirs"] = None
        _state["success"] = False
        _state["authority"] = AUTHORITY_SCOPING
        _state["cfd_ran"] = False
        _state["mesh_ok"] = None
        _state["n_cells"] = None
        _state["failed_checks"] = None
        _state["checkMesh"] = None
        _state["Ft_N"] = None
        _state["Fd_N"] = None
        _state["result"] = None
    return True


def _complete_job(phase: str, error: str | None = None, *, mesh: bool = False) -> None:
    now = time.time()
    with _lock:
        t0 = _state.get("t0")
        _state["elapsed_s"] = (now - float(t0)) if t0 else 0.0
        _state["phase"] = phase
        if error:
            _state["error"] = error
        if mesh:
            _state["authority"] = AUTHORITY_SCOPING
            _state["success"] = False
            _state["cfd_ran"] = False
            _clear_status_forces(_state)
        _state["heartbeat"] = _heartbeat_line(_state)


def _run_mesh(knobs: dict[str, Any]) -> dict[str, Any]:
    from .run import run_job

    job = knobs_to_job(knobs)
    info = write_preview(job)
    job_path = Path(info["job_json"])
    of = openfoam_available()
    if not of:
        return {
            "ok": True,
            "authority": AUTHORITY_SCOPING,
            "predicted": True,
            "success": False,
            "cfd_ran": False,
            "of_present": False,
            "mesh_ok": info.get("n_cells") is not None,
            "failed_checks": None,
            "note": "OpenFOAM not on this machine. Case written (SCOPING). Building a case is not running CFD.",
            "png": "/mesh.png",
            "outline_png": "/outline.png",
            "mesh_png": "/mesh.png",
            "n_cells": info.get("n_cells"),
            "mesh_kind": info.get("mesh_kind"),
            "hardware_correlated": False,
            "report": None,
        }
    rep = run_job(job_path, skip_solve=True)
    return {
        "ok": True,
        "authority": AUTHORITY_SCOPING,  # checkMesh is not FIELD_CFD
        "predicted": True,
        "success": False,
        "cfd_ran": False,
        "of_present": True,
        "mesh_ok": (rep.get("flags") or {}).get("mesh_ok"),
        "failed_checks": (rep.get("checkMesh") or {}).get("failed_checks"),
        "errors": rep.get("errors"),
        "n_cells": (rep.get("mesh") or {}).get("n_cells"),
        "png": "/mesh.png",
        "outline_png": "/outline.png",
        "mesh_png": "/mesh.png",
        "note": "checkMesh ran. Building a case / meshing is not FIELD_CFD.",
        "hardware_correlated": False,
        "report": rep,
    }


def _errors_from(out: dict[str, Any]) -> str | None:
    if out.get("error"):
        return str(out["error"])
    errs = out.get("errors") or []
    if errs:
        return "; ".join(str(e) for e in errs[:4])
    return None


def _mesh_worker(knobs: dict[str, Any]) -> None:
    acquire_job_tree(PREVIEW_CASE)
    try:
        out = _run_mesh(knobs)
        err = _errors_from(out)
        with _lock:
            rep = out.get("report")
            if isinstance(rep, dict):
                _apply_report(rep, cfd_this_click=False)
            _state["of_present"] = bool(out.get("of_present"))
            _state["authority"] = AUTHORITY_SCOPING
            _state["success"] = False
            _state["cfd_ran"] = False
            _state["mesh_ok"] = out.get("mesh_ok")
            _state["n_cells"] = out.get("n_cells")
            _state["failed_checks"] = out.get("failed_checks")
            _state["checkMesh"] = out.get("failed_checks")
            _clear_status_forces(_state)
            if err:
                _state["error"] = err
        phase = "done-fail" if (err or not out.get("ok")) else "done-ok"
        _complete_job(phase, error=err, mesh=True)
    except Exception as exc:
        _complete_job("error", error=_safe_error(exc), mesh=True)
    finally:
        release_job_tree(PREVIEW_CASE)


def _solve_worker(knobs: dict[str, Any]) -> None:
    from .run import run_job

    acquire_job_tree(PREVIEW_CASE)
    try:
        job = knobs_to_job(knobs)
        job.setdefault("cfd", {})
        job["cfd"].pop("smoke_end_s", None)
        t_req = knobs.get("t_end_s")
        t_min, t_max = 7.9827173e-05, 2.6609057819508715e-04
        if t_req not in (None, ""):
            job["cfd"]["end_time_s"] = min(t_max, max(t_min, float(t_req)))
        else:
            job["cfd"]["end_time_s"] = t_min
        info = write_preview(job)
        job_path = Path(info["job_json"])
        of = openfoam_available()
        if not of:
            _complete_job(
                "error",
                error=(
                    "OpenFOAM not on this machine. Outline still works (SCOPING). "
                    "FIELD_CFD requires rhoCentralFoam on this case."
                ),
            )
            return
        rep = run_job(job_path, skip_solve=False)
        err = None if not (rep.get("errors")) else "; ".join(str(e) for e in (rep.get("errors") or [])[:4])
        with _lock:
            _apply_report(rep)
            _state["of_present"] = True
        auth = authority_from_report(rep)
        ok = bool(rep.get("success")) and auth == AUTHORITY_FIELD_CFD and not err
        _complete_job("done-ok" if ok else "done-fail", error=err)
    except Exception as exc:
        _complete_job("error", error=_safe_error(exc))
    finally:
        release_job_tree(PREVIEW_CASE)


class Handler(BaseHTTPRequestHandler):
    server_version = "ImpulseCalc3/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code: int, body: bytes, ctype: str, extra: dict[str, str] | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-ImpulseCalc3-Authority", str(_state.get("authority") or AUTHORITY_SCOPING))
        if extra:
            for k, v in extra.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: dict[str, Any]) -> None:
        payload = _public(obj)
        raw = json.dumps(payload, indent=2, default=str).encode("utf-8") + b"\n"
        self._send(code, raw, "application/json; charset=utf-8")

    def _read_json(self) -> dict[str, Any]:
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return {}
        if n > 2_000_000:
            raise ValueError("body too large")
        raw = self.rfile.read(n)
        if not raw:
            return {}
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON object required")
        return data

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html", "/main35.html"):
            html = MAIN35_HTML.read_bytes() if MAIN35_HTML.is_file() else b"<h1>missing viewer/main35.html</h1>"
            self._send(200, html, "text/html; charset=utf-8")
            return
        if path == "/legacy.html":
            html = APP_HTML.read_bytes() if APP_HTML.is_file() else b"<h1>missing viewer/app.html</h1>"
            self._send(200, html, "text/html; charset=utf-8")
            return
        if path == "/outline.png":
            png = resolve_outline_png()
            if png is not None:
                self._send(200, png.read_bytes(), "image/png")
            else:
                self._json(404, {"ok": False, "error": "no outline yet", "authority": AUTHORITY_SCOPING})
            return
        if path == "/mesh.png":
            png = resolve_mesh_png()
            if png is not None:
                self._send(200, png.read_bytes(), "image/png")
            else:
                self._json(404, {"ok": False, "error": "no mesh preview yet — Run mesh", "authority": AUTHORITY_SCOPING})
            return
        if path == "/plots":
            with _lock:
                ran = bool(_state.get("cfd_ran"))
            self._json(
                200,
                {
                    "ok": True,
                    "cfd_ran": ran,
                    "plots": list_field_plots() if ran else [],
                    "authority": AUTHORITY_SCOPING if not ran else AUTHORITY_FIELD_CFD,
                    "predicted": True,
                    "note": None if ran else "No solve yet — Run solve when OF present",
                },
            )
            return
        if path.startswith("/plot/"):
            name = path[len("/plot/"):]
            png = resolve_plot(name)
            if png is not None:
                self._send(200, png.read_bytes(), "image/png")
            else:
                self._json(
                    404,
                    {
                        "ok": False,
                        "error": "plot not found",
                        "name": Path(name).name,
                        "authority": AUTHORITY_SCOPING,
                    },
                )
            return
        if path == "/status":
            self._json(200, status_payload())
            return
        if path == "/result":
            _rehydrate_state_from_disk()
            if LAST_REPORT.is_file():
                try:
                    rep = json.loads(LAST_REPORT.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    self._json(500, {"ok": False, "error": "report JSON unreadable", "authority": AUTHORITY_SCOPING})
                    return
                with _lock:
                    allow_forces = _status_forces_live(dict(_state))
                    ram = dict(_state.get("result") or {})
                ft, fd = _forces_from_report(rep) if allow_forces else (None, None)
                loss = _loss_from_report(rep, include_of=allow_forces)
                for k, v in ram.items():
                    if k in loss and loss.get(k) is None and v is not None:
                        loss[k] = v
                if allow_forces and loss.get("Mw1_OF") is None:
                    of_st = ram.get("of_station") or loss.get("of_station")
                    if isinstance(of_st, dict):
                        loss["of_station"] = of_st
                        loss["Mw1_OF"] = of_st.get("Mw1_OF")
                ntrs = ram.get("ntrs_checks") if ram.get("ntrs_checks") is not None else rep.get("ntrs_checks")
                payload = {
                    "ok": True,
                    "authority": authority_from_report(rep) if allow_forces else AUTHORITY_SCOPING,
                    "predicted": True,
                    "success": bool(rep.get("success")),
                    "flags": rep.get("flags"),
                    "errors": rep.get("errors"),
                    "Ft_N": ft,
                    "Fd_N": fd,
                    "hardware_correlated": False,
                    "name": rep.get("name"),
                    "ntrs_checks": ntrs,
                    "eta_from_cfd": None,
                }
                payload.update(loss)
                payload["eta_from_cfd"] = None
                self._json(200, payload)
                return
            self._json(
                200,
                {
                    "ok": True,
                    "authority": AUTHORITY_SCOPING,
                    "predicted": True,
                    "success": False,
                    "Ft_N": None,
                    "Fd_N": None,
                    "euler_work_j_kg": None,
                    "power_w": None,
                    "of_power_w": None,
                    "Mw1": None,
                    "Mw1_OF": None,
                    "eta_from_cfd": None,
                },
            )
            return
        if path in ("/filters.html", "/filters"):
            html = FILTERS_HTML.read_bytes() if FILTERS_HTML.is_file() else b"<h1>missing viewer/filters.html</h1>"
            self._send(200, html, "text/html; charset=utf-8")
            return
        if path == "/api/filters/schema":
            from .rts_filters import schema_payload
            self._json(200, schema_payload())
            return
        if path == "/api/profile":
            from .rts_filters import defaults_profile
            if PROFILE_PATH.is_file():
                try:
                    prof = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    self._json(500, {"ok": False, "error": "profile JSON unreadable", "authority": AUTHORITY_SCOPING})
                    return
            else:
                prof = defaults_profile()
            from .rts_filters import sanitize_impulse_betas
            prof = sanitize_impulse_betas(prof)
            if PROFILE_PATH.is_file():
                PROFILE_PATH.write_text(json.dumps(prof, indent=2), encoding="utf-8")
            self._json(200, {"ok": True, "profile": prof, "authority": AUTHORITY_SCOPING, "predicted": True})
            return
        if path == "/defaults":
            self._json(
                200,
                {
                    "family": "impulse_bucket",
                    "hu_mm": 5.0,
                    "hl_mm": 2.2,
                    "le_mm": 0.4,
                    "te_mm": 0.0,
                    "lin_mm": 4.25,
                    "lout_mm": 4.25,
                    "t_mm": 1.4,
                    "r_tr_mm": 3.5,
                    "r_main_mm": 4.8,
                    "psi_tr_deg": 0.0,
                    "beta1": 72.0,
                    "beta2": -72.0,
                    "s_mm": 8.1,
                    "sigma": 1.234,
                    "packing": {"s_mm": 8.1, "sigma": 1.234, "c": 0.01, "r": 0.0375},
                    "of_present": openfoam_available(),
                    "authority": AUTHORITY_SCOPING,
                    "disclaimer": "Engineering aid, not a flight certificate. PREDICTED is not a design load.",
                },
            )
            return
        self._json(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            knobs = self._read_json()
        except Exception as exc:
            self._json(400, {"ok": False, "error": _safe_error(exc), "authority": AUTHORITY_SCOPING})
            return
        if path == "/api/passage_depth/auto":
            from .rts_filters import defaults_profile, sanitize_impulse_betas, auto_passage_depth_for_profile
            base = defaults_profile()
            if PROFILE_PATH.is_file():
                try:
                    base = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    pass
            vals = dict(base.get("values") or {})
            incoming = knobs.get("values") if isinstance(knobs.get("values"), dict) else knobs
            if isinstance(incoming, dict):
                for k, v in incoming.items():
                    if k in ("format", "authority", "predicted", "article", "rts_map", "values"):
                        continue
                    vals[k] = v
            base["values"] = vals
            base = sanitize_impulse_betas(base)
            base, meas = auto_passage_depth_for_profile(base)
            PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
            PROFILE_PATH.write_text(json.dumps(base, indent=2), encoding="utf-8")
            self._json(
                200,
                {
                    "ok": bool(meas.get("ok")),
                    "profile": base,
                    "measure": meas,
                    "passage_depth_mm": (base.get("values") or {}).get("passage_depth_mm"),
                    "authority": AUTHORITY_SCOPING,
                    "predicted": True,
                },
            )
            return
        if path == "/api/profile/update":
            from .rts_filters import defaults_profile, profile_to_ic3_knobs
            base = defaults_profile()
            if PROFILE_PATH.is_file():
                try:
                    base = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    pass
            vals = dict(base.get("values") or {})
            incoming = knobs.get("values") if isinstance(knobs.get("values"), dict) else knobs
            if isinstance(incoming, dict):
                for k, v in incoming.items():
                    if k in ("format", "authority", "predicted", "article", "rts_map", "values"):
                        continue
                    vals[k] = v
            base["values"] = vals
            from .rts_filters import sanitize_impulse_betas, apply_constant_passage_to_profile
            base = sanitize_impulse_betas(base)
            base, cpw_report = apply_constant_passage_to_profile(base)
            base["format"] = "impulsecalc35_profile_v1"
            base["authority"] = AUTHORITY_SCOPING
            base["predicted"] = True
            PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
            PROFILE_PATH.write_text(json.dumps(base, indent=2), encoding="utf-8")
            mapped = profile_to_ic3_knobs(base)
            mapped["constant_passage_width"] = bool((base.get("values") or {}).get("constant_passage_width"))
            if cpw_report.get("ok") and cpw_report.get("hl_mm") is not None:
                mapped["hl_mm"] = float(cpw_report["hl_mm"])
            self._json(
                200,
                {
                    "ok": True,
                    "profile": base,
                    "knobs": mapped,
                    "constant_passage": cpw_report,
                    "path": str(PROFILE_PATH),
                    "authority": AUTHORITY_SCOPING,
                    "predicted": True,
                },
            )
            return
        if path == "/outline":
            with _lock:
                busy = _state["phase"] in _RUNNING
            if busy:
                self._json(
                    200,
                    {
                        "ok": True,
                        "noop": True,
                        "authority": AUTHORITY_SCOPING,
                        "predicted": True,
                        "success": False,
                        "cfd_ran": False,
                        "png": "/outline.png",
                        "note": "outline paused — mesh/solve owns this tree. Keep last PNG.",
                        "hardware_correlated": False,
                    },
                )
                return
            try:
                info = outline_from_knobs(knobs)
            except RuntimeError as exc:
                if "holds this tree" in str(exc):
                    self._json(
                        200,
                        {
                            "ok": True,
                            "noop": True,
                            "authority": AUTHORITY_SCOPING,
                            "predicted": True,
                            "success": False,
                            "cfd_ran": False,
                            "png": "/outline.png",
                            "note": "outline paused — mesh/solve owns this tree. Keep last PNG.",
                            "hardware_correlated": False,
                        },
                    )
                    return
                self._json(
                    400,
                    {
                        "ok": False,
                        "error": _safe_error(exc),
                        "authority": AUTHORITY_SCOPING,
                        "predicted": True,
                        "cfd_ran": False,
                    },
                )
                return
            except Exception as exc:
                self._json(
                    400,
                    {
                        "ok": False,
                        "error": _safe_error(exc),
                        "authority": AUTHORITY_SCOPING,
                        "predicted": True,
                        "cfd_ran": False,
                    },
                )
                return
            # write_preview dict (SCOPING). Browser PNG routes stay distinct.
            out = dict(info)
            out["png"] = "/outline.png"
            out["outline_png"] = "/outline.png"
            out["mesh_png"] = "/mesh.png" if resolve_mesh_png() is not None else None
            out["ok"] = True
            out["authority"] = AUTHORITY_SCOPING
            out["predicted"] = True
            out["success"] = False
            out["cfd_ran"] = False
            out["hardware_correlated"] = False
            self._json(200, out)
            return
        if path == "/mesh":
            if not _begin_job("meshing", "mesh"):
                with _lock:
                    ph = _state["phase"]
                    auth = _state["authority"]
                self._json(409, {"ok": False, "error": f"{ph} running", "authority": auth, "phase": ph})
                return
            t = threading.Thread(target=_mesh_worker, args=(knobs,), daemon=True)
            t.start()
            self._json(
                202,
                {
                    "ok": True,
                    "started": True,
                    "phase": "meshing",
                    "authority": AUTHORITY_SCOPING,
                    "predicted": True,
                    "note": "Background job. Meshing is not FIELD_CFD. Keep this tab open.",
                    "hardware_correlated": False,
                },
            )
            return
        if path == "/solve":
            if not openfoam_available():
                with _lock:
                    _state["phase"] = "error"
                    _state["authority"] = AUTHORITY_SCOPING
                    _state["predicted"] = True
                    _state["success"] = False
                    _state["error"] = (
                        "OpenFOAM not on this machine. Outline still works (SCOPING). "
                        "FIELD_CFD requires rhoCentralFoam on this case."
                    )
                    _state["heartbeat"] = _heartbeat_line(_state)
                    err = _state["error"]
                self._json(
                    200,
                    {
                        "ok": False,
                        "started": False,
                        "phase": "error",
                        "authority": AUTHORITY_SCOPING,
                        "predicted": True,
                        "success": False,
                        "cfd_ran": False,
                        "of_present": False,
                        "error": err,
                        "hardware_correlated": False,
                    },
                )
                return
            if not _begin_job("solving", "solve"):
                with _lock:
                    ph = _state["phase"]
                self._json(409, {"ok": False, "error": f"{ph} already running", "phase": ph})
                return
            t = threading.Thread(target=_solve_worker, args=(knobs,), daemon=True)
            t.start()
            self._json(
                202,
                {
                    "ok": True,
                    "started": True,
                    "phase": "solving",
                    "authority": AUTHORITY_SCOPING,
                    "predicted": True,
                    "note": "Background job. Do not treat a running solve as FIELD_CFD. Keep this tab open.",
                    "hardware_correlated": False,
                },
            )
            return
        self._json(404, {"ok": False, "error": "not found"})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="ImpulseCalc3 local UI (this machine only)")
    p.add_argument("--host", default=HOST, help="bind address (default 127.0.0.1, not a share)")
    p.add_argument("--port", type=int, default=PORT)
    args = p.parse_args(argv)
    _state["of_present"] = openfoam_available()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    shown = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    url = f"http://{shown}:{args.port}/"
    print("ImpulseCalc3 — engineering aid, not a flight certificate.")
    print("This URL is THIS MACHINE ONLY. Not a share link. Do not send it to PURPL.")
    print("Close this window to stop.")
    print(url)
    print(f"OpenFOAM present: {openfoam_available()}  (missing OF → SCOPING outline still works)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
