"""Job panel /status, async /mesh, missing foam log, one writer on the preview tree."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from impulsecalc3.case import (
    acquire_job_tree,
    release_job_tree,
    write_case,
)
from impulsecalc3.geometry import BladeSpec
from impulsecalc3.job import ROOT, domain_x, load_job
from impulsecalc3.meanline import compute_meanline
from impulsecalc3.preview import AUTHORITY_SCOPING, outline_from_knobs
from impulsecalc3.run import FoamLogMissing, read_foam_log, run_job
from impulsecalc3.times import compute_times

CONFIGS = ROOT / "configs"


def _json(url: str, data: dict | None = None, method: str | None = None) -> tuple[int, dict]:
    body = None if data is None else json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method=method or ("POST" if body is not None else "GET"),
        headers={"Content-Type": "application/json"} if body is not None else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"error": raw[:400]}
        return exc.code, parsed


def _spec_times(job):
    g, gas = job["geometry"], job["gas"]
    spec = BladeSpec(
        chord_m=g["chord_m"],
        beta1_metal_deg=g["beta1_flow_deg"],
        beta2_metal_deg=g["beta2_flow_deg"],
        thickness_c=g["thickness_c"],
        le_radius_c=g["le_radius_c"],
        te_radius_c=g["te_radius_c"],
    )
    ml = compute_meanline(job)
    xin, xout = domain_x(job)
    times = compute_times(
        chord_m=g["chord_m"],
        w1_m_s=gas["w1_m_s"],
        gamma=gas["gamma"],
        r_specific=gas["r_specific_j_kg_k"],
        t1_k=gas["t1_k"],
        x_in_m=xin,
        x_out_m=xout,
        n_chords_min=job["cfd"]["n_chords_min"],
    )
    return spec, ml, times


def test_read_foam_log_missing_is_not_filenotfound(tmp_path: Path):
    with pytest.raises(FoamLogMissing, match="never started or case was wiped mid-run"):
        read_foam_log(tmp_path, "log.checkMesh", "checkMesh")
    try:
        read_foam_log(tmp_path, "log.checkMesh", "checkMesh")
    except FileNotFoundError:
        pytest.fail("FileNotFoundError leaked")
    except FoamLogMissing as exc:
        assert "missing log.checkMesh" in str(exc)


def test_missing_checkmesh_log_is_structured_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Fake checkMesh writes no log — run_job must not raise FileNotFoundError."""
    from impulsecalc3 import run as runmod

    monkeypatch.setattr(runmod, "run_foam", lambda args, cwd, log_name, env=None: 0)
    monkeypatch.setattr(runmod, "openfoam_available", lambda: True)
    monkeypatch.setattr(runmod, "foam_env", lambda: {})

    src = json.loads((CONFIGS / "geom_impulse_bucket.json").read_text(encoding="utf-8"))
    src["output_dir"] = str(tmp_path / "out")
    src["name"] = "missing_log_job"
    jp = tmp_path / "job.json"
    jp.write_text(json.dumps(src), encoding="utf-8")
    r = run_job(jp, skip_solve=True)
    assert r.get("success") is False
    blob = " ".join(str(e) for e in (r.get("errors") or []))
    assert "never started" in blob or "wiped mid-run" in blob
    assert "log.checkMesh" in blob
    note = str((r.get("checkMesh") or {}).get("note") or "")
    assert "never started" in note or "wiped mid-run" in blob


def test_write_case_will_not_unlink_logs_while_job_holds_tree(tmp_path: Path):
    case = tmp_path / "held"
    case.mkdir()
    log = case / "log.checkMesh"
    log.write_text("KEEP ME", encoding="utf-8")
    job = load_job(CONFIGS / "geom_impulse_bucket.json")
    spec, ml, times = _spec_times(job)

    ready = threading.Event()
    done = threading.Event()

    def holder():
        acquire_job_tree(case)
        ready.set()
        done.wait(timeout=30)
        release_job_tree(case)

    t = threading.Thread(target=holder, daemon=True)
    t.start()
    assert ready.wait(timeout=5)
    with pytest.raises(RuntimeError, match="holds this tree"):
        write_case(case, job, ml, times, spec)
    assert log.is_file()
    assert log.read_text(encoding="utf-8") == "KEEP ME"
    done.set()
    t.join(timeout=5)


def test_outline_will_not_unlink_logs_while_job_holds_tree(tmp_path: Path):
    dest = tmp_path / "outline_held"
    dest.mkdir()
    log = dest / "log.checkMesh"
    log.write_text("KEEP OUTLINE", encoding="utf-8")
    ready = threading.Event()
    done = threading.Event()

    def holder():
        acquire_job_tree(dest)
        ready.set()
        done.wait(timeout=30)
        release_job_tree(dest)

    t = threading.Thread(target=holder, daemon=True)
    t.start()
    assert ready.wait(timeout=5)
    with pytest.raises(RuntimeError, match="holds this tree"):
        outline_from_knobs({"family": "cup", "hu_c": 0.50}, dest=dest)
    assert log.is_file()
    assert log.read_text(encoding="utf-8") == "KEEP OUTLINE"
    done.set()
    t.join(timeout=5)


def test_status_reports_phase_elapsed_for_inflight_job():
    from impulsecalc3 import serve

    serve.reset_state()
    with serve._lock:
        serve._state["phase"] = "solving"
        serve._state["t0"] = time.time() - 12.0
        serve._state["error"] = None
        serve._state["authority"] = AUTHORITY_SCOPING
        serve._state["success"] = False
    snap = serve.status_payload()
    assert snap["phase"] == "solving"
    assert 11.0 <= float(snap["elapsed_s"]) <= 20.0
    hb = str(snap["heartbeat"]).lower()
    assert "still running" in hb
    assert "not stuck" in hb
    for k in (
        "phase",
        "elapsed_s",
        "heartbeat",
        "of_present",
        "predicted",
        "authority",
        "success",
        "Ft_N",
        "Fd_N",
        "error",
    ):
        assert k in snap
    assert snap["predicted"] is True
    serve.reset_state()


def test_app_html_has_job_panel_ids():
    html = (ROOT / "viewer" / "app.html").read_text(encoding="utf-8")
    for i in (
        "long_job_panel",
        "long_job_phase",
        "long_job_elapsed",
        "long_job_heartbeat",
        "long_job_detail",
        "long_job_bar",
        "long_job_keep",
    ):
        assert f'id="{i}"' in html
    assert "Keep this tab open" in html
    assert "not stuck" in html.lower()
    assert 'id="btn-undo"' in html
    assert "type=range" in html.replace(" ", "") or 'type="range"' in html
    assert "Ctrl+Z" in html or "ctrlKey" in html
    assert "Inlet absolute velocity (C1)" in html
    assert "Outlet absolute velocity (C2)" in html
    assert "Inlet relative velocity (W1)" in html
    assert "Outlet relative velocity (W2)" in html
    assert "Inlet abs. angle" in html
    assert "Outlet abs. angle" in html
    assert "Relative inlet" in html
    assert "Relative outlet" in html
    assert "Turbine/blade speed (U)" in html
    assert 'id="triangle_table"' in html
    assert "class=\"knob" in html or "class='knob'" in html or "knob sys-" in html


@pytest.fixture
def serve_http(monkeypatch: pytest.MonkeyPatch):
    from impulsecalc3 import serve

    def fake_mesh(knobs):
        time.sleep(0.35)
        return {
            "ok": True,
            "authority": AUTHORITY_SCOPING,
            "predicted": True,
            "success": False,
            "of_present": False,
            "report": None,
        }

    monkeypatch.setattr(serve, "_run_mesh", fake_mesh)
    serve.reset_state()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), serve.Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    port = httpd.server_address[1]
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()
    serve.reset_state()


def test_mesh_post_returns_202_without_blocking_on_of(serve_http: str):
    t0 = time.time()
    code, body = _json(serve_http + "/mesh", {})
    elapsed = time.time() - t0
    assert code == 202
    assert elapsed < 1.5
    assert body.get("started") is True
    assert body.get("phase") == "meshing"
    assert body.get("authority") == AUTHORITY_SCOPING
    st_code, st = _json(serve_http + "/status")
    assert st_code == 200
    assert st["phase"] == "meshing"
    assert "elapsed_s" in st
    assert "still running" in str(st.get("heartbeat") or "").lower()


def test_outline_noop_while_meshing(serve_http: str):
    from impulsecalc3 import serve

    with serve._lock:
        serve._state["phase"] = "meshing"
    code, body = _json(serve_http + "/outline", {"family": "cup"})
    assert code == 200
    assert body.get("noop") is True
    assert body.get("authority") == AUTHORITY_SCOPING


def test_mesh_worker_surfaces_missing_log_on_status(monkeypatch: pytest.MonkeyPatch):
    from impulsecalc3 import serve

    def boom(knobs):
        raise FileNotFoundError(
            "[Errno 2] No such file or directory: "
            "'/workspace/ImpulseCalc3/output/geom_tests/knobs_preview/openfoam_cases/knobs_preview/log.checkMesh'"
        )

    monkeypatch.setattr(serve, "_run_mesh", boom)
    serve.reset_state()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), serve.Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}"
        code, body = _json(url + "/mesh", {})
        assert code == 202
        assert body.get("started") is True
        deadline = time.time() + 4.0
        snap = {}
        while time.time() < deadline:
            _, snap = _json(url + "/status")
            if snap.get("phase") in ("error", "done-fail"):
                break
            time.sleep(0.05)
        assert snap.get("phase") in ("error", "done-fail")
        err = str(snap.get("error") or "")
        assert "never started" in err or "wiped mid-run" in err or "log.checkMesh" in err
        assert "heartbeat" in snap
    finally:
        httpd.shutdown()
        serve.reset_state()



def test_apply_report_mesh_only_clears_leftover_forces():
    from impulsecalc3 import serve

    serve.reset_state()
    leftover = {
        "success": True,
        "flags": {"solve_ok": True, "newtons_trusted": False},
        "solve": {"rc": 0, "fatal": False, "note": "skip_solve: reused existing time directories"},
        "errors": [],
        "checkMesh": {},
        "forces": {"blades": [{"Ft_N": 110.8, "Fd_N": 10.06}]},
    }
    serve._apply_report(leftover, cfd_this_click=False)
    assert serve._state["Ft_N"] is None
    assert serve._state["Fd_N"] is None
    assert serve._state["cfd_ran"] is False
    # skip_solve leftover must not populate even if the flag is forgotten
    serve._apply_report(leftover, cfd_this_click=True)
    assert serve._state["Ft_N"] is None
    assert serve._state["Fd_N"] is None
    serve.reset_state()


def test_apply_report_solve_keeps_this_click_forces():
    from impulsecalc3 import serve

    serve.reset_state()
    with serve._lock:
        serve._state["kind"] = "solve"
        serve._state["cfd_ran"] = True
    rep = {
        "success": True,
        "name": "knobs_preview",
        "flags": {"solve_ok": True, "newtons_trusted": False},
        "solve": {"rc": 0, "fatal": False, "note": "rhoCentralFoam End"},
        "errors": [],
        "checkMesh": {},
        "forces": {"blades": [{"Ft_N": 12.3, "Fd_N": 4.5}]},
    }
    serve._apply_report(rep)
    assert serve._state["Ft_N"] == 12.3
    assert serve._state["Fd_N"] == 4.5
    assert serve._state["cfd_ran"] is True
    snap = serve.status_payload()
    assert snap["Ft_N"] == 12.3
    assert snap["Fd_N"] == 4.5
    serve.reset_state()


def test_mesh_only_status_nulls_leftover_report_ft(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Mesh-only /status Ft_N is None even if leftover report JSON has Ft_N set."""
    from impulsecalc3 import serve

    leftover = {
        "name": "knobs_preview",
        "success": True,
        "flags": {"solve_ok": True, "newtons_trusted": False, "mesh_ok": True},
        "solve": {
            "rc": 0,
            "fatal": False,
            "note": "skip_solve: reused existing time directories",
        },
        "errors": [],
        "checkMesh": {"failed_checks": None},
        "forces": {"blades": [{"blade_index": 0, "Ft_N": 110.8, "Fd_N": 10.06}]},
    }
    report_path = tmp_path / "knobs_preview_report.json"
    report_path.write_text(json.dumps(leftover), encoding="utf-8")
    monkeypatch.setattr(serve, "LAST_REPORT", report_path)

    def fake_mesh(knobs):
        time.sleep(0.05)
        return {
            "ok": True,
            "authority": AUTHORITY_SCOPING,
            "predicted": True,
            "success": False,
            "cfd_ran": False,
            "of_present": True,
            "report": leftover,
        }

    monkeypatch.setattr(serve, "_run_mesh", fake_mesh)
    serve.reset_state()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), serve.Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}"
        code, body = _json(url + "/mesh", {})
        assert code == 202
        assert body.get("started") is True
        deadline = time.time() + 4.0
        snap = {}
        while time.time() < deadline:
            _, snap = _json(url + "/status")
            if snap.get("phase") in ("done-ok", "done-fail", "error"):
                break
            time.sleep(0.05)
        assert snap.get("phase") == "done-ok"
        assert snap.get("kind") == "mesh"
        assert snap.get("Ft_N") is None
        assert snap.get("Fd_N") is None
        on_disk = json.loads(report_path.read_text(encoding="utf-8"))
        assert on_disk["forces"]["blades"][0]["Ft_N"] == 110.8
        rcode, result = _json(url + "/result")
        assert rcode == 200
        assert result.get("Ft_N") is None
        assert result.get("Fd_N") is None
    finally:
        httpd.shutdown()
        serve.reset_state()


def test_app_html_field_plot_ids_and_plot_routes_404_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    html = (ROOT / "viewer" / "app.html").read_text(encoding="utf-8")
    assert 'id="force_history"' in html
    assert 'id="contour_p"' in html
    assert 'id="wall_cp_blade0"' in html
    assert "/plot/" in html
    assert "frameset" not in html.lower()

    from impulsecalc3 import serve

    empty = tmp_path / "no_plots"
    empty.mkdir()
    monkeypatch.setattr(serve, "PLOT_DIRS", (empty,))
    serve.reset_state()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), serve.Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}"
        for name in ("force_history.png", "contour_p.png", "wall_cp_blade0.png"):
            code, body = _json(url + "/plot/" + name)
            assert code == 404
            assert body.get("ok") is False
            assert "error" in body
        code, body = _json(url + "/plot/../secret.png")
        assert code == 404
    finally:
        httpd.shutdown()
        serve.reset_state()



def test_apply_report_solve_feeds_setloss_euler_and_of_power():
    """ /status after a fake report with Ft_N includes euler_work_j_kg and Ft·U. """
    from impulsecalc3 import serve

    serve.reset_state()
    with serve._lock:
        serve._state["kind"] = "solve"
        serve._state["cfd_ran"] = True
    rep = {
        "success": True,
        "name": "knobs_preview",
        "flags": {"solve_ok": True, "mesh_ok": True, "newtons_trusted": False},
        "solve": {"rc": 0, "fatal": False, "note": "rhoCentralFoam End"},
        "errors": [],
        "checkMesh": {},
        "forces": {"blades": [{"Ft_N": 12.3, "Fd_N": 4.5}]},
        "euler_work_j_kg": 111000.0,
        "power_w": 222.0,
        "u_m_s": 157.08,
        "Mw1": 1.404,
        "loss_scoping": {"profile": {"Yp": 0.08}, "shock": {"status": "not_implemented"}},
        "eta_from_cfd": 0.99,  # must be stripped
    }
    serve._apply_report(rep)
    assert serve._state["Ft_N"] == 12.3
    assert serve._state["euler_work_j_kg"] == 111000.0
    assert serve._state["power_w"] == 222.0
    assert abs(float(serve._state["of_power_w"]) - 12.3 * 157.08) < 1e-6
    assert serve._state["Mw1"] == 1.404
    assert serve._state["eta_from_cfd"] is None
    assert (serve._state["result"] or {}).get("eta_from_cfd") is None
    snap = serve.status_payload()
    assert snap["euler_work_j_kg"] == 111000.0
    assert snap["Ft_N"] == 12.3
    assert abs(float(snap["of_power_w"]) - 12.3 * 157.08) < 1e-6
    assert snap.get("eta_from_cfd") in (None,)
    res = snap.get("result") or {}
    assert res.get("euler_work_j_kg") == 111000.0
    assert res.get("eta_from_cfd") is None
    serve.reset_state()


def test_status_rehydrate_finished_case_fills_losses(tmp_path, monkeypatch):
    from impulsecalc3 import serve
    from impulsecalc3.preview import AUTHORITY_FIELD_CFD

    serve.reset_state()
    job = {
        "format": "impulsecalc3_job_v1",
        "name": "knobs_preview",
        "predicted": True,
        "geometry": {
            "chord_m": 0.01,
            "solidity": 1.13688,
            "n_blades_machine": 25,
            "mean_radius_m": 0.0375,
            "span_m": 0.005,
            "beta1_flow_deg": 72,
            "beta2_flow_deg": -72,
            "incidence_deg": 0,
            "deviation_deg": 0,
            "profile_family": "impulse_bucket",
            "thickness_c": 0.12,
            "le_radius_c": 0.02,
            "te_radius_c": 0.02,
        },
        "gas": {
            "p1_pa": 550000,
            "t1_k": 1100,
            "gamma": 1.3,
            "r_specific_j_kg_k": 320,
            "mu_pa_s": 4.5e-5,
            "w1_m_s": 950,
            "blade_speed_u_m_s": 157.08,
            "predicted": True,
        },
        "cfd": {"n_chords_min": 5, "end_time_s": 1e-4},
        "engine": {},
    }
    jp = tmp_path / "knobs_preview.json"
    jp.write_text(__import__("json").dumps(job), encoding="utf-8")
    rep = {
        "success": True,
        "name": "knobs_preview",
        "flags": {"solve_ok": True, "mesh_ok": True, "force_plateau": True, "eta_from_cfd": False},
        "solve": {"rc": 0, "fatal": False, "note": "rhoCentralFoam End"},
        "errors": [],
        "checkMesh": {"failed_checks": None},
        "forces": {"blades": [{"Ft_N": 24.58, "Fd_N": -3.5}]},
        "meanline": {"euler_work_j_kg": 50000.0, "Mw1": 1.4, "eta_from_cfd": None},
        "eta_from_cfd": None,
    }
    rp = tmp_path / "knobs_preview_report.json"
    rp.write_text(__import__("json").dumps(rep), encoding="utf-8")
    monkeypatch.setattr(serve, "LAST_JOB", jp)
    monkeypatch.setattr(serve, "LAST_REPORT", rp)
    monkeypatch.setattr(serve, "_foam_case_complete", lambda case_dir: True)
    monkeypatch.setattr(serve, "_of_progress", lambda case_dir: {"sim_time_s": 1e-4, "end_time_s": 1e-4, "n_time_dirs": 8})
    monkeypatch.setattr(
        serve,
        "_foam_log_tail",
        lambda case_dir: {"exists": True, "ended": True, "clock_s": 12.0, "fresh": False},
    )

    def fake_extract(case_dir, job=None):
        return {
            "Mw1_OF": 1.25,
            "w_inlet_m_s": 937.0,
            "w_outlet_m_s": 378.0,
            "w2_over_w1": 0.403,
            "probe": "test probe",
            "predicted": True,
            "eta_from_cfd": None,
        }

    monkeypatch.setattr("impulsecalc3.post.extract_of_stations", fake_extract)
    snap = serve.status_payload()
    assert snap["Ft_N"] == 24.58
    assert snap["euler_work_j_kg"] is not None
    assert snap["power_w"] is not None
    assert snap["of_power_w"] is not None
    assert abs(float(snap["of_power_w"]) - 24.58 * float(snap["u_m_s"])) < 1e-4
    assert snap["Mw1"] is not None
    assert snap["Mw1_OF"] == 1.25
    ids = [f["id"] for f in (snap.get("ntrs_checks") or [])]
    assert "mw1_of_inlet" in ids
    assert "mw1_of_goldman_range" in ids
    assert "of_w2_over_w1" in ids
    assert snap.get("eta_from_cfd") in (None,)
    assert (snap.get("result") or {}).get("eta_from_cfd") is None
    serve.reset_state()
