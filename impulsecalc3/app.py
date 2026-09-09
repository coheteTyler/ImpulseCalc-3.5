"""Shareable local app — one page, real mesh writer, port 8765.

  ./start.sh
  http://127.0.0.1:8765/

Mesh-only works without OpenFOAM. Solve runs only if ESI v2412 is present.
Never remeshes the live Marlin 80 µs report.
"""

from __future__ import annotations

import argparse
import base64
import json
import threading
import time
import traceback
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import sys

from . import APP_NAME
from .job import ROOT, load_job
from .ofenv import openfoam_available
from .preview import knobs_to_job, write_preview, APP_OUTPUT, APP_NAME as PREVIEW_NAME

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PORT_DEFAULT = 8765
VIEWER = ROOT / "viewer"
_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()
_LAST_PNG: Path | None = None


def _job_set(job_id: str, **fields: Any) -> None:
    with _JOBS_LOCK:
        cur = _JOBS.get(job_id) or {"id": job_id}
        cur.update(fields)
        cur["updated_at"] = time.time()
        _JOBS[job_id] = cur


def _json(handler: BaseHTTPRequestHandler, code: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, default=str).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _bytes(handler: BaseHTTPRequestHandler, code: int, data: bytes, ctype: str) -> None:
    handler.send_response(code)
    handler.send_header("Content-Type", ctype)
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    n = int(handler.headers.get("Content-Length") or 0)
    raw = handler.rfile.read(n) if n else b"{}"
    if not raw:
        return {}
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("JSON body must be an object")
    return data


def _png_b64(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _do_preview(knobs: dict[str, Any]) -> dict[str, Any]:
    global _LAST_PNG
    job = knobs_to_job(knobs)
    info = write_preview(job)
    png = Path(info["png"]) if info.get("png") else None
    _LAST_PNG = png
    info["png_b64"] = _png_b64(png)
    info["openfoam"] = openfoam_available()
    info["predicted"] = True
    info["eta_from_cfd"] = None
    return info


def _do_checkmesh(case_dir: Path) -> dict[str, Any]:
    from .ofenv import foam_env, run_foam

    if not openfoam_available():
        return {"ran": False, "note": "OpenFOAM ESI v2412 not available — mesh written without checkMesh"}
    env = foam_env()
    rc = run_foam(["checkMesh"], cwd=case_dir, log_name="log.checkMesh", env=env)
    log = (case_dir / "log.checkMesh").read_text(encoding="utf-8", errors="replace")
    from viewer.ofio import parse_checkmesh

    chk = parse_checkmesh(log)
    chk["ran"] = True
    chk["rc"] = int(rc)
    nfail = int(chk.get("failed_checks") or 0)
    chk["mesh_ok"] = bool(chk.get("mesh_ok_strict")) and nfail == 0
    return chk


def _do_solve(knobs: dict[str, Any]) -> dict[str, Any]:
    from .run import run_job

    if not openfoam_available():
        raise RuntimeError(
            "OpenFOAM ESI v2412 not available. Mesh-only still works. "
            "On Windows use WSL: wsl -- bash ./start.sh configs/geom_impulse_bucket.json --skip-solve"
        )
    job = knobs_to_job(knobs)
    job["geometry_test"] = True
    job["name"] = PREVIEW_NAME
    job["output_dir"] = APP_OUTPUT
    job.setdefault("cfd", {})
    job["cfd"].pop("smoke_end_s", None)
    t_min, t_max = 7.9827173e-05, 2.6609057819508715e-04
    raw = knobs.get("t_end_s")
    job["cfd"]["end_time_s"] = min(t_max, max(t_min, float(raw))) if raw not in (None, "") else t_min
    out = ROOT / APP_OUTPUT
    out.mkdir(parents=True, exist_ok=True)
    jp = out / f"{PREVIEW_NAME}_run.json"
    jp.write_text(json.dumps(job, indent=2, default=str) + "\n", encoding="utf-8")
    report = run_job(jp, skip_solve=False, post_only=False)
    blades = []
    for b in (report.get("forces") or {}).get("blades") or []:
        blades.append(
            {
                "blade_index": b.get("blade_index"),
                "Ft_N": b.get("Ft_N"),
                "Fd_N": b.get("Fd_N"),
                "predicted": True,
                "climbing": (report.get("flags") or {}).get("force_climbing"),
                "trusted": False,
            }
        )
    png = Path(report.get("case_dir") or "") / "mesh_preview.png"
    global _LAST_PNG
    if png.is_file():
        _LAST_PNG = png
    return {
        "ok": bool(report.get("success")),
        "predicted": True,
        "success": bool(report.get("success")),
        "climbing": (report.get("flags") or {}).get("force_climbing"),
        "mesh_ok": (report.get("flags") or {}).get("mesh_ok"),
        "solve_ok": (report.get("flags") or {}).get("solve_ok"),
        "newtons_trusted": False,
        "eta_from_cfd": None,
        "errors": report.get("errors") or [],
        "blades": blades,
        "flags": report.get("flags"),
        "t_end_s": report.get("t_end_s"),
        "n_chords": (report.get("times") or {}).get("n_chords"),
        "png_b64": _png_b64(_LAST_PNG),
        "report_path": report.get("report_path"),
        "note": "PREDICTED. Do not treat Newtons as a design load. Not the live Marlin 31 N shot.",
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "ImpulseCalc3/0.2"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"  {self.address_string()} {fmt % args}")

    def do_GET(self) -> None:  # noqa: N802
        u = urlparse(self.path)
        path = u.path
        try:
            if path in ("/", "/index.html", "/calc.html"):
                html = (VIEWER / "index.html").read_bytes()
                return _bytes(self, 200, html, "text/html; charset=utf-8")
            if path == "/marlin.html":
                html = (VIEWER / "marlin.html").read_bytes()
                return _bytes(self, 200, html, "text/html; charset=utf-8")
            if path == "/api/health":
                return _json(
                    self,
                    200,
                    {
                        "ok": True,
                        "app": APP_NAME,
                        "port": PORT_DEFAULT,
                        "openfoam": openfoam_available(),
                        "predicted": True,
                        "eta_from_cfd": False,
                        "live_marlin_untouched": True,
                    },
                )
            if path == "/api/defaults":
                cup = load_job(ROOT / "configs" / "geom_impulse_bucket.json")
                foil_path = ROOT / "configs" / "geom_foil.json"
                foil = load_job(foil_path) if foil_path.is_file() else None
                g = cup["geometry"]
                return _json(
                    self,
                    200,
                    {
                        "family": g.get("profile_family", "impulse_bucket"),
                        "upper_sagitta_c": g.get("upper_sagitta_c", 0.50),
                        "lower_sagitta_c": g.get("lower_sagitta_c", 0.22),
                        "le_fillet_r_c": g.get("le_fillet_r_c", 0.04),
                        "te_fillet_r_c": g.get("te_fillet_r_c", 0.04),
                        "thickness_c": 0.12,
                        "beta1_flow_deg": g["beta1_flow_deg"],
                        "beta2_flow_deg": g["beta2_flow_deg"],
                        "chord_m": g["chord_m"],
                        "solidity": g["solidity"],
                        "n_blades_machine": g["n_blades_machine"],
                        "mean_radius_m": g["mean_radius_m"],
                        "rotor_tip_radius_m": g["rotor_tip_radius_m"],
                        "hub_radius_m": g["hub_radius_m"],
                        "span_m": g["span_m"],
                        "openfoam": openfoam_available(),
                        "foil_family": (foil or {}).get("geometry", {}).get("profile_family"),
                        "predicted": True,
                    },
                )
            if path == "/api/preview.png":
                png = _LAST_PNG
                if png is None or not png.is_file():
                    return _json(self, 404, {"ok": False, "error": "no preview yet — hit Preview mesh"})
                return _bytes(self, 200, png.read_bytes(), "image/png")
            if path.startswith("/api/job/"):
                jid = path.split("/api/job/", 1)[1].strip("/")
                with _JOBS_LOCK:
                    rec = _JOBS.get(jid)
                if not rec:
                    return _json(self, 404, {"ok": False, "error": f"unknown job {jid}"})
                return _json(self, 200, rec)
            if path.startswith("/viewer/"):
                rel = path[len("/viewer/") :]
                fp = (VIEWER / rel).resolve()
                if not str(fp).startswith(str(VIEWER.resolve())) or not fp.is_file():
                    return _json(self, 404, {"ok": False, "error": "not found"})
                ctype = "image/png" if fp.suffix == ".png" else "application/octet-stream"
                if fp.suffix == ".html":
                    ctype = "text/html; charset=utf-8"
                return _bytes(self, 200, fp.read_bytes(), ctype)
            return _json(self, 404, {"ok": False, "error": f"no route {path}"})
        except Exception as exc:  # noqa: BLE001
            return _json(
                self,
                500,
                {"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-1500:]},
            )

    def do_POST(self) -> None:  # noqa: N802
        u = urlparse(self.path)
        path = u.path
        try:
            knobs = _read_json(self)
        except Exception as exc:  # noqa: BLE001
            return _json(self, 400, {"ok": False, "error": f"bad JSON: {exc}"})
        try:
            if path == "/api/preview":
                info = _do_preview(knobs)
                return _json(self, 200, info)
            if path == "/api/mesh":
                info = _do_preview(knobs)
                chk = _do_checkmesh(Path(info["case_dir"]))
                info["checkMesh"] = chk
                info["mesh_ok"] = chk.get("mesh_ok")
                if chk.get("ran") and not chk.get("mesh_ok"):
                    info["errors"] = [
                        f"checkMesh failed {chk.get('failed_checks')} checks (mesh_ok tracks failed_checks)"
                    ]
                return _json(self, 200, info)
            if path == "/api/solve":
                jid = uuid.uuid4().hex[:12]
                _job_set(jid, kind="solve", status="running", success=None, message="solve started", predicted=True)

                def _worker() -> None:
                    try:
                        result = _do_solve(knobs)
                        _job_set(
                            jid,
                            status="done",
                            success=bool(result.get("success")),
                            result=result,
                            message="solve done" if result.get("success") else "solve finished with flags",
                        )
                    except Exception as exc:  # noqa: BLE001
                        _job_set(
                            jid,
                            status="done",
                            success=False,
                            error=str(exc),
                            message=str(exc),
                            traceback=traceback.format_exc()[-1500:],
                            result={"ok": False, "predicted": True, "error": str(exc)},
                        )

                threading.Thread(target=_worker, name=f"ic3-solve-{jid}", daemon=True).start()
                return _json(
                    self,
                    200,
                    {
                        "async": True,
                        "job_id": jid,
                        "status": "running",
                        "predicted": True,
                        "message": "solve running — poll /api/job/" + jid,
                    },
                )
            return _json(self, 404, {"ok": False, "error": f"no POST route {path}"})
        except Exception as exc:  # noqa: BLE001
            return _json(
                self,
                400,
                {
                    "ok": False,
                    "predicted": True,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()[-2000:],
                },
            )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ImpulseCalc3 local app (port 8765)")
    ap.add_argument("--port", type=int, default=PORT_DEFAULT)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--app", action="store_true", help="ignored; this module is the app")
    args = ap.parse_args(argv)
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"ImpulseCalc3 — {url}")
    print("  Mesh-only works without OpenFOAM. Solve needs ESI v2412 (Linux / WSL).")
    print("  PREDICTED. Do not treat 31 N as a design load. Live Marlin report is not overwritten.")
    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
