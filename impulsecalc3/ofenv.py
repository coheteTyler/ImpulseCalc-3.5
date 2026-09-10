"""ESI OpenFOAM v2412 environment.

Native install at /usr/lib/openfoam/openfoam2412 if present.
This box's ESI apt host drops TLS; fallback is Docker image opencfd/openfoam-run:2412
(same binaries, same WM=v2412). Same solvers. Not a second CFD stack.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

BASHRC = Path("/usr/lib/openfoam/openfoam2412/etc/bashrc")
FOAM_DIR = Path("/usr/lib/openfoam/openfoam2412")
DOCKER_IMAGE = "opencfd/openfoam-run:2412"


def _docker_bin() -> str | None:
    return shutil.which("docker")


def _docker_sock_ok() -> bool:
    sock = Path("/var/run/docker.sock")
    return sock.exists()


def _docker_host_env() -> dict[str, str]:
    """Prefer unix sock; fall back to host daemon on loopback TCP 2375 (sand-box)."""
    env = os.environ.copy()
    if _docker_sock_ok():
        env.setdefault("DOCKER_HOST", "unix:///var/run/docker.sock")
        return env
    # Host Docker API exposed into this container (no nested dockerd).
    env["DOCKER_HOST"] = "tcp://127.0.0.1:2375"
    return env


_MERGED_CACHE: str | None | bool = False


def _self_merged_dir() -> str | None:
    """Host path of this container's overlay merged root (for -v binds via host Docker).

    Plain -v /workspace:/workspace from inside the sand-box mounts an empty host
    dir. Binding MergedDir+/workspace shares the live box filesystem.
    """
    global _MERGED_CACHE
    if _MERGED_CACHE is not False:
        return _MERGED_CACHE  # type: ignore[return-value]
    merged: str | None = None
    try:
        mi = Path("/proc/self/mountinfo").read_text(encoding="utf-8")
    except OSError:
        mi = ""
    # upperdir=.../overlay2/<id>/diff  →  .../overlay2/<id>/merged
    import re

    m = re.search(r"upperdir=([^,\s]+)/diff", mi)
    if m:
        cand = m.group(1) + "/merged"
        # Host docker sees this path; we may not be able to listdir it.
        merged = cand if cand.startswith("/var/lib/docker/overlay2/") else None
    if merged is None:
        docker = _docker_bin()
        if docker:
            ps = subprocess.run(
                [docker, "ps", "--format", "{{.ID}} {{.Names}}"],
                check=False,
                capture_output=True,
                text=True,
                env=_docker_host_env(),
            )
            if ps.returncode == 0:
                for line in ps.stdout.splitlines():
                    parts = line.strip().split(None, 1)
                    if not parts:
                        continue
                    cid = parts[0]
                    insp = subprocess.run(
                        [
                            docker,
                            "inspect",
                            cid,
                            "--format",
                            "{{.GraphDriver.Data.MergedDir}}",
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                        env=_docker_host_env(),
                    )
                    if insp.returncode == 0:
                        cand = insp.stdout.strip()
                        # Prefer the container whose merged contains our cwd marker.
                        if cand.startswith("/var/lib/docker/overlay2/"):
                            # Heuristic: hostname match or first overlay2 merged.
                            hostname = Path("/etc/hostname").read_text(encoding="utf-8").strip()
                            hn = subprocess.run(
                                [
                                    docker,
                                    "inspect",
                                    cid,
                                    "--format",
                                    "{{.Config.Hostname}}",
                                ],
                                check=False,
                                capture_output=True,
                                text=True,
                                env=_docker_host_env(),
                            )
                            if hn.returncode == 0 and hn.stdout.strip() == hostname:
                                merged = cand
                                break
                            if merged is None:
                                merged = cand
    _MERGED_CACHE = merged
    return merged


def _docker_volume_spec(cwd: Path) -> str:
    """Return docker -v src:dst so host daemon sees box files."""
    cwd = Path(cwd).resolve()
    merged = _self_merged_dir()
    if merged:
        return f"{merged}{cwd}:{cwd}"
    return f"{cwd}:{cwd}"


def _docker_image_present() -> bool:
    docker = _docker_bin()
    if not docker:
        return False
    if not _docker_sock_ok() and "DOCKER_HOST" not in os.environ:
        # Still allow tcp fallback probe.
        pass
    proc = subprocess.run(
        [docker, "image", "inspect", DOCKER_IMAGE],
        check=False,
        capture_output=True,
        env=_docker_host_env(),
    )
    return proc.returncode == 0


def _docker_prefix() -> list[str]:
    sock = Path("/var/run/docker.sock")
    if sock.exists() and os.access(str(sock), os.R_OK | os.W_OK):
        return []
    if shutil.which("sg") and _user_in_docker_group():
        return ["sg", "docker", "-c"]
    if shutil.which("sudo"):
        return ["sudo", "-n"]
    return []


def _user_in_docker_group() -> bool:
    try:
        import grp

        g = grp.getgrnam("docker")
        return os.getuid() == 0 or os.environ.get("USER", "") in g.gr_mem or "box" in g.gr_mem
    except KeyError:
        return False


def _flatten_docker_cmd(inner: list[str]) -> list[str]:
    """sg docker -c needs a single shell string; sudo -n takes argv."""
    pref = _docker_prefix()
    if pref[:1] == ["sg"]:
        return ["sg", "docker", "-c", " ".join(_shell_quote(x) for x in inner)]
    return pref + inner


def _shell_quote(s: str) -> str:
    if not s or any(c in s for c in ' \t\n"$\'\\'):
        return "'" + s.replace("'", "'\\''") + "'"
    return s


def _esi_native() -> bool:
    """True only for ESI v2412 on disk. Foundation OF12 stub is not this."""
    return BASHRC.is_file()


def openfoam_available() -> bool:
    # Do not treat /opt/openfoam12 rhoCentralFoam (shockFluid wrapper) as present.
    if _esi_native():
        return True
    return _docker_image_present()


def foam_env() -> dict[str, str]:
    """Return env with OpenFOAM sourced. Empty PATH extras if missing.

    Docker path does not mutate host PATH; run_foam wraps the image instead.
    """
    env = os.environ.copy()
    if not BASHRC.is_file():
        return env
    cmd = f"set +u; . {BASHRC} >/dev/null 2>&1; /usr/bin/env -0"
    proc = subprocess.run(
        ["bash", "-lc", cmd],
        check=False,
        capture_output=True,
    )
    if proc.returncode != 0:
        return env
    out = proc.stdout.split(b"\0")
    for item in out:
        if not item or b"=" not in item:
            continue
        k, _, v = item.partition(b"=")
        try:
            env[k.decode()] = v.decode()
        except UnicodeDecodeError:
            continue
    return env


def _native_solver_ok(args: list[str]) -> bool:
    if not args:
        return False
    return _esi_native()


def run_foam(args: list[str], cwd: Path, log_name: str, env: dict[str, str] | None = None) -> int:
    cwd = Path(cwd).resolve()
    log = cwd / log_name
    cwd.mkdir(parents=True, exist_ok=True)
    if _native_solver_ok(args):
        env = env or foam_env()
        # Always source ESI bashrc so Foundation OF12 never wins PATH.
        cmdline = "set +u; . " + str(BASHRC) + " >/dev/null 2>&1; exec " + " ".join(
            _shell_quote(a) for a in args
        )
        with log.open("w", encoding="utf-8") as fh:
            proc = subprocess.run(
                ["bash", "-lc", cmdline],
                cwd=str(cwd),
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                check=False,
            )
        return int(proc.returncode)
    if not _docker_image_present():
        with log.open("w", encoding="utf-8") as fh:
            fh.write("OpenFOAM v2412 missing (no native install, no opencfd/openfoam-run:2412).\n")
        return 127
    uid = os.getuid()
    gid = os.getgid()
    vol = _docker_volume_spec(cwd)
    inner = [
        "docker",
        "run",
        "--rm",
        "--user",
        f"{uid}:{gid}",
        "-v",
        vol,
        "-w",
        str(cwd),
        "--entrypoint",
        "bash",
        DOCKER_IMAGE,
        "-lc",
        " ".join(_shell_quote(a) for a in args),
    ]
    cmd = _flatten_docker_cmd(inner)
    with log.open("w", encoding="utf-8") as fh:
        fh.write(f"# docker volume: {vol}\n")
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=fh,
            stderr=subprocess.STDOUT,
            check=False,
            env=_docker_host_env(),
        )
    return int(proc.returncode)
