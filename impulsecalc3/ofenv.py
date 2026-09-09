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


def _docker_image_present() -> bool:
    docker = _docker_bin()
    if not docker or not _docker_sock_ok():
        return False
    proc = subprocess.run(
        _docker_prefix() + [docker, "image", "inspect", DOCKER_IMAGE],
        check=False,
        capture_output=True,
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
    inner = [
        "docker",
        "run",
        "--rm",
        "--user",
        f"{uid}:{gid}",
        "-v",
        f"{cwd}:{cwd}",
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
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=fh,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(proc.returncode)
