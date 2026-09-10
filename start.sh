#!/usr/bin/env bash
# ImpulseCalc3 — Marlin V2 rotor 2D relative cascade (Linux).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi
.venv/bin/python -m pip install -q -U pip
.venv/bin/python -m pip install -q -r requirements.txt
OF_BASHRC="/usr/lib/openfoam/openfoam2412/etc/bashrc"
if [[ -f "$OF_BASHRC" ]]; then
  set +euo pipefail
  # OpenFOAM bashrc must not see our CLI args (it sources $1 as prefs).
  _ic3_args=("$@")
  set --
  # shellcheck disable=SC1090
  source "$OF_BASHRC"
  set -- "${_ic3_args[@]}"
  unset _ic3_args
  set -eo pipefail
else
  echo "WARNING: $OF_BASHRC not found. Outline/UI still work (SCOPING). checkMesh/rhoCentralFoam will not run." >&2
fi
if [[ "${1:-}" == "--ui" ]]; then
  shift
  exec .venv/bin/python -m impulsecalc3.serve "$@"
fi
exec .venv/bin/python -m impulsecalc3.run "$@"
