#!/usr/bin/env bash
# ImpulseCalc3 UI from Ubuntu WSL. Solver is ESI OpenFOAM v2412 rhoCentralFoam.
# Do not source Foundation OF12: that remaps rhoCentralFoam to shockFluid.
set +u
if [ -f /usr/lib/openfoam/openfoam2412/etc/bashrc ]; then
  . /usr/lib/openfoam/openfoam2412/etc/bashrc
else
  echo "ESI OpenFOAM v2412 missing: /usr/lib/openfoam/openfoam2412/etc/bashrc"
  echo "Install: curl -fsSL https://dl.openfoam.com/add-debian-repo.sh | sudo bash"
  echo "         sudo apt-get install -y openfoam2412-default"
fi
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
VENV="$HOME/.venv-impulsecalc3"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
fi
"$VENV/bin/pip" install -q -U pip
"$VENV/bin/pip" install -q -r requirements.txt
echo "WM_PROJECT_VERSION=${WM_PROJECT_VERSION:-unset}"
echo "rhoCentralFoam $(command -v rhoCentralFoam || echo MISSING)"
echo "checkMesh $(command -v checkMesh || echo MISSING)"
fuser -k 8766/tcp 2>/dev/null || true
exec "$VENV/bin/python" -m impulsecalc3.serve --host 0.0.0.0 --port 8766
