#!/usr/bin/env bash
# Prepare the shared environment once, even when multiple ranks start together.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
venv="${APPFL_VENV:-$repo_root/appfl_venv_v2}"
[[ "$venv" = /* ]] || venv="$PWD/$venv"
mkdir -p -- "$venv"
(
    flock -x 9
    if [[ ! -x "$venv/bin/python" ]]; then
        "${PYTHON:-python3}" -m venv "$venv" >&2
    fi
    marker="$venv/.llama400m-ready"
    if [[ ! -f "$marker" ]]; then
        source "$venv/bin/activate"
        python -m pip install "appfl[examples,mpi]" >&2
        python -m pip install transformers accelerate >&2
        # Use this checkout's APPFL changes with the installed dependencies.
        python -m pip install --no-deps -e "$repo_root" >&2
        touch -- "$marker"
    fi
) 9>"$venv/.llama400m-setup.lock"
