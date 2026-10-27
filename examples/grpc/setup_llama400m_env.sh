#!/usr/bin/env bash
# Prepare the Conda environment once, even when multiple ranks start together.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
# Initialize conda for noninteractive shells (including Slurm jobs).
conda_exe="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "$conda_exe" ]]; then
    echo "Conda not found; add conda to PATH or set CONDA_EXE to its executable" >&2
    exit 2
fi
conda_base="$("$conda_exe" info --base)"
source "$conda_base/etc/profile.d/conda.sh"
cd -- "$repo_root"
(
    flock -x 9
    if ! conda env list | awk '$1 == "appfl_venv_v2" {found=1} END {exit !found}'; then
        conda create -n appfl_venv_v2 python=3.10
    fi
    conda activate appfl_venv_v2
    marker="$CONDA_PREFIX/.llama400m-ready"
    if [[ ! -f "$marker" ]]; then
        pip install -e ".[mpi,dev,examples]"
        pip install transformers
        touch -- "$marker"
    fi
) 9>"$repo_root/.llama400m-setup.lock"
