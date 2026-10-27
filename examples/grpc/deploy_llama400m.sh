#!/usr/bin/env bash
# Start one role; --deployment replaces the default deployment JSON.
# --cuda-device cuda:N overrides JSON cuda_device for client training only.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
if [[ $# -lt 1 || ( "$1" != server && "$1" != client ) ]]; then
    echo "Usage: $0 server|client [--client-index N] [--deployment FILE] [--cuda-device cuda:N|cuda|cpu] [training flags]" >&2
    exit 2
fi
role="$1"
shift
deployment="$repo_root/examples/resources/configs/llama400m/deployment.json"
args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --deployment)
            [[ $# -ge 2 && -n "$2" ]] || { echo "--deployment requires a file" >&2; exit 2; }
            deployment="$2"
            [[ "$deployment" = /* ]] || deployment="$PWD/$deployment"
            shift 2
            ;;
        --deployment=*)
            deployment="${1#*=}"
            [[ -n "$deployment" ]] || { echo "--deployment requires a file" >&2; exit 2; }
            [[ "$deployment" = /* ]] || deployment="$PWD/$deployment"
            shift
            ;;
        *) args+=("$1"); shift ;;
    esac
done
# Prefer the environment created by setup_llama400m_env.sh when present.
# Otherwise let Conda resolve the name in its configured environment directories
# (including its default envs directory). Explicit APPFL_VENV overrides win.
if [[ -z "${APPFL_VENV:-}" ]]; then
    if [[ -d "$repo_root/appfl_venv_v2/conda-meta" ]]; then
        APPFL_VENV="$repo_root/appfl_venv_v2"
    else
        APPFL_VENV="appfl_venv_v2"
    fi
fi
export APPFL_VENV
# Initialize conda for noninteractive shells (including Slurm jobs).
conda_exe="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "$conda_exe" ]]; then
    echo "Conda not found; add conda to PATH or set CONDA_EXE to its executable" >&2
    exit 2
fi
conda_base="$("$conda_exe" info --base)"
source "$conda_base/etc/profile.d/conda.sh"
# Resolve prefix paths before changing to the repository directory.
if [[ "$APPFL_VENV" == */* && "$APPFL_VENV" != /* ]]; then
    APPFL_VENV="$PWD/$APPFL_VENV"
fi
conda activate "$APPFL_VENV"
cd -- "$repo_root"
exec "$CONDA_PREFIX/bin/python" "$script_dir/train_llama400m.py" "$role" \
    --deployment "$deployment" "${args[@]}"
