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
export APPFL_VENV="${APPFL_VENV:-$repo_root/appfl_venv_v2}"
[[ "$APPFL_VENV" = /* ]] || APPFL_VENV="$PWD/$APPFL_VENV"
bash "$script_dir/setup_llama400m_env.sh"
source "$APPFL_VENV/bin/activate"
cd -- "$repo_root"
exec "$APPFL_VENV/bin/python" "$script_dir/train_llama400m.py" "$role" \
    --deployment "$deployment" "${args[@]}"
