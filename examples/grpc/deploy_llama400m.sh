#!/usr/bin/env bash
# Start one role; --deployment replaces the default deployment JSON.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
if [[ $# -lt 1 || ( "$1" != server && "$1" != client ) ]]; then
    echo "Usage: $0 server|client [--client-index N] [--deployment FILE] [training flags]" >&2
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
cd -- "$repo_root"
exec "${PYTHON:-python}" "$script_dir/train_llama400m.py" "$role" \
    --deployment "$deployment" "${args[@]}"
