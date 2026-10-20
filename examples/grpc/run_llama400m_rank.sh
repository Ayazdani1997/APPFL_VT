#!/usr/bin/env bash
# Start once per rank through srun or a non-Slurm launcher. Rank 0 serves; rank K runs ClientK.
set -euo pipefail
if (( $# < 4 )); then
    echo "Usage: $0 LAUNCHER DEPLOYMENT SERVER_HOST LOG_DIR [training flags]" >&2
    echo "Set APPFL_RANK and APPFL_WORLD_SIZE outside Slurm." >&2
    exit 2
fi
launcher="$1"
deployment="$2"
server_host="$3"
logs="$4"
shift 4
rank="${APPFL_RANK:-${SLURM_PROCID:-}}"
world_size="${APPFL_WORLD_SIZE:-${SLURM_NTASKS:-}}"
if [[ ! "$rank" =~ ^(0|[1-9][0-9]*)$ || ! "$world_size" =~ ^[1-9][0-9]*$ ]]; then
    echo "Set nonnegative APPFL_RANK and positive APPFL_WORLD_SIZE (or Slurm equivalents)" >&2
    exit 2
fi
if (( rank >= world_size )); then
    echo "Rank must be less than world size" >&2
    exit 2
fi
mkdir -p -- "$logs"
exec >"$logs/rank_${rank}.out" 2>"$logs/rank_${rank}.err"
echo "Rank $rank on $(hostname); server: $server_host"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Require an existing environment on this node before installing dependencies.
: "${APPFL_VENV:?Missing environment path}"
if [[ ! -x "$APPFL_VENV/bin/python" || ! -f "$APPFL_VENV/bin/activate" ]]; then
    echo "Missing Python environment on $(hostname): $APPFL_VENV" >&2
    echo "Create it on this node with: APPFL_VENV=\"$APPFL_VENV\" bash \"$script_dir/setup_llama400m_env.sh\"" >&2
    exit 2
fi
bash "$script_dir/setup_llama400m_env.sh"
# Activate this node's environment before clients use Python for readiness.
source "${APPFL_VENV:?Missing environment path}/bin/activate"
export PYTHON="$APPFL_VENV/bin/python"
# Validate the effective deployment after installation on this node.
metadata="$(bash "$launcher" server --deployment "$deployment" "$@" --hostname "$server_host" --print-config | "$PYTHON" -c '
import sys
from omegaconf import OmegaConf
c = OmegaConf.load(sys.stdin).server_configs
g = c.comm_configs.grpc_configs
if g.get("use_ssl", False):
    raise SystemExit("This rank launcher expects use_ssl: false")
print(c.num_clients, g.server_uri.rsplit(":", 1)[1])
')"
read -r num_clients port <<< "$metadata"
if (( world_size != num_clients + 1 )); then
    echo "Need exactly $((num_clients + 1)) ranks for $num_clients clients" >&2
    exit 2
fi
if (( rank > 0 )); then
    bash "$launcher" client --client-index "$rank" --deployment "$deployment" "$@" --hostname "$server_host" --print-config | "$PYTHON" -c '
import sys
from omegaconf import OmegaConf
c = OmegaConf.load(sys.stdin)
if c.comm_configs.grpc_configs.get("use_ssl", False):
    raise SystemExit("This rank launcher expects use_ssl: false on clients too")
'
fi
if (( rank == 0 )); then
    exec bash "$launcher" server --deployment "$deployment" "$@" --hostname "$server_host" --port "$port" --bind-host 0.0.0.0
fi
# Connect directly to the dedicated server node; no shared discovery file.
uri="$("$PYTHON" -c '
import grpc, sys
host = sys.argv[1]
uri = f"[{host}]:{sys.argv[2]}" if ":" in host else f"{host}:{sys.argv[2]}"
with grpc.insecure_channel(uri) as channel:
    try:
        grpc.channel_ready_future(channel).result(timeout=float(sys.argv[3]))
    except grpc.FutureTimeoutError:
        raise SystemExit("Server readiness timed out; inspect rank_0.err/rank_0.out")
print(uri)
' "$server_host" "$port" "${SERVER_READY_TIMEOUT:-600}")"
exec bash "$launcher" client --client-index "$rank" --deployment "$deployment" "$@" --hostname "$server_host" --port "$port" --server-uri "$uri"
