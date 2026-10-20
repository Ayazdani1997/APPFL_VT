#!/usr/bin/env bash
#SBATCH --job-name=appfl-llama400m
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=llama400m-%j.out
#SBATCH --error=llama400m-%j.err

# Submit from the APPFL root. Rank 0 is the server; rank K is ClientK.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch or run inside a Slurm allocation}"
repo_root="${APPFL_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd -- "$repo_root"
repo_root="$PWD"
launcher="$repo_root/examples/grpc/deploy_llama400m.sh"
[[ -f "$launcher" ]] || { echo "Set APPFL_ROOT to the APPFL checkout" >&2; exit 2; }
deployment="$repo_root/examples/resources/configs/llama400m/deployment.json"
# Optional first argument replaces the default; remaining arguments are training flags.
if [[ $# -gt 0 && "$1" != --* ]]; then
    deployment="$1"
    [[ "$deployment" = /* ]] || deployment="${SLURM_SUBMIT_DIR:-$repo_root}/$deployment"
    shift
fi
export PYTHON="${PYTHON:-python}"
export PYTHONUNBUFFERED=1
# Validate the server configuration and derive its client count and port.
metadata="$(bash "$launcher" server --deployment "$deployment" "$@" --print-config | "$PYTHON" -c '
import sys
from omegaconf import OmegaConf
c = OmegaConf.load(sys.stdin).server_configs
g = c.comm_configs.grpc_configs
if g.get("use_ssl", False):
    raise SystemExit("This Slurm example expects use_ssl: false")
print(c.num_clients, g.server_uri.rsplit(":", 1)[1])
')"
read -r num_clients port <<< "$metadata"
tasks=$((num_clients + 1))
if (( ${SLURM_NTASKS:-0} < tasks )); then
    echo "Need at least $tasks tasks for $num_clients clients; allocated ${SLURM_NTASKS:-0}" >&2
    exit 2
fi
# Build every client configuration before launching any ranks.
for ((index=1; index<=num_clients; index++)); do
    echo "Validating Client$index"
    bash "$launcher" client --client-index "$index" --deployment "$deployment" "$@" --print-config | "$PYTHON" -c '
import sys
from omegaconf import OmegaConf
config = OmegaConf.load(sys.stdin)
if config.comm_configs.grpc_configs.get("use_ssl", False):
    raise SystemExit("This Slurm example expects use_ssl: false on clients too")
'
done
logs="$repo_root/output/llama400m/slurm-$SLURM_JOB_ID"
mkdir -p -- "$logs"
# A unique directory prevents stale host discovery when restarting a job.
rendezvous="$(mktemp -d "$logs/rendezvous.XXXXXX")"
echo "Launching $tasks ranks: rank 0 server, ranks 1-$num_clients clients; logs: $logs"
# One step gives each role a global SLURM_PROCID. Slurm handles node placement
# and GPU isolation, and terminates peer ranks if any rank fails.
exec srun --ntasks="$tasks" --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
    --gpus-per-task=1 --gpu-bind=single:1 --kill-on-bad-exit=1 \
    --output="$logs/rank_%t.out" --error="$logs/rank_%t.err" \
    bash "$repo_root/examples/grpc/run_llama400m_rank.sh" \
    "$repo_root/examples/grpc/train_llama400m.py" "$deployment" "$port" "$rendezvous" "$@"
