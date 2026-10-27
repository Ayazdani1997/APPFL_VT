#!/usr/bin/env bash
#SBATCH --job-name=appfl-llama400m
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=llama400m-%j.out
#SBATCH --error=llama400m-%j.err

# Submit from the APPFL root. Rank 0 is the server; rank K is ClientK.
# Set JSON "cuda_device": "cuda:0" or pass --cuda-device cuda:0 after
# the optional deployment path. CLI takes precedence for all clients.
# With one GPU per task, cuda:0 selects the GPU exposed to that task.
set -euo pipefail
# Load cluster modules here before environment setup or srun. Uncomment and
# replace these placeholders with names/versions listed by `module avail`.
# module load cuda/YOUR_CUDA_VERSION
# module load openmpi/YOUR_MPI_VERSION
# module load python/YOUR_PYTHON_VERSION
# Choose MPI compatible with mpi4py and CUDA compatible with installed PyTorch.
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
# Resolve the named environment through Conda on each allocated node.
export APPFL_VENV="${APPFL_VENV:-appfl_venv_v2}"
if [[ "$APPFL_VENV" == */* && "$APPFL_VENV" != /* ]]; then
    APPFL_VENV="$repo_root/$APPFL_VENV"
fi
export PYTHONUNBUFFERED=1
nodes="${SLURM_JOB_NUM_NODES:?Missing allocated node count}"
if (( ${SLURM_NTASKS:-0} != nodes )); then
    echo "Request exactly one task per node (--ntasks=$nodes --ntasks-per-node=1)" >&2
    exit 2
fi
# srun reads this hostfile on the batch host. Each listed node receives one rank
# in this exact order, making the first allocated node the dedicated server.
hostfile="$(mktemp)"
trap 'rm -f -- "$hostfile"' EXIT
scontrol show hostnames "${SLURM_JOB_NODELIST:?Missing allocated node list}" > "$hostfile"
mapfile -t allocated_hosts < "$hostfile"
if (( ${#allocated_hosts[@]} != nodes )); then
    echo "Allocated node list does not match node count" >&2
    exit 2
fi
server_host="${allocated_hosts[0]}"
export SLURM_HOSTFILE="$hostfile"
logs="$repo_root/output/llama400m/slurm-$SLURM_JOB_ID"
echo "Launching $nodes ranks; dedicated server: $server_host; per-node logs: $logs"
# The hostfile assigns exactly one rank per node. The wrapper creates its logs
# locally before redirecting output, so node-local storage is supported.
srun --nodes="$nodes" --ntasks="$nodes" --distribution=arbitrary \
    --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
    --gpus-per-task=1 --gpu-bind=single:1 --kill-on-bad-exit=1 \
    bash "$repo_root/examples/grpc/run_llama400m_rank.sh" \
    "$launcher" "$deployment" "$server_host" "$logs" "$@"
