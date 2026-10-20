#!/usr/bin/env bash
# srun starts this script once per rank. Rank 0 serves; rank K runs ClientK.
set -euo pipefail
training_script="$1"
deployment="$2"
port="$3"
rendezvous="$4"
shift 4
: "${SLURM_PROCID:?Missing Slurm rank}"
if (( SLURM_PROCID == 0 )); then
    # Publish rank 0's actual host on the shared filesystem.
    hostname > "$rendezvous/server_host"
    exec "$PYTHON" "$training_script" server --deployment "$deployment" "$@" --bind-host 0.0.0.0
fi
# Wait for rank 0's service; hostname publication alone is not readiness.
uri="$("$PYTHON" -c '
import grpc, pathlib, sys, time
host_file = pathlib.Path(sys.argv[1]) / "server_host"
deadline = time.monotonic() + float(sys.argv[3])
while True:
    host = host_file.read_text().strip() if host_file.exists() else ""
    if host:
        break
    if time.monotonic() >= deadline:
        raise SystemExit("Timed out waiting for rank 0 hostname")
    time.sleep(0.2)
uri = f"{host}:{sys.argv[2]}"
with grpc.insecure_channel(uri) as channel:
    try:
        grpc.channel_ready_future(channel).result(timeout=max(0, deadline - time.monotonic()))
    except grpc.FutureTimeoutError:
        raise SystemExit("Server readiness timed out; inspect rank_0.err/rank_0.out")
print(uri)
' "$rendezvous" "$port" "${SERVER_READY_TIMEOUT:-600}")"
exec "$PYTHON" "$training_script" client --client-index "$SLURM_PROCID" --deployment "$deployment" "$@" --server-uri "$uri"
