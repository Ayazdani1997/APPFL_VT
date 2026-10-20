# Llama-400M deployment configuration

For environment setup, execution, testing, and Slurm/srun instructions, see the
[Llama-400M execution guide](../../../grpc/README_llama400m.md).

Configure the deployment in `deployment.json`. One base client YAML serves all
clients; `base_server.yaml` holds shared training/model and server defaults.
The launcher creates the effective configuration inline, without temporary YAMLs.
The old numbered YAMLs remain available for direct `--config` launches.

Start one process in each terminal, with your APPFL Python environment activated:

```bash
# Terminal 1: server
bash examples/grpc/deploy_llama400m.sh server
# Terminal 2: first client
bash examples/grpc/deploy_llama400m.sh client --client-index 1
# Terminal 3: second client
bash examples/grpc/deploy_llama400m.sh client --client-index 2
```

For `num_clients: N`, start clients 1 through N once each. The script starts only
the selected role in the foreground. Client N gets ID `ClientN`, dataset partition
`N-1`, and its own log paths. Server and dataset client counts both come from JSON.
The server must be running before clients connect. Ctrl-C stops that terminal's
process. No remote execution or background process management is performed.

Use the same deployment JSON in every terminal/machine:

```bash
bash examples/grpc/deploy_llama400m.sh client --client-index 3 --deployment /path/to/deployment.json
```

Deployment fields:

| Field | Meaning |
| --- | --- |
| `num_clients` | Number of clients and dataset partitions |
| `base_client`, `base_server` | Base YAML paths, relative to the JSON file |
| `hostname`, `port` | Server address clients connect to |
| `server_uri` | Optional explicit client target; overrides hostname/port |
| `bind_host` | Server listening interface; defaults to `0.0.0.0` |
| `enable_compression` | Enable or disable compression on every role |
| `error_bound` | Positive compressor error bound |
| `max_message_size`, `model_chunk_size` | Positive integer sizes in bytes |
| `use_model_chunking` | Enable or disable model chunking |
| `log_dir` | Parent directory for experiment logs |

For multiple hosts, set `hostname` to the server's reachable address. The server
listens on `bind_host:port`; `server_uri` affects clients only. IPv6 hostnames are
bracketed automatically. Set model/data paths in the base YAMLs for each machine.
The Bash launcher changes to the APPFL root so relative resource paths resolve.
Set `PYTHON=/path/to/venv/bin/python` to choose the Python executable explicitly.

CLI flags override JSON, which overrides base YAML settings:

```bash
bash examples/grpc/deploy_llama400m.sh client --client-index 1 \
  --hostname server.example --port 50123 --error-bound 1e-6 \
  --max-message-size 8388608 --model-chunk-size 2097152 --print-config
```

Remove `--print-config` to start training. Apply shared experiment/network changes
consistently to all launches. An explicit `server_uri` in JSON still takes
precedence over hostname/port flags; use `--server-uri` to replace it.

Training, communication, and compression log directories reflect the effective
compression mode/error bound, role/client ID, chunk size, and message size.
Repeated identical runs use the same directories. Disabling compression produces
`no_comp` labels. Source YAMLs remain unchanged.

Legacy direct launches still accept `--config`, `--overrides` (the old flat
`runtime_overrides.json`), and experiment flags. These options cannot be combined
with `--deployment`.

For compression-free base configurations, use `deployment_no_comp.json`, which
selects `base_client_no_comp.yaml` and `base_server_no_comp.yaml`. These bases
contain no `compressor_configs` block; the deployment omits `error_bound`.
Model chunking and communication logging remain available.

The default deployment is `examples/resources/configs/llama400m/deployment.json`
in the APPFL checkout. To select another JSON, use `--deployment FILE` with the
terminal launcher, or pass FILE as the optional first argument to the Slurm script:

```bash
bash examples/grpc/deploy_llama400m.sh server --deployment examples/resources/configs/llama400m/deployment_no_comp.json
bash examples/grpc/deploy_llama400m.sh client --client-index 1 --deployment examples/resources/configs/llama400m/deployment_no_comp.json
sbatch examples/grpc/deploy_llama400m.slurm examples/resources/configs/llama400m/deployment_no_comp.json
```

The selected JSON replaces the default for both preflight checks and training.
Relative custom paths are resolved from the calling/submission directory.
