# Llama-400M deployment configuration

For environment setup, execution, testing, and Slurm/srun instructions, see the
[Llama-400M execution guide](../../../grpc/README_llama400m.md).

Configure the deployment in `deployment.json`. One base client YAML serves all
clients; `base_server.yaml` holds shared training/model and server defaults.
**The scripts do not download the dataset or model by default.** Supply them before
training and configure their existing paths in the base YAMLs selected by
`deployment.json`: set `data_configs.dataset_kwargs.data_path` and the client
model's `model_dir` in `base_client`, and the server model's `model_dir` in
`base_server`. Model/data paths are fields in those referenced YAMLs, not direct
deployment JSON fields. **The model directory, dataset file, and its parent directory must already exist
and be readable on each host that uses them.**

The launcher creates the effective configuration inline, without temporary YAMLs.
The old numbered YAMLs remain available for direct `--config` launches.


Configuration preview (excerpts; keep the remaining settings in the real files).
Replace `/path/to/...` with existing paths on each host.

`deployment.json` selects the base YAML files (paths relative to the JSON):

```json
{
  "num_clients": 2,
  "base_client": "base_client.yaml",
  "base_server": "base_server.yaml",
  "enable_compression": true,
  "error_bound": 1e-10
}
```

`base_client.yaml` supplies the dataset and tokenizer/model directory:

```yaml
data_configs:
  dataset_kwargs:
    data_path: "/path/to/dataset/train.json.gz"
    model_dir: "/path/to/Llama-400M"
    num_clients: 2
comm_configs:
  grpc_configs:
    server_uri: "localhost:8888"
# Other client settings omitted.
```

`base_server.yaml` supplies the model checkpoint directory:

```yaml
client_configs:
  model_configs:
    model_kwargs:
      model_dir: "/path/to/Llama-400M"
      local_files_only: true
server_configs:
  num_clients: 2
  comm_configs:
    grpc_configs:
      server_uri: "localhost:8888"
# Other server settings omitted.
```

These excerpts do not download any assets. The model directory must contain the
checkpoint weights, configuration, and tokenizer. Deployment mode derives client
counts from JSON and overrides the YAML network defaults with CLI hostname/port.

Start one process in each terminal. The launcher prepares and activates the
repository-root `appfl_venv_v2` automatically, including
`pip install "appfl[examples,mpi]"` and an editable install of this checkout:

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
| `hostname`, `port` | Optional legacy address settings; defaults localhost and 8888; prefer CLI flags |
| `server_uri` | Optional legacy client target; explicit CLI host/port overrides it |
| `bind_host` | Optional listening interface; defaults to hostname |
| `enable_compression` | Enable or disable compression on every role |
| `error_bound` | Positive compressor error bound |
| `max_message_size`, `model_chunk_size` | Positive integer sizes in bytes |
| `use_model_chunking` | Enable or disable model chunking |
| `log_dir` | Parent directory for experiment logs |

The deployment JSONs omit network settings and the base YAMLs use
`localhost:8888`. Pass `--hostname SERVER_HOST --port 8888` for multiple hosts.
Slurm supplies the allocated server hostname automatically and defaults to port
8888; pass `--port` to change it. The server
listens on `bind_host:port`; `server_uri` affects clients only. IPv6 hostnames are
bracketed automatically. Set model/data paths in the base YAMLs for each machine.
The Bash launcher changes to the APPFL root so relative resource paths resolve.
Set `APPFL_VENV=/path/to/venv` to select a different environment. To install
dependencies before launch, run `bash examples/grpc/setup_llama400m_env.sh`.
See the execution guide for setup caching, MPI requirements, and Slurm details.

Before testing configuration with `--print-config`, install the environment and
dependencies successfully from the APPFL root:

```bash
bash examples/grpc/setup_llama400m_env.sh
```

For automated tests, also install `pytest` in that environment; see the
[configuration test prerequisites and commands](../../../grpc/README_llama400m.md#preview-and-test-configuration).

CLI flags override JSON, which overrides base YAML settings:

```bash
bash examples/grpc/deploy_llama400m.sh client --client-index 1 \
  --hostname server.example --port 50123 --error-bound 1e-6 \
  --max-message-size 8388608 --model-chunk-size 2097152 --print-config
```

Remove `--print-config` to start training. Apply shared experiment/network changes
consistently to all launches. CLI hostname/port override a JSON `server_uri`;
an explicit CLI `--server-uri` has the highest client-target precedence.

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

## Hostname, port, and non-Slurm ranks

The checked-in deployment JSONs omit network addresses. Without address flags,
`deploy_llama400m.sh` and deployment-mode `train_llama400m.py` use
`localhost:8888`; the base YAMLs use the same URI. For separate machines, supply
`--hostname SERVER_HOST --port PORT` to each role. For example:

```bash
bash examples/grpc/deploy_llama400m.sh server --hostname server.example.org --port 8888
bash examples/grpc/deploy_llama400m.sh client --client-index 1 --hostname server.example.org --port 8888
```

Use `--bind-host 0.0.0.0` on the server to listen on all IPv4 interfaces.
Slurm automatically supplies its dedicated server hostname; a custom port can
be supplied as a trailing `--port PORT` after the deployment path.

For automatic client readiness checks without Slurm, call
`run_llama400m_rank.sh LAUNCHER DEPLOYMENT SERVER_HOST LOG_DIR [training flags]`
once per process. Set `APPFL_RANK=0` for the server or `APPFL_RANK=K` for ClientK,
`APPFL_WORLD_SIZE=num_clients+1`, and `APPFL_VENV` to an existing environment's
absolute path. The wrapper exits if that environment is missing. Pass `--port`
to change its default of 8888. See the
[complete non-Slurm setup and launch example](../../../grpc/README_llama400m.md#use-the-rank-wrapper-without-slurm)
for per-host commands, logging, readiness, and GPU assignment.
