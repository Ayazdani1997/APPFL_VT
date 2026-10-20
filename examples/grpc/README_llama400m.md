# Test federated Llama-400M training

The example trains a locally stored Hugging Face Llama checkpoint using APPFL,
gRPC, deterministic dataset partitions, and optional SZ2 compression. Use
`train_llama400m.py` through the Bash launcher or the Slurm batch script below.

## Prepare the environment

Run commands from the APPFL repository root. The launcher automatically prepares
and activates `appfl_venv_v2` in that root. You can prepare it separately before
training or submitting a job (recommended when compute nodes lack internet):

```bash
bash examples/grpc/setup_llama400m_env.sh
```

The setup script reuses the existing environment or creates it with
`${PYTHON:-python3} -m venv`. Inside it, setup runs
`python -m pip install "appfl[examples,mpi]"`, installs `transformers` and
`accelerate`, then installs this checkout in editable mode with `--no-deps` so
training uses your local APPFL changes. Setup requires package-index access and
an MPI installation suitable for `mpi4py`; load your site's MPI/CUDA modules first.
Compressed runs also require the SZ2 and Blosc dependencies used by APPFL's
`SZ2Compressor`. Use a CUDA-enabled PyTorch installation for GPU clients.

A successful setup writes `appfl_venv_v2/.llama400m-ready`; subsequent launches
reuse it. A `flock` lock serializes setup across simultaneous launches on the
shared filesystem. Failed installs do not write the marker and are retried.
Remove the marker and rerun setup when dependencies need reinstalling.
Set `APPFL_VENV=/shared/path/to/venv` consistently for setup and launch to use
another environment. `PYTHON` selects only the interpreter for creating a new
environment; training always uses the selected environment's Python. Setup output
goes to stderr so `--print-config` stdout remains valid YAML.

Edit these files before training:

- [base_client.yaml](../resources/configs/llama400m/base_client.yaml): set
  `data_configs.dataset_kwargs.data_path` to your text/JSONL dataset (optionally
  gzip), and `model_dir` to the checkpoint/tokenizer directory.
- [base_server.yaml](../resources/configs/llama400m/base_server.yaml): set
  `client_configs.model_configs.model_kwargs.model_dir` to the same model.
  Set global epochs, local training steps, batch size, and optimizer settings here.
- [deployment.json](../resources/configs/llama400m/deployment.json): set client
  count, error bound, chunk/message sizes, compression mode,
  and output directory. Sizes are in bytes.

**The scripts do not download the dataset or model by default.** Provide both before
training: `deployment.json` selects the `base_client` and `base_server` YAMLs,
which must specify your existing dataset path (`data_path`) and model directory
(`model_dir`). These paths are configured in the referenced YAMLs, not as direct
fields in the deployment JSON. **The model directory, dataset file, and its parent directory must already exist
and be readable on every host that uses them.**
The checked-in model and dataset paths are machine-specific and must be updated.
The model is loaded locally (`local_files_only: true`); prepare its weights,
configuration, and tokenizer first. Each client reads the same source dataset
and selects its own partition. `max_documents` limits startup work; ensure enough
tokens remain for every client's partition. The default server uses CPU, while
clients use CUDA. Adjust the bases for your available memory/devices.


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

See the [configuration reference](../resources/configs/llama400m/README.md) for
all deployment fields and precedence rules.

## Preview and test configuration

Prerequisite: complete [Prepare the environment](#prepare-the-environment)
successfully before running configuration previews or tests. From the APPFL
root, install the environment and the test runner first:

```bash
export APPFL_VENV="${APPFL_VENV:-$PWD/appfl_venv_v2}"
bash examples/grpc/setup_llama400m_env.sh
"$APPFL_VENV/bin/python" -m pip install pytest
```

```bash
bash examples/grpc/deploy_llama400m.sh server --print-config
bash examples/grpc/deploy_llama400m.sh client --client-index 1 --print-config
"$APPFL_VENV/bin/python" -m pytest tests/test_llama400m_runtime_config.py -q
```

Previews show the effective YAML and exit without loading the model or starting
network services. Internally, the loaded OmegaConf is updated in place and passed
directly to the agent. No temporary YAML files are written.

## Run in separate terminals

Each launcher activates the environment automatically:

```bash
# Terminal 1: start the server first; wait for it to listen.
bash examples/grpc/deploy_llama400m.sh server
# Terminal 2
bash examples/grpc/deploy_llama400m.sh client --client-index 1
# Terminal 3
bash examples/grpc/deploy_llama400m.sh client --client-index 2
```

Start clients 1 through `num_clients`, once each. Each command runs exactly one
foreground process. Client K receives ID `ClientK` and dataset partition K-1.
For different hosts, pass `--hostname SERVER_HOST --port 8888` to every launcher.
With no address flags, the default is `localhost:8888`. The server listens on
that hostname unless `--bind-host` selects a different listening interface.
The deployment JSONs omit network settings; the base YAMLs use `localhost:8888`. Start each client on its own host
or GPU; separate terminals alone do not isolate GPU resources.

```bash
APPFL_VENV=/path/to/venv bash examples/grpc/deploy_llama400m.sh client \
  --client-index 1 --deployment /shared/experiment/deployment.json \
  --error-bound 1e-6 --model-chunk-size 2097152 --max-message-size 8388608
```

Use consistent shared overrides for all roles. Training, compression, and
communication log directories include the effective error bound (or `no_comp`),
role/client ID, chunk size, and message size. Repeated identical runs reuse the
same directories; choose a new `log_dir` for independent experiments.

## Use the rank wrapper without Slurm

`run_llama400m_rank.sh` can also be called from Bash, SSH, or another scheduler.
Run it once per process on the intended machine; it does not launch remote
processes itself. Its positional arguments are:

```text
bash run_llama400m_rank.sh LAUNCHER DEPLOYMENT SERVER_HOST LOG_DIR [training flags]
```

Set `APPFL_RANK` (0 for server, K for ClientK), `APPFL_WORLD_SIZE`
(`num_clients + 1`), and `APPFL_VENV` (absolute environment path on that host).
These rank variables take precedence over `SLURM_PROCID` and `SLURM_NTASKS`.
An existing environment is required: run `setup_llama400m_env.sh` explicitly
on each host first. A missing environment makes the wrapper exit with status 2.
The wrapper installs any pending dependencies, activates the environment,
validates configuration, and calls `deploy_llama400m.sh`.

For two clients, run the following setup on **each host**, from its APPFL root:

```bash
export APPFL_VENV="$PWD/appfl_venv_v2"
bash examples/grpc/setup_llama400m_env.sh
launcher="$PWD/examples/grpc/deploy_llama400m.sh"
deployment="$PWD/examples/resources/configs/llama400m/deployment.json"
logs="$PWD/output/llama400m/manual-run"
server_host=server.example.org  # Reachable hostname of your chosen server
```

Then execute one command per host/terminal, using the same hostname and port:

```bash
# Server host
APPFL_RANK=0 APPFL_WORLD_SIZE=3 bash examples/grpc/run_llama400m_rank.sh \
  "$launcher" "$deployment" "$server_host" "$logs" --port 8888

# First client host
APPFL_RANK=1 APPFL_WORLD_SIZE=3 bash examples/grpc/run_llama400m_rank.sh \
  "$launcher" "$deployment" "$server_host" "$logs" --port 8888

# Second client host
APPFL_RANK=2 APPFL_WORLD_SIZE=3 bash examples/grpc/run_llama400m_rank.sh \
  "$launcher" "$deployment" "$server_host" "$logs" --port 8888
```

The third positional argument supplies the hostname; use `--port` for the port
(default 8888). For a single-machine run, use `localhost`. The wrapper binds the
server to `0.0.0.0`; clients wait for gRPC readiness, with
`SERVER_READY_TIMEOUT=600` seconds by default. It requires `use_ssl: false`.
Use matching deployment settings and port on all hosts. Each process stays in
the foreground and redirects output to `LOG_DIR/rank_K.out` and `rank_K.err`
on its own host. Storage need not be shared. Assign GPUs yourself, for example
with `CUDA_VISIBLE_DEVICES`; the wrapper does not allocate resources. Outside
Slurm, manage peer termination yourself if a process fails.

## Run with Slurm and srun

[deploy_llama400m.slurm](deploy_llama400m.slurm) launches a single `srun`
step that runs [run_llama400m_rank.sh](run_llama400m_rank.sh) once per task/rank: rank 0 runs the server, and rank K runs ClientK
with dataset partition K-1. Every rank executes `deploy_llama400m.sh` for its
role. Slurm runs exactly one task/rank per node. The rank wrapper first checks that its environment exists, then runs
`setup_llama400m_env.sh`, activates the environment, and validates configuration
before starting its role. The first node in Slurm's allocated node list is dedicated to the server. A temporary
hostfile read by `srun` pins rank 0 to that node and each remaining rank to a
different node. The hostname is passed directly to all wrappers; clients wait
for its gRPC service. No shared rendezvous file is needed.

The defaults request **three nodes and three tasks, one task per node**, eight
CPUs and one GPU per task, and 64 GiB memory per node.
The CPU server also reserves a GPU to keep task resource requests uniform.
Adjust CPUs, memory, GPUs, time, partition, and account for your cluster. The
example expects Slurm GPU resources and unencrypted gRPC (`use_ssl: false`).

Load your site's MPI/CUDA modules before submission. The batch script calls only
the rank wrapper through a single `srun`; it performs no Python setup or preflight
on the batch host. Each wrapper exits with status 2 if the environment Python or
activation script is missing on its node. Create the environment explicitly with
`setup_llama400m_env.sh` on each node (or once on shared storage) before submission.
For an existing environment, the wrapper installs dependencies before using Python;
a shared environment reuses the completed installation under the setup lock.
Each training rank activates its node's environment and executes the launcher.
The checkout, model, dataset, and JSON must be accessible at the configured paths
on all nodes. Output directories may be node-local: each wrapper creates its log
directory on its own node before redirecting stdout/stderr. `APPFL_VENV` may
point to the same absolute node-local path on each node (for example,
`/tmp/your-user/appfl_venv_v2`) or to shared storage. Nodes needing installation
require package-index access and a bootstrap Python with venv support. Preparing
only a login-node environment does not prepare node-local environments elsewhere.
Submit from the APPFL root:

```bash
sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION \
  examples/grpc/deploy_llama400m.slurm
```

For four clients, set `num_clients` to 4 and request five nodes and five tasks:

```bash
sbatch --nodes=5 --ntasks=5 --ntasks-per-node=1 \
  --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION \
  examples/grpc/deploy_llama400m.slurm /shared/experiment/deployment.json \
  --error-bound 1e-6 --max-message-size 8388608
```

After setup, each rank checks the server configuration, verifies that the task
count equals `num_clients + 1`, and checks that SSL is disabled. Client ranks also
validate their own configuration. These previews do not load model weights or
datasets or verify GPU availability. Setup and validation errors terminate peer
ranks through `--kill-on-bad-exit=1`.

The first script argument is the JSON path; subsequent arguments are shared
training flags. `#SBATCH` directives are static: request exactly
`num_clients + 1` nodes and tasks, with `--ntasks-per-node=1`. Slurm assigns each
rank its own GPU visibility.

If submitting elsewhere, export `APPFL_ROOT=/shared/path/to/APPFL`. The server
binds to `0.0.0.0` at the configured port; clients use the dedicated server's
Slurm hostname, overriding any JSON `server_uri`. That hostname must resolve
and the port must be reachable between nodes. Pass `--port 9012` after the
optional deployment path to select another port; the default is 8888. The wrapper
passes the same host and port explicitly to `train_llama400m.py` for both roles.

The readiness timeout is 600 seconds; export `SERVER_READY_TIMEOUT=1200` for
slower server setup or model loading. The timer starts after the client
finishes its own setup. `srun --kill-on-bad-exit=1` terminates peer ranks on failure.
Cancel the allocation with `scancel JOB_ID`.

Inspect `llama400m-JOB_ID.out/.err` for batch status and
`output/llama400m/slurm-JOB_ID/rank_0.out` and `rank_0.err` for the server;
`rank_K.out/.err` contain ClientK output. With node-local storage, retrieve each
rank's logs from its assigned node; they are not collected automatically. The
rank/node mapping follows the allocated node list. Agent logs remain under deployment
`log_dir`. A completed run should finish the configured global epochs and exit
successfully; this guide does not assert model quality.

Slurm option reference: [srun](https://slurm.schedmd.com/srun.html) and
[sbatch](https://slurm.schedmd.com/sbatch.html).

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
