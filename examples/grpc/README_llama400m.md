# Test federated Llama-400M training

The example trains a locally stored Hugging Face Llama checkpoint using APPFL,
gRPC, deterministic dataset partitions, and optional SZ2 compression. Use
`train_llama400m.py` through the Bash launcher or the Slurm batch script below.

## Prepare the environment

Run commands from the APPFL repository root. Activate an environment with this
checkout installed and the example dependencies available (`torch`, `transformers`,
`accelerate`, `omegaconf`, and `grpcio`). Compressed runs also require the SZ2 and
Blosc dependencies used by APPFL's `SZ2Compressor`. Use a CUDA-enabled PyTorch
installation for the default GPU client configuration.

Edit these files before training:

- [base_client.yaml](../resources/configs/llama400m/base_client.yaml): set
  `data_configs.dataset_kwargs.data_path` to your text/JSONL dataset (optionally
  gzip), and `model_dir` to the checkpoint/tokenizer directory.
- [base_server.yaml](../resources/configs/llama400m/base_server.yaml): set
  `client_configs.model_configs.model_kwargs.model_dir` to the same model.
  Set global epochs, local training steps, batch size, and optimizer settings here.
- [deployment.json](../resources/configs/llama400m/deployment.json): set client
  count, network address, error bound, chunk/message sizes, compression mode,
  and output directory. Sizes are in bytes.

The checked-in model and dataset paths are machine-specific and must be updated.
The model is loaded locally (`local_files_only: true`); prepare its weights,
configuration, and tokenizer first. Each client reads the same source dataset
and selects its own partition. `max_documents` limits startup work; ensure enough
tokens remain for every client's partition. The default server uses CPU, while
clients use CUDA. Adjust the bases for your available memory/devices.

See the [configuration reference](../resources/configs/llama400m/README.md) for
all deployment fields and precedence rules.

## Preview and test configuration

```bash
bash examples/grpc/deploy_llama400m.sh server --print-config
bash examples/grpc/deploy_llama400m.sh client --client-index 1 --print-config
python -m pytest tests/test_llama400m_runtime_config.py -q
```

Previews show the effective YAML and exit without loading the model or starting
network services. Internally, the loaded OmegaConf is updated in place and passed
directly to the agent. No temporary YAML files are written.

## Run in separate terminals

With the same environment activated in each terminal:

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
For different hosts, set the server's reachable `hostname` and `port` in the same
JSON used on all hosts. The server listens on `bind_host:port`; `server_uri`, when
present, explicitly sets the clients' target. Start each client on its own host
or GPU; separate terminals alone do not isolate GPU resources.

```bash
PYTHON=/path/to/venv/bin/python bash examples/grpc/deploy_llama400m.sh client \
  --client-index 1 --deployment /shared/experiment/deployment.json \
  --error-bound 1e-6 --model-chunk-size 2097152 --max-message-size 8388608
```

Use consistent shared overrides for all roles. Training, compression, and
communication log directories include the effective error bound (or `no_comp`),
role/client ID, chunk size, and message size. Repeated identical runs reuse the
same directories; choose a new `log_dir` for independent experiments.

## Run with Slurm and srun

[deploy_llama400m.slurm](deploy_llama400m.slurm) launches a single `srun`
step that runs [run_llama400m_rank.sh](run_llama400m_rank.sh) once per task/rank: rank 0 runs the server, and rank K runs ClientK
with dataset partition K-1. Slurm places the ranks on the allocated nodes;
multiple ranks can share a node. Rank 0 publishes its actual hostname through a
unique directory on the shared filesystem. Clients wait for its gRPC service.

The defaults request **one node and three tasks**, eight CPUs and one GPU per
task, and 64 GiB memory for the node. This requires three GPUs on that node.
The CPU server also reserves a GPU to keep task resource requests uniform.
Adjust CPUs, memory, GPUs, time, partition, and account for your cluster. The
example expects Slurm GPU resources and unencrypted gRPC (`use_ssl: false`).

Activate the Python environment and load your site's CUDA/modules before
submission. The checkout, environment, model, dataset, JSON, and output directory
must be accessible on all allocated nodes. Submit from the APPFL root:

```bash
export PYTHON=/shared/venv/bin/python
sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION \
  examples/grpc/deploy_llama400m.slurm
```

For four clients, set `num_clients` to 4 and request five tasks. You can spread
those ranks across nodes, for example three on one node and two on another
(subject to available CPUs/GPUs and cluster policy):

```bash
sbatch --nodes=2 --ntasks=5 --ntasks-per-node=3 \
  --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION \
  examples/grpc/deploy_llama400m.slurm /shared/experiment/deployment.json \
  --error-bound 1e-6 --max-message-size 8388608
```

Before launching ranks, the batch script builds the server and every client
configuration and checks that SSL is disabled for this example. These previews
do not load model weights or datasets or verify GPU availability.

The first script argument is the JSON path; subsequent arguments are shared
training flags. `#SBATCH` directives are static: request at least
`num_clients + 1` tasks at submission. The script launches exactly that many
ranks and checks the task allocation. Node count is independent of client count;
choose a layout with sufficient per-task CPUs and GPUs. Slurm assigns each rank
its own GPU visibility.

If submitting elsewhere, export `APPFL_ROOT=/shared/path/to/APPFL`. The server
binds to `0.0.0.0` at the configured port; clients use rank 0's discovered hostname,
overriding any JSON `server_uri`. Inter-node communication on that port and a
shared output filesystem are required when using multiple nodes.

The readiness timeout is 600 seconds; export `SERVER_READY_TIMEOUT=1200` for
slower model loading. `srun --kill-on-bad-exit=1` terminates peer ranks on failure.
Cancel the allocation with `scancel JOB_ID`.

Inspect `llama400m-JOB_ID.out/.err` for batch status and
`output/llama400m/slurm-JOB_ID/rank_0.out` and `rank_0.err` for the server;
`rank_K.out/.err` contain ClientK output. Agent logs remain under deployment
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
