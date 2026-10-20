"""Launch an APPFL gRPC server or client for federated Llama-400M training.

Run from the APPFL repository root with an explicit --config path; model and
dataset paths in the Llama YAMLs are relative to that root.
"""

import argparse
import json
import math
import re
from pathlib import Path

from omegaconf import OmegaConf
from appfl.logger import ClientAgentFileLogger

import faulthandler
import signal

faulthandler.enable()
faulthandler.register(signal.SIGUSR1, all_threads=True)

OVERRIDE_KEYS = {"error_bound", "max_message_size", "model_chunk_size", "log_dir"}


def load_runtime_config(config, overrides=None):
    """Update a loaded OmegaConf in place and return it; source YAMLs stay intact.

    A YAML path is also accepted for direct configuration launches.
    """
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)
    overrides = overrides or {}
    unknown = set(overrides) - OVERRIDE_KEYS
    if unknown:
        raise ValueError(f"unknown override keys: {', '.join(sorted(unknown))}")
    for key in ("max_message_size", "model_chunk_size"):
        if key in overrides:
            value = overrides[key]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{key} must be a positive integer (bytes)")
    if "error_bound" in overrides:
        value = overrides["error_bound"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("error_bound must be a finite positive number")
    if "log_dir" in overrides and (not isinstance(overrides["log_dir"], str) or not overrides["log_dir"]):
        raise ValueError("log_dir must be a nonempty path string")

    is_server = "server_configs" in config
    role = config.server_configs if is_server else config
    comm = role.comm_configs
    grpc = comm.grpc_configs
    for key in ("max_message_size", "model_chunk_size"):
        if key in overrides:
            grpc[key] = overrides[key]
    # An error bound alone does not turn on compression for no-comp YAMLs.
    compressor = comm.get("compressor_configs")
    if compressor is not None and "error_bound" in overrides:
        compressor.error_bound = overrides["error_bound"]

    compressed = compressor is not None and compressor.get("enable_compression", False)
    mode = f"compress_{compressor.error_bound}" if compressed else "no_comp"
    identity = "server" if is_server else re.sub(r"[^a-zA-Z0-9_-]", "_", str(config.client_id))
    chunk = f"chunk_{grpc.model_chunk_size}" if grpc.get("use_model_chunking", False) else "unchunked"
    suffix = f"{mode}_{identity}_{chunk}_msg_{grpc.max_message_size}"
    train = role if is_server else config.train_configs
    root = Path(overrides.get("log_dir", Path(train.get("logging_output_dirname", "./output/llama400m/train")).parent))
    train.logging_output_dirname = str(root / f"train_{suffix}")
    comm.comm_logging_output_dirname = str(root / f"comm_{suffix}")
    if compressor is not None:
        compressor.compression_logging_output_dirname = str(root / f"compression_{suffix}")
    return config


DEPLOYMENT_KEYS = OVERRIDE_KEYS | {
    "num_clients", "hostname", "port", "server_uri", "bind_host",
    "base_client", "base_server", "enable_compression", "use_model_chunking",
}


def load_deployment_config(path, role, client_index=None, cli=None):
    """Build one role's configuration inline from shared deployment settings.

    Client numbers are one-based; dataset partition IDs are zero-based.
    Template paths resolve relative to the deployment JSON.
    """
    path = Path(path).resolve()
    settings = json.loads(path.read_text())
    if not isinstance(settings, dict):
        raise ValueError("deployment JSON must be an object")
    # Explicit CLI host/port must override a stored client URI too.
    if cli and ("hostname" in cli or "port" in cli) and "server_uri" not in cli:
        settings.pop("server_uri", None)
    settings.update(cli or {})
    unknown = set(settings) - DEPLOYMENT_KEYS
    if unknown:
        raise ValueError(f"unknown deployment keys: {', '.join(sorted(unknown))}")
    count = settings.get("num_clients", 2)
    port = settings.get("port", 8888)
    for name, value in (("num_clients", count), ("port", port)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if port > 65535:
        raise ValueError("port must be at most 65535")
    for name in ("hostname", "bind_host", "server_uri", "base_client", "base_server"):
        if name in settings and (not isinstance(settings[name], str) or not settings[name].strip()):
            raise ValueError(f"{name} must be a nonempty string")
    for name in ("enable_compression", "use_model_chunking"):
        if name in settings and not isinstance(settings[name], bool):
            raise ValueError(f"{name} must be a boolean")
    if role == "client":
        if isinstance(client_index, bool) or not isinstance(client_index, int) or not 1 <= client_index <= count:
            raise ValueError(f"client requires --client-index between 1 and {count}")
    elif client_index is not None:
        raise ValueError("--client-index is only valid for clients")
    template = settings.get(f"base_{role}", f"base_{role}.yaml")
    config = OmegaConf.load(path.parent / template)
    is_server = role == "server"
    if ("server_configs" in config) != is_server:
        raise ValueError(f"base_{role} has the wrong role structure")
    if is_server:
        config.server_configs.num_clients = count
        role_config = config.server_configs
    else:
        config.client_id = f"Client{client_index}"
        config.data_configs.dataset_kwargs.num_clients = count
        config.data_configs.dataset_kwargs.client_id = client_index - 1
        config.train_configs.logging_output_filename = f"client_{client_index}"
        role_config = config
    comm = role_config.comm_configs
    def endpoint(host):
        # Accept both bare IPv6 hosts and bracketed IPv6 hosts.
        host = f"[{host}]" if ":" in host and not host.startswith("[") else host
        return f"{host}:{port}"
    comm.grpc_configs.server_uri = (
        endpoint(settings.get("bind_host", settings.get("hostname", "localhost"))) if is_server
        else settings.get("server_uri", endpoint(settings.get("hostname", "localhost")))
    )
    if "use_model_chunking" in settings:
        comm.grpc_configs.use_model_chunking = settings["use_model_chunking"]
    if "enable_compression" in settings:
        if "compressor_configs" not in comm and settings["enable_compression"]:
            raise ValueError("base YAML needs compressor_configs to enable compression")
        if "compressor_configs" in comm:
            comm.compressor_configs.enable_compression = settings["enable_compression"]
    return load_runtime_config(config, {key: settings[key] for key in OVERRIDE_KEYS if key in settings})


def create_compressor_logger(agent_config):
    """Create a communication logger using only an OmegaConf agent config."""
    is_server = "server_configs" in agent_config
    role_config = agent_config.server_configs if is_server else agent_config
    comm_config = role_config.get("comm_configs", {})
    compression_config = comm_config.get("compressor_configs", {})
                         


    return ClientAgentFileLogger(
        logging_id="Server" if is_server else str(agent_config.get("client_id", "")),
        file_dir=compression_config.get("compression_logging_output_dirname", "./output/llama400m"),
        file_name=compression_config.get("compression_logging_output_filename", "compression_result"),
        experiment_id=str(agent_config.get("experiment_id", "")),
    )

def create_comm_logger(agent_config):
    """Create a communication logger using only an OmegaConf agent config."""
    is_server = "server_configs" in agent_config
    role_config = agent_config.server_configs if is_server else agent_config
    comm_config = role_config.get("comm_configs", {})

    return ClientAgentFileLogger(
        logging_id="Server" if is_server else str(agent_config.get("client_id", "")),
        file_dir=comm_config.get("comm_logging_output_dirname", "./output/llama400m"),
        file_name=comm_config.get("comm_logging_output_filename", "comm_result"),
        experiment_id=str(agent_config.get("experiment_id", "")),
    )

def run_server(config) -> None:
    from appfl.agent import ServerAgent
    from appfl.comm.grpc import GRPCServerCommunicator, serve

    compression_logger = create_compressor_logger(config)
    comm_logger = create_comm_logger(config)
    agent = ServerAgent(server_agent_config=config, 
                        compression_logger=compression_logger,
                        comm_logger=comm_logger)
    grpc_config = config.server_configs.comm_configs.grpc_configs
    communicator = GRPCServerCommunicator(
        agent, logger=agent.logger, **grpc_config
    )
    serve(communicator, **grpc_config)


def run_client(config) -> None:
    from appfl.agent import ClientAgent
    from appfl.comm.grpc import GRPCClientCommunicator

    compression_logger = create_compressor_logger(config)
    comm_logger = create_comm_logger(config)
    agent = ClientAgent(client_agent_config=config, 
                        compression_logger=compression_logger,
                        comm_logger=comm_logger)
    grpc_config = config.comm_configs.grpc_configs
    communicator = GRPCClientCommunicator(
        client_id=agent.get_id(),
        logger=agent.logger,
        comm_logger=comm_logger,
        **grpc_config,
    )
    shared_config = communicator.get_configuration()
    agent.load_config(shared_config)
    global_model = communicator.get_global_model(init_model=True)
    global_model, metadata = (
        global_model if isinstance(global_model, tuple) else (global_model, {})
    )
    agent.load_parameters(global_model)
    communicator.invoke_custom_action(
        action="set_sample_size", sample_size=agent.get_sample_size()
    )

    while True:
        agent.train(**metadata)
        local_model = agent.get_parameters()
        local_model, metadata = (
            local_model if isinstance(local_model, tuple) else (local_model, {})
        )
        global_model, metadata = communicator.update_global_model(
            local_model, **metadata
        )
        if metadata["status"] == "DONE":
            break
        if "local_steps" in metadata:
            agent.trainer.train_configs.num_local_steps = metadata["local_steps"]
        agent.load_parameters(global_model)
    communicator.invoke_custom_action(action="close_connection")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("server", "client"))
    parser.add_argument("--deployment", type=Path, help="Deployment JSON; builds one role inline")
    parser.add_argument("--client-index", type=int, help="Client number, from 1 to num_clients")
    parser.add_argument("--num-clients", type=int)
    parser.add_argument("--hostname", help="Server hostname (default: localhost)")
    parser.add_argument("--port", type=int, help="Server port (default: 8888)")
    parser.add_argument("--server-uri", help="Explicit client target, takes precedence over hostname/port")
    parser.add_argument("--bind-host", help="Server listening interface")
    parser.add_argument("--config", help="YAML configuration path")
    parser.add_argument("--overrides", type=Path, help="Shared JSON experiment settings")
    parser.add_argument("--error-bound", type=float)
    parser.add_argument("--max-message-size", type=int, help="gRPC message limit in bytes")
    parser.add_argument("--model-chunk-size", type=int, help="Model chunk size in bytes")
    parser.add_argument("--log-dir", help="Root directory for experiment logs")
    parser.add_argument("--print-config", action="store_true", help="Print effective YAML and exit without training")
    args = parser.parse_args()
    default = (
        str(Path(__file__).resolve().parents[1] / "resources/configs/llama400m/server_fedavg.yaml")
        if args.role == "server"
        else str(Path(__file__).resolve().parents[1] / "resources/configs/llama400m/client_1.yaml")
    )
    config_path = args.config or default
    try:
        cli = {key: getattr(args, key) for key in OVERRIDE_KEYS if getattr(args, key) is not None}
        network_cli = {key: getattr(args, key) for key in ("num_clients", "hostname", "port", "server_uri", "bind_host") if getattr(args, key) is not None}
        if args.deployment:
            if args.config or args.overrides:
                raise ValueError("--deployment cannot be combined with --config or --overrides")
            config = load_deployment_config(args.deployment, args.role, args.client_index, {**cli, **network_cli})
        else:
            if args.client_index is not None or network_cli:
                raise ValueError("client index and deployment network options require --deployment")
            overrides = json.loads(args.overrides.read_text()) if args.overrides else {}
            if not isinstance(overrides, dict):
                raise ValueError("overrides JSON must be an object")
            overrides.update(cli)
            config = load_runtime_config(config_path, overrides)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.print_config:
        print(OmegaConf.to_yaml(config))
        return
    (run_server if args.role == "server" else run_client)(config)


if __name__ == "__main__":
    main()
