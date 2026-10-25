"""Runtime settings must agree across all Llama server/client configurations."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("train_llama400m", ROOT / "examples/grpc/train_llama400m.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


@pytest.mark.parametrize("path", sorted((ROOT / "examples/resources/configs/llama400m").glob("*.yaml")))
def test_shared_overrides_and_log_paths(path, tmp_path):
    original = path.read_bytes()
    settings = dict(error_bound=0.002, max_message_size=8388608, model_chunk_size=2097152, log_dir=str(tmp_path))
    config = launcher.load_runtime_config(path, settings)
    server = "server_configs" in config
    role = config.server_configs if server else config
    comm = role.comm_configs
    assert comm.grpc_configs.max_message_size == 8388608
    assert comm.grpc_configs.model_chunk_size == 2097152
    compressed = "compressor_configs" in comm
    train = role if server else config.train_configs
    directories = [train.logging_output_dirname, comm.comm_logging_output_dirname]
    if compressed:
        assert comm.compressor_configs.error_bound == 0.002
        directories.append(comm.compressor_configs.compression_logging_output_dirname)
    for directory in directories:
        assert str(tmp_path) in directory
        assert ("compress_0.002" if compressed else "no_comp") in directory
        assert "chunk_2097152_msg_8388608" in directory
        assert ("server" if server else config.client_id) in directory
    assert path.read_bytes() == original


@pytest.mark.parametrize("settings", [dict(error_bound=-1), dict(error_bound=float("nan")), dict(max_message_size=0), dict(model_chunk_size=1.5), dict(max_message_size=True), dict(typo=4)])
def test_invalid_overrides(settings):
    with pytest.raises(ValueError):
        launcher.load_runtime_config(ROOT / "examples/resources/configs/llama400m/client_1.yaml", settings)


def deployment_file(tmp_path, **settings):
    import json
    folder = ROOT / "examples/resources/configs/llama400m"
    values = dict(base_client=str(folder / "base_client.yaml"), base_server=str(folder / "base_server.yaml"), num_clients=3, hostname="server.example", port=50123, bind_host="0.0.0.0", error_bound=0.003)
    values.update(settings)
    path = tmp_path / "deployment.json"
    path.write_text(json.dumps(values))
    return path


def test_deployment_partitioning_and_server(tmp_path):
    path = deployment_file(tmp_path)
    server = launcher.load_deployment_config(path, "server")
    assert server.server_configs.num_clients == 3
    assert server.server_configs.comm_configs.grpc_configs.server_uri == "0.0.0.0:50123"
    for index in range(1, 4):
        client = launcher.load_deployment_config(path, "client", index)
        assert client.client_id == f"Client{index}"
        assert client.data_configs.dataset_kwargs.client_id == index - 1
        assert client.data_configs.dataset_kwargs.num_clients == 3
        assert client.comm_configs.grpc_configs.server_uri == "server.example:50123"
        assert f"Client{index}" in client.train_configs.logging_output_dirname
        assert client.comm_configs.compressor_configs.error_bound == 0.003


def test_client_device_json_and_cli_precedence(tmp_path):
    path = deployment_file(tmp_path, cuda_device="cuda:2")
    client = launcher.load_deployment_config(path, "client", 1)
    assert client.train_configs.device == "cuda:2"
    client = launcher.load_deployment_config(path, "client", 1, dict(cuda_device="cpu"))
    assert client.train_configs.device == "cpu"
    server = launcher.load_deployment_config(path, "server")
    assert "train_configs" not in server


@pytest.mark.parametrize("device", [0, True, "", "cuda:-1", "invalid"])
def test_invalid_client_device(tmp_path, device):
    with pytest.raises(ValueError, match="cuda_device"):
        launcher.load_deployment_config(deployment_file(tmp_path, cuda_device=device), "client", 1)


def test_deployment_network_precedence_and_no_compression(tmp_path):
    path = deployment_file(tmp_path, server_uri="dns:///explicit:5000", enable_compression=False)
    client = launcher.load_deployment_config(path, "client", 1, dict(hostname="other", port=6000, error_bound=0.5))
    assert client.comm_configs.grpc_configs.server_uri == "other:6000"
    assert not client.comm_configs.compressor_configs.enable_compression
    assert "no_comp" in client.train_configs.logging_output_dirname
    server = launcher.load_deployment_config(path, "server", cli=dict(bind_host="::", port=6000))
    assert server.server_configs.comm_configs.grpc_configs.server_uri == "[::]:6000"


@pytest.mark.parametrize("settings,index", [(dict(num_clients=0), 1), (dict(port=65536), 1), (dict(enable_compression="false"), 1), ({}, 0), ({}, 4), ({}, None), (dict(hostname=""), 1)])
def test_invalid_deployment(tmp_path, settings, index):
    with pytest.raises(ValueError):
        launcher.load_deployment_config(deployment_file(tmp_path, **settings), "client", index)


def test_bash_launcher_preview_from_explicit_deployment(tmp_path):
    import os
    import subprocess
    import sys
    from omegaconf import OmegaConf
    script = ROOT / "examples/grpc/deploy_llama400m.sh"
    path = deployment_file(tmp_path)
    # Provide an already prepared environment without installing over the network.
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    import shlex
    python_wrapper = venv / "bin/python"
    python_wrapper.write_text(f'#!/bin/bash\nexec {shlex.quote(sys.executable)} "$@"\n')
    python_wrapper.chmod(0o755)
    (venv / "bin/activate").write_text("")
    (venv / ".llama400m-ready").touch()
    for role, extra in [("server", []), ("client", ["--client-index", "3"])]:
        result = subprocess.run(["bash", str(script), role, "--deployment", str(path), *extra, "--print-config"], cwd=tmp_path, env={**os.environ, "APPFL_VENV": str(venv)}, text=True, capture_output=True, check=True)
        config = OmegaConf.create(result.stdout)
        comm = config.server_configs.comm_configs if role == "server" else config.comm_configs
        assert comm.compressor_configs.error_bound == 0.003


@pytest.mark.parametrize("name", ["base_client.yaml", "base_server.yaml"])
def test_runtime_updates_loaded_omegaconf_in_place(name):
    from omegaconf import OmegaConf
    config = OmegaConf.load(ROOT / "examples/resources/configs/llama400m" / name)
    role = config.server_configs if "server_configs" in config else config
    grpc = role.comm_configs.grpc_configs
    result = launcher.load_runtime_config(config, dict(error_bound=0.125, max_message_size=8388608, model_chunk_size=2097152))
    assert result is config
    assert grpc.max_message_size == 8388608
    assert grpc.model_chunk_size == 2097152
    assert role.comm_configs.compressor_configs.error_bound == 0.125
    train = role if "server_configs" in config else config.train_configs
    assert "compress_0.125" in train.logging_output_dirname


@pytest.mark.parametrize("role,index", [("server", None), ("client", 1), ("client", 2)])
def test_no_compression_deployment_has_no_compressor_settings(role, index):
    config = launcher.load_deployment_config(ROOT / "examples/resources/configs/llama400m/deployment_no_comp.json", role, index)
    comm = config.server_configs.comm_configs if role == "server" else config.comm_configs
    assert "compressor_configs" not in comm
    assert "no_comp" in comm.comm_logging_output_dirname


def test_deployment_default_and_cli_address(tmp_path):
    import json
    path = deployment_file(tmp_path)
    values = json.loads(path.read_text())
    for key in ("hostname", "port", "bind_host"):
        values.pop(key)
    path.write_text(json.dumps(values))
    for role in ("server", "client"):
        index = 1 if role == "client" else None
        config = launcher.load_deployment_config(path, role, index)
        comm = config.server_configs.comm_configs if role == "server" else config.comm_configs
        assert comm.grpc_configs.server_uri == "localhost:8888"
        config = launcher.load_deployment_config(path, role, index, dict(hostname="node01", port=9012))
        comm = config.server_configs.comm_configs if role == "server" else config.comm_configs
        assert comm.grpc_configs.server_uri == "node01:9012"
