from concurrent.futures import ThreadPoolExecutor

import pytest

from appfl.misc.timing import collect_latencies, measure_latency


def test_nested_scope_and_exception_restore_outer_context(monkeypatch):
    ticks = iter([1.0, 1.25, 2.0, 2.5, 3.0, 4.0])
    monkeypatch.setattr("appfl.misc.timing.perf_counter", lambda: next(ticks))
    with collect_latencies() as outer:
        with measure_latency("copy"):
            pass
        with pytest.raises(ValueError):
            with collect_latencies() as inner:
                with measure_latency("copy"):
                    raise ValueError("test")
        with measure_latency("copy"):
            pass
    # No active collector: this must not consume another clock tick.
    with measure_latency("copy"):
        pass
    assert outer.total_seconds == {"copy": 1.25}
    assert inner.total_seconds == {"copy": 0.5}


def test_thread_request_has_separate_collection():
    def request():
        with collect_latencies() as context:
            with measure_latency("copy"):
                pass
        return context

    with collect_latencies() as outer:
        with ThreadPoolExecutor(max_workers=1) as executor:
            inner = executor.submit(request).result()
    assert outer.total_seconds == {}
    assert inner.total_seconds["copy"] >= 0.0


def test_sz2_collects_each_non_scalar_copy_in_batched_models(monkeypatch):
    import torch
    from omegaconf import OmegaConf
    from appfl.compressor import SZ2Compressor

    compressor = SZ2Compressor(
        OmegaConf.create({"lossless_compressor": "gzip", "param_cutoff": 100})
    )
    model = torch.nn.Linear(2, 2)
    payload = compressor.compress_model([model.state_dict()] * 2, batched=True)
    ticks = iter(range(16))
    monkeypatch.setattr("appfl.misc.timing.perf_counter", lambda: next(ticks))
    with collect_latencies() as context:
        decoded = compressor.decompress_model(payload, model, batched=True)
    assert context.total_seconds == {
        "SZ2 NumPy Frombuffer Time (s)": 4.0,
        "SZ2 Tensor Copy Time (s)": 4.0,
    }
    for weights in decoded:
        for name, expected in model.state_dict().items():
            torch.testing.assert_close(weights[name], expected)


def test_disabled_collection_does_not_read_clock(monkeypatch):
    def unexpected_clock_read():
        pytest.fail("Disabled measurements must not read the clock")

    monkeypatch.setattr("appfl.misc.timing.perf_counter", unexpected_clock_read)
    with collect_latencies() as outer:
        with collect_latencies(enabled=False) as disabled:
            with measure_latency("copy"):
                pass
    assert disabled is None
    assert outer.total_seconds == {}


def test_named_counters_accumulate_independently(monkeypatch):
    ticks = iter([0.0, 0.25, 1.0, 1.5, 2.0, 3.0])
    monkeypatch.setattr("appfl.misc.timing.perf_counter", lambda: next(ticks))
    with collect_latencies() as context:
        with measure_latency(key="copy"):
            pass
        with measure_latency(key="convert"):
            pass
        with measure_latency(key="copy"):
            pass
    assert context.total_seconds == {"copy": 1.25, "convert": 0.5}
