import csv
from concurrent.futures import ThreadPoolExecutor

import pytest

from appfl.logger import ClientAgentFileLogger, LoggerRegistry, ServerAgentFileLogger


def test_registry_enablement_and_schema(tmp_path):
    registry = LoggerRegistry()
    logger = ClientAgentFileLogger(file_dir=str(tmp_path), file_name="metrics", experiment_id="exp")
    registry.register("serialization", logger, columns=["seconds"])
    registry.write("missing", {"unexpected": 1})
    registry.disable("serialization")
    registry.write("serialization", {"seconds": 99})
    assert registry.get("serialization") is None
    registry.enable("serialization")
    registry.write("serialization", {"seconds": .5})
    with open(logger.csv_filepath) as stream:
        assert list(csv.reader(stream)) == [["seconds"], ["0.5"]]
    with pytest.raises(ValueError, match="already registered"):
        registry.register("serialization", logger, columns=["seconds"])
    with pytest.raises(ValueError, match="own logger"):
        registry.register("other", logger, columns=["seconds"])
    logger.titles = ["changed"]
    with pytest.raises(ValueError, match="schema changed"):
        registry.write("serialization", {"seconds": 1})


@pytest.mark.parametrize("logger_cls", [ClientAgentFileLogger, ServerAgentFileLogger])
def test_purposes_have_independent_text_files(tmp_path, logger_cls):
    first = logger_cls(file_dir=str(tmp_path), file_name="first", experiment_id="exp")
    second = logger_cls(file_dir=str(tmp_path), file_name="second", experiment_id="exp")
    assert first.logger is not second.logger
    first.info("FIRST_ONLY")
    second.info("SECOND_ONLY")
    first_text = next(tmp_path.glob("first*.txt")).read_text()
    second_text = next(tmp_path.glob("second*.txt")).read_text()
    assert "FIRST_ONLY" in first_text and "SECOND_ONLY" not in first_text
    assert "SECOND_ONLY" in second_text and "FIRST_ONLY" not in second_text


def test_concurrent_csv_rows_have_one_header(tmp_path):
    logger = ServerAgentFileLogger(file_dir=str(tmp_path), file_name="concurrent", experiment_id="exp")
    logger.set_title(["request"])
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: logger.log_content({"request": i}), range(40)))
    with open(logger.csv_filepath) as stream:
        rows = list(csv.reader(stream))
    assert rows[0] == ["request"]
    assert len(rows) == 41
    assert sorted(int(row[0]) for row in rows[1:]) == list(range(40))


def test_first_write_fixes_schema_and_legacy_adapter(tmp_path):
    registry = LoggerRegistry()
    logger = ServerAgentFileLogger(file_dir=str(tmp_path), file_name="lazy", experiment_id="exp")
    registry.register_legacy(compression=logger, communication=None)
    registry.write("compression", {"seconds": .1})
    assert logger.titles == ["seconds"]
    registry.disable("compression")
    registry.register_legacy(compression=logger)
    assert registry.get("compression") is None
    registry.enable("compression")
    with pytest.raises(ValueError, match="not defined"):
        registry.write("compression", {"other": 1})
