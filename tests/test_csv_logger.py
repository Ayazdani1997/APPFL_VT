import csv

import pytest

from appfl.logger import ClientAgentFileLogger, ServerAgentFileLogger


@pytest.mark.parametrize(
    ("logger_cls", "name_fragment"),
    [
        (ClientAgentFileLogger, "metrics_client_exp.csv"),
        (ServerAgentFileLogger, "metrics_Server_exp.csv"),
    ],
)
def test_structured_logs_are_written_to_csv(tmp_path, logger_cls, name_fragment):
    kwargs = {
        "file_dir": str(tmp_path),
        "file_name": "metrics",
        "experiment_id": "exp",
    }
    if logger_cls is ClientAgentFileLogger:
        kwargs["logging_id"] = "client"

    logger = logger_cls(**kwargs)
    logger.log_title(["Epoch", "Step", "Compute Time"])
    logger.log_content({"Epoch": 2, "Step": 3, "Compute Time": 0.125})

    assert logger.get_csv_filepath().endswith(name_fragment)
    with open(logger.get_csv_filepath(), newline="", encoding="utf-8") as csv_file:
        assert list(csv.reader(csv_file)) == [
            ["Epoch", "Step", "Compute Time"],
            ["2", "3", "0.125"],
        ]


def test_csv_logging_requires_titles(tmp_path):
    logger = ClientAgentFileLogger(
        logging_id="client",
        file_dir=str(tmp_path),
        file_name="metrics",
        experiment_id="exp",
    )

    with pytest.raises(ValueError, match="Titles must be set"):
        logger.log_content([1, 2])
