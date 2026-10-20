"""Text dataset and deterministic client partitioning for Llama fine-tuning."""

import gzip
import json
from pathlib import Path
from typing import Iterable, Optional, TextIO, Tuple

import torch
from torch.utils.data import Dataset, Subset
from transformers import AutoTokenizer


class CausalLanguageModelingDataset(Dataset):
    """Tokenize text into fixed-length next-token-prediction examples."""

    def __init__(self, texts: Iterable[str], tokenizer, sequence_length: int = 256):
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        token_ids = []
        separator = tokenizer.eos_token_id
        for text in texts:
            ids = tokenizer.encode(str(text), add_special_tokens=False)
            token_ids.extend(ids)
            if separator is not None:
                token_ids.append(separator)

        block_size = sequence_length + 1
        usable = len(token_ids) // block_size * block_size
        if usable == 0:
            raise ValueError(
                f"The dataset has too few tokens for sequence_length={sequence_length}"
            )
        self.blocks = torch.tensor(token_ids[:usable], dtype=torch.long).view(
            -1, block_size
        )

    def __len__(self) -> int:
        return self.blocks.size(0)

    def __getitem__(self, index: int):
        block = self.blocks[index]
        return block[:-1], block[1:]


def _open_text(path: Path) -> TextIO:
    """Open plain text or gzip-compressed text without extracting it."""
    if path.suffix.lower() == ".gz":
        with path.open("rb") as stream:
            header = stream.read(64)
        if header.startswith(b"version https://git-lfs.github.com/spec/"):
            raise RuntimeError(
                f"{path} is a Git LFS pointer, not the dataset shard. "
                "Install Git LFS and download the repository's LFS objects "
                "before running this example."
            )
        if not header.startswith(b"\x1f\x8b"):
            raise ValueError(
                f"{path} has a .gz extension but is not gzip-compressed "
                f"(first bytes: {header[:8]!r})"
            )
        return gzip.open(path, mode="rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def _record_text(record, text_field: str, location: str) -> str:
    if not isinstance(record, dict):
        raise TypeError(f"Expected a JSON object at {location}")
    if text_field not in record:
        raise KeyError(f"Missing '{text_field}' at {location}")
    return str(record[text_field])


def _read_json_lines(
    stream: TextIO, text_field: str, max_documents: Optional[int]
) -> list[str]:
    texts = []
    for line_number, line in enumerate(stream, start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        texts.append(_record_text(record, text_field, f"line {line_number}"))
        if max_documents is not None and len(texts) >= max_documents:
            break
    return texts


def _read_texts(
    data_path: str, text_field: str, max_documents: Optional[int] = None
) -> list[str]:
    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(f"Training data not found: {path.resolve()}")
    if max_documents is not None and max_documents < 1:
        raise ValueError("max_documents must be positive")

    suffixes = [suffix.lower() for suffix in path.suffixes]
    is_gzip = suffixes[-1:] == [".gz"]
    content_suffix = suffixes[-2] if is_gzip and len(suffixes) > 1 else path.suffix.lower()

    # C4 files named *.json.gz contain one JSON object per line.
    if content_suffix == ".jsonl" or (content_suffix == ".json" and is_gzip):
        with _open_text(path) as stream:
            return _read_json_lines(stream, text_field, max_documents)

    if content_suffix == ".json":
        with _open_text(path) as stream:
            content = json.load(stream)
        if isinstance(content, dict):
            records = content.get("data", [content])
        elif isinstance(content, list):
            records = content
        else:
            raise TypeError("JSON input must be an object or an array of objects")
        if max_documents is not None:
            records = records[:max_documents]
        return [
            _record_text(record, text_field, f"JSON record {index}")
            for index, record in enumerate(records)
        ]

    # Blank lines delimit documents in ordinary text files.
    with _open_text(path) as stream:
        texts = [part.strip() for part in stream.read().split("\n\n") if part.strip()]
    return texts[:max_documents] if max_documents is not None else texts


def get_llama400m_dataset(
    data_path: str,
    model_dir: str = "./llama-400m-fresh",
    num_clients: int = 2,
    client_id: int = 0,
    sequence_length: int = 256,
    text_field: str = "text",
    max_documents: Optional[int] = None,
    validation_fraction: float = 0.0,
    seed: int = 42,
) -> Tuple[Subset, Optional[Subset]]:
    """Load, tokenize, shuffle, and partition a local text/JSON dataset.

    Every client constructs the same seeded permutation and receives a disjoint
    strided shard. Plain or gzip-compressed JSON Lines, JSON documents, and text
    files are supported. A fraction of the client shard can optionally be held
    out.
    """
    if num_clients < 1 or not 0 <= client_id < num_clients:
        raise ValueError("client_id must be in [0, num_clients)")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, use_fast=True
    )
    dataset = CausalLanguageModelingDataset(
        _read_texts(data_path, text_field, max_documents), tokenizer, sequence_length
    )
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator).tolist()[
        client_id::num_clients
    ]
    if not indices:
        raise ValueError("This client received no examples; use more data or fewer clients")

    val_size = int(len(indices) * validation_fraction)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return Subset(dataset, train_indices), (
        Subset(dataset, val_indices) if val_indices else None
    )
