"""APPFL-compatible wrapper for a locally stored Llama causal language model."""

import torch
from torch import nn
from transformers import AutoModelForCausalLM


class Llama400M(nn.Module):
    """Load a local Hugging Face checkpoint and expose logits to APPFL.

    ``VanillaTrainer`` expects a tensor from ``forward`` and applies its configured
    loss itself. Hugging Face causal language models return a model-output object,
    so this small wrapper returns logits in ``(batch, vocab, sequence)`` order for
    use with :class:`torch.nn.CrossEntropyLoss`.
    """

    def __init__(
        self,
        model_dir: str = "./llama-400m-fresh",
        torch_dtype: str = "float32",
        local_files_only: bool = True,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        if not hasattr(torch, torch_dtype):
            raise ValueError(f"Unknown torch dtype: {torch_dtype}")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=getattr(torch, torch_dtype),
            local_files_only=local_files_only,
            low_cpu_mem_usage=True,
        )
        # KV caching is useful for generation but wastes memory during training.
        self.model.config.use_cache = False
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Dataset examples are fixed-length token blocks and contain no padding.
        attention_mask = torch.ones_like(input_ids)
        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits
        return logits.transpose(1, 2).contiguous()
