"""Hardware detection, precision and model-loading helpers."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger("lora_ft")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S", level=level)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@dataclass
class Precision:
    device: str  # "cuda", "mps" or "cpu"
    dtype_name: str  # "bfloat16" | "float16" | "float32"
    bf16: bool  # trainer mixed precision flags
    fp16: bool

    @property
    def torch_dtype(self):
        import torch

        return getattr(torch, self.dtype_name)


def detect_precision(requested: str = "auto") -> Precision:
    """Pick sensible dtypes for the available hardware.

    * Ampere+ GPUs (A100, L4, RTX 30/40, H100): bfloat16
    * Older GPUs (Colab T4, V100): float16 — they have no bf16 support
    * Apple Silicon (MPS) / CPU: float32 (most stable)
    """
    import torch

    if torch.cuda.is_available():
        device = "cuda"
        # Native bf16 needs compute capability 8.0+ (Ampere). Don't use torch.cuda.is_bf16_supported():
        # recent PyTorch returns True on a T4 (7.5) because bf16 can be *emulated*, which is slow and fragile.
        bf16_ok = torch.cuda.get_device_capability()[0] >= 8
        name = {"auto": "bfloat16" if bf16_ok else "float16"}.get(requested, requested)
        if name == "bfloat16" and not bf16_ok:
            logger.warning("bfloat16 isn't supported on this GPU, falling back to float16")
            name = "float16"
        return Precision(device, name, bf16=name == "bfloat16", fp16=name == "float16")
    device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    name = "float32" if requested == "auto" else requested
    return Precision(device, name, bf16=False, fp16=False)


def quantization_config(kind: str, prec: Precision):
    """BitsAndBytes config for QLoRA (4-bit NF4) or 8-bit loading. Requires a CUDA GPU."""
    if kind == "none":
        return None
    if prec.device != "cuda":
        raise RuntimeError(f"{kind} quantization needs an NVIDIA GPU (bitsandbytes). Set model.quantization: none")
    from transformers import BitsAndBytesConfig

    if kind == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=prec.torch_dtype,
        )
    return BitsAndBytesConfig(load_in_8bit=True)


def model_init_kwargs(cfg, prec: Precision) -> dict:
    # trust_remote_code is passed separately via SFTConfig, which forwards it to from_pretrained.
    kw: dict = {"dtype": prec.torch_dtype}
    if cfg.model.attn_implementation:
        kw["attn_implementation"] = cfg.model.attn_implementation
    if cfg.model.device_map:
        kw["device_map"] = cfg.model.device_map
    return kw


def upcast_trainable_params(model) -> int:
    """Cast trainable (LoRA) weights to float32. Returns how many tensors were changed.

    fp16 mixed precision uses a gradient scaler that only accepts fp32/fp16 gradients, but QLoRA
    can create the adapter weights in bf16. Keeping the small adapter in fp32 is also more stable.
    """
    import torch

    changed = 0
    for p in model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
            changed += 1
    return changed


def count_parameters(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def human(n: float) -> str:
    for unit in ["", "K", "M", "B"]:
        if abs(n) < 1000:
            return f"{n:.1f}{unit}" if unit else f"{int(n)}"
        n /= 1000
    return f"{n:.1f}T"


def load_tokenizer(name_or_path: str, trust_remote_code: bool = False):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def chat_prompt(tokenizer, messages: list[dict]) -> str:
    """Render a conversation with the model's chat template, ready for the assistant to reply.

    `enable_thinking=False` switches off Qwen3's reasoning block; other templates ignore it.
    """
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
