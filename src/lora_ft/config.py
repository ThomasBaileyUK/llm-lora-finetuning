"""Typed run configuration loaded from YAML, with dotted CLI overrides.

Kept free of heavy imports (torch/transformers) so it can be unit-tested anywhere.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen3-0.6B"
    # "auto" picks bf16 on GPUs that support it, fp16 on older GPUs (e.g. Colab T4), fp32 on CPU.
    dtype: str = "auto"
    # "none" = LoRA in 16-bit, "4bit" = QLoRA (NF4), "8bit" = LoRA on an 8-bit base.
    quantization: str = "none"
    attn_implementation: str | None = None
    trust_remote_code: bool = False
    device_map: str | None = None


@dataclass
class LoraConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    # "all-linear" targets every linear layer except the output head (best quality per parameter).
    target_modules: str | list[str] = "all-linear"
    bias: str = "none"
    use_rslora: bool = False
    use_dora: bool = False


@dataclass
class DataConfig:
    # A local .jsonl/.json/.csv file, or a Hugging Face dataset id.
    train: str = "data/recruiter_assistant/train.jsonl"
    eval: str | None = "data/recruiter_assistant/eval.jsonl"
    hf_config: str | None = None
    train_split: str = "train"
    eval_split: str | None = None
    # If no eval set is given, hold out this fraction of train (0 disables).
    eval_fraction: float = 0.0
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    # Prepended as a system message to every example that doesn't already have one.
    system_prompt: str | None = None
    seed: int = 42


@dataclass
class TrainConfig:
    output_dir: str = "outputs/run"
    num_train_epochs: float = 3.0
    max_steps: int = -1
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    # int = number of steps; float < 1 = fraction of total steps.
    warmup_steps: float = 0.03
    weight_decay: float = 0.0
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_length: int = 1024
    packing: bool = False
    # Only learn from the assistant's replies (needs a chat template with generation markers;
    # TRL patches this automatically for common families such as Qwen3).
    assistant_only_loss: bool = True
    gradient_checkpointing: bool = True
    logging_steps: int = 5
    eval_steps: int | None = 25
    save_steps: int = 100
    save_total_limit: int = 2
    seed: int = 42
    report_to: str = "none"  # "none", "tensorboard", "wandb", "trackio"
    push_to_hub: bool = False
    hub_model_id: str | None = None


@dataclass
class RunConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


VALID_QUANT = {"none", "4bit", "8bit"}
VALID_DTYPE = {"auto", "bfloat16", "float16", "float32"}


def _build(cls, data: dict[str, Any], path: str):
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"Unknown key(s) in '{path or 'config'}': {sorted(unknown)}. Valid: {sorted(known)}")
    kwargs = {}
    for name, value in data.items():
        default = getattr(cls(), name)
        if is_dataclass(default):
            if not isinstance(value, dict):
                raise ValueError(f"'{path + '.' if path else ''}{name}' must be a mapping")
            kwargs[name] = _build(type(default), value, f"{path + '.' if path else ''}{name}")
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _coerce(raw: str) -> Any:
    """Parse a CLI override value using YAML rules (numbers, bools, null, lists)."""
    return yaml.safe_load(raw)


def apply_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply `section.key=value` overrides, e.g. `train.max_steps=20`."""
    out = copy.deepcopy(data)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must look like section.key=value")
        key, raw = item.split("=", 1)
        parts = key.strip().split(".")
        node = out
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ValueError(f"Cannot set '{key}': '{p}' is not a section")
        node[parts[-1]] = _coerce(raw)
    return out


def validate(cfg: RunConfig) -> RunConfig:
    if cfg.model.quantization not in VALID_QUANT:
        raise ValueError(f"model.quantization must be one of {sorted(VALID_QUANT)}")
    if cfg.model.dtype not in VALID_DTYPE:
        raise ValueError(f"model.dtype must be one of {sorted(VALID_DTYPE)}")
    if cfg.lora.r <= 0 or cfg.lora.alpha <= 0:
        raise ValueError("lora.r and lora.alpha must be positive")
    if not 0 <= cfg.lora.dropout < 1:
        raise ValueError("lora.dropout must be in [0, 1)")
    if not 0 <= cfg.data.eval_fraction < 1:
        raise ValueError("data.eval_fraction must be in [0, 1)")
    if cfg.train.max_steps <= 0 and cfg.train.num_train_epochs <= 0:
        raise ValueError("Set train.num_train_epochs > 0 or train.max_steps > 0")
    if cfg.train.push_to_hub and not cfg.train.hub_model_id:
        raise ValueError("train.push_to_hub needs train.hub_model_id (e.g. your-username/my-lora)")
    return cfg


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> RunConfig:
    data: dict[str, Any] = {}
    if path:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    data = apply_overrides(data, overrides or [])
    return validate(_build(RunConfig, data, ""))


def save_config(cfg: RunConfig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
