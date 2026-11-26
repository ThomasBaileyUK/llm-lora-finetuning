"""Checks the arguments we hand to TRL / PEFT without needing torch installed.

Fake `trl`, `peft` and `torch` modules record the kwargs; we assert they're all real
parameter names of SFTConfig (TRL 1.x) and LoraConfig, so a typo can't slip through.
"""

import sys
import types

import pytest

from lora_ft.config import load_config
from lora_ft.utils import Precision

# Parameter names from the TRL 1.x SFTConfig / PEFT LoraConfig signatures that we rely on.
SFT_PARAMS = {
    "output_dir", "num_train_epochs", "max_steps", "learning_rate", "lr_scheduler_type", "warmup_steps",
    "weight_decay", "per_device_train_batch_size", "per_device_eval_batch_size", "gradient_accumulation_steps",
    "max_length", "packing", "assistant_only_loss", "gradient_checkpointing", "bf16", "fp16", "use_cpu",
    "logging_steps", "eval_strategy", "eval_steps", "save_strategy", "save_steps", "save_total_limit", "seed",
    "report_to", "push_to_hub", "hub_model_id", "model_init_kwargs", "trust_remote_code",
}
LORA_PARAMS = {"r", "lora_alpha", "lora_dropout", "target_modules", "bias", "use_rslora", "use_dora", "task_type"}


class Recorder:
    def __init__(self, **kw):
        self.kw = kw


@pytest.fixture
def fakes(monkeypatch):
    trl = types.ModuleType("trl")
    trl.SFTConfig = Recorder
    peft = types.ModuleType("peft")
    peft.LoraConfig = Recorder
    torch = types.ModuleType("torch")
    torch.bfloat16, torch.float16, torch.float32 = "bf16", "fp16", "fp32"
    for name, mod in {"trl": trl, "peft": peft, "torch": torch}.items():
        monkeypatch.setitem(sys.modules, name, mod)


@pytest.mark.parametrize(
    "prec",
    [Precision("cuda", "bfloat16", True, False), Precision("cuda", "float16", False, True), Precision("cpu", "float32", False, False)],
    ids=["ampere", "t4", "cpu"],
)
def test_sft_config_kwargs(fakes, prec):
    from lora_ft.train import build_sft_config

    cfg = load_config("configs/qwen3_0.6b_lora.yaml")
    args = build_sft_config(cfg, prec, has_eval=True).kw
    assert set(args) <= SFT_PARAMS, set(args) - SFT_PARAMS
    assert (args["bf16"], args["fp16"]) == (prec.bf16, prec.fp16)
    assert args["use_cpu"] is (prec.device == "cpu")
    assert args["eval_strategy"] == "steps"
    assert args["model_init_kwargs"]["dtype"] == {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}[prec.dtype_name]
    assert "trust_remote_code" not in args["model_init_kwargs"]


def test_eval_disabled_without_eval_set(fakes):
    from lora_ft.train import build_sft_config

    cfg = load_config("configs/qwen3_0.6b_lora.yaml")
    args = build_sft_config(cfg, Precision("cpu", "float32", False, False), has_eval=False).kw
    assert args["eval_strategy"] == "no" and args["eval_steps"] is None


def test_lora_config_kwargs(fakes):
    from lora_ft.train import build_peft_config

    cfg = load_config("configs/qwen3_4b_qlora.yaml", ["lora.r=8"])
    kw = build_peft_config(cfg).kw
    assert set(kw) <= LORA_PARAMS
    assert kw["r"] == 8 and kw["task_type"] == "CAUSAL_LM" and kw["target_modules"] == "all-linear"


@pytest.mark.parametrize("capability, expected", [((7, 5), "float16"), ((7, 0), "float16"), ((8, 0), "bfloat16"), ((8, 9), "bfloat16"), ((9, 0), "bfloat16")])
def test_precision_detection_uses_compute_capability(monkeypatch, capability, expected):
    """T4 (7.5) must get fp16 even though PyTorch reports bf16 as 'supported' via emulation."""
    torch = types.ModuleType("torch")
    torch.bfloat16, torch.float16, torch.float32 = "bf16", "fp16", "fp32"
    torch.cuda = types.SimpleNamespace(is_available=lambda: True, get_device_capability=lambda: capability, is_bf16_supported=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", torch)
    from lora_ft.utils import detect_precision

    prec = detect_precision("auto")
    assert prec.dtype_name == expected
    assert (prec.bf16, prec.fp16) == (expected == "bfloat16", expected == "float16")
    assert detect_precision("bfloat16").dtype_name == expected  # an explicit bf16 request also falls back on a T4


def test_upcast_trainable_params(monkeypatch):
    """Only trainable non-fp32 tensors are converted (the frozen 4-bit base must be left alone)."""
    torch = types.ModuleType("torch")
    torch.float32, torch.bfloat16 = "fp32", "bf16"
    monkeypatch.setitem(sys.modules, "torch", torch)

    class Data:
        def __init__(self, dtype):
            self.dtype = dtype

        def float(self):
            return Data("fp32")

    class Param:
        def __init__(self, dtype, requires_grad):
            self._data, self.requires_grad = Data(dtype), requires_grad

        @property
        def dtype(self):
            return self._data.dtype

        @property
        def data(self):
            return self._data

        @data.setter
        def data(self, v):
            self._data = v

    params = [Param("bf16", True), Param("fp32", True), Param("bf16", False), Param("uint8", False)]
    model = types.SimpleNamespace(parameters=lambda: iter(params))
    from lora_ft.utils import upcast_trainable_params

    assert upcast_trainable_params(model) == 1
    assert [p.dtype for p in params] == ["fp32", "fp32", "bf16", "uint8"]
