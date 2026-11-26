from pathlib import Path

import pytest

from lora_ft.config import RunConfig, apply_overrides, load_config, save_config

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = sorted((ROOT / "configs").glob("*.yaml"))


def test_configs_exist():
    assert CONFIGS, "expected example configs in configs/"


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_every_shipped_config_loads(path):
    cfg = load_config(path)
    assert isinstance(cfg, RunConfig)
    assert cfg.model.name
    assert cfg.train.output_dir.startswith("outputs/")


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_local_data_paths_exist(path):
    cfg = load_config(path)
    for spec in [cfg.data.train, cfg.data.eval]:
        if spec and spec.endswith((".jsonl", ".json", ".csv")):
            assert (ROOT / spec).exists(), spec


def test_defaults_without_file():
    cfg = load_config(None)
    assert cfg.lora.target_modules == "all-linear"
    assert cfg.model.quantization == "none"


def test_overrides_are_typed():
    cfg = load_config(CONFIGS[0], ["train.max_steps=3", "lora.r=4", "train.packing=true", "data.eval=null", "lora.target_modules=[q_proj, v_proj]"])
    assert cfg.train.max_steps == 3
    assert cfg.lora.r == 4
    assert cfg.train.packing is True
    assert cfg.data.eval is None
    assert cfg.lora.target_modules == ["q_proj", "v_proj"]


def test_override_creates_nested_section():
    out = apply_overrides({}, ["model.name=foo/bar"])
    assert out == {"model": {"name": "foo/bar"}}


@pytest.mark.parametrize(
    "override, msg",
    [
        ("model.quantization=2bit", "quantization"),
        ("lora.r=0", "positive"),
        ("lora.dropout=1.5", "dropout"),
        ("train.push_to_hub=true", "hub_model_id"),
        ("train.bogus=1", "Unknown key"),
    ],
)
def test_invalid_values_raise(override, msg):
    with pytest.raises(ValueError, match=msg):
        load_config(None, [override])


def test_bad_override_syntax():
    with pytest.raises(ValueError, match="section.key=value"):
        load_config(None, ["train.max_steps"])


def test_round_trip(tmp_path):
    cfg = load_config(CONFIGS[0], ["train.max_steps=7"])
    save_config(cfg, tmp_path / "c.yaml")
    again = load_config(tmp_path / "c.yaml")
    assert again == cfg
