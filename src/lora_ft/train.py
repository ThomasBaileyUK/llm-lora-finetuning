"""Fine-tune a causal LLM with LoRA / QLoRA using TRL's SFTTrainer.

    python -m lora_ft.train --config configs/qwen3_0.6b_lora.yaml
    python -m lora_ft.train --config configs/smoke_test.yaml --set train.max_steps=5
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from .config import RunConfig, load_config, save_config
from .utils import count_parameters, detect_precision, human, load_tokenizer, logger, model_init_kwargs, quantization_config, setup_logging


def build_peft_config(cfg: RunConfig):
    from peft import LoraConfig

    return LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        bias=cfg.lora.bias,
        use_rslora=cfg.lora.use_rslora,
        use_dora=cfg.lora.use_dora,
        task_type="CAUSAL_LM",
    )


def build_sft_config(cfg: RunConfig, prec, has_eval: bool):
    from trl import SFTConfig

    t = cfg.train
    eval_on = has_eval and t.eval_steps
    return SFTConfig(
        output_dir=t.output_dir,
        num_train_epochs=t.num_train_epochs,
        max_steps=t.max_steps,
        learning_rate=t.learning_rate,
        lr_scheduler_type=t.lr_scheduler_type,
        warmup_steps=t.warmup_steps,
        weight_decay=t.weight_decay,
        per_device_train_batch_size=t.per_device_train_batch_size,
        per_device_eval_batch_size=t.per_device_eval_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        max_length=t.max_length,
        packing=t.packing,
        assistant_only_loss=t.assistant_only_loss,
        gradient_checkpointing=t.gradient_checkpointing,
        # bf16 on Ampere+, fp16 on T4/V100, neither on CPU/MPS
        bf16=prec.bf16,
        fp16=prec.fp16,
        use_cpu=prec.device == "cpu",
        logging_steps=t.logging_steps,
        eval_strategy="steps" if eval_on else "no",
        eval_steps=t.eval_steps if eval_on else None,
        save_strategy="steps",
        save_steps=t.save_steps,
        save_total_limit=t.save_total_limit,
        seed=t.seed,
        report_to=t.report_to,
        push_to_hub=t.push_to_hub,
        hub_model_id=t.hub_model_id,
        model_init_kwargs=model_init_kwargs(cfg, prec),
        trust_remote_code=cfg.model.trust_remote_code,
    )


def train(cfg: RunConfig) -> dict:
    from trl import SFTTrainer

    from .data import load_datasets

    prec = detect_precision(cfg.model.dtype)
    logger.info("Device: %s | dtype: %s | quantization: %s", prec.device, prec.dtype_name, cfg.model.quantization)

    train_ds, eval_ds = load_datasets(cfg)
    logger.info("Train examples: %d | eval examples: %s", len(train_ds), len(eval_ds) if eval_ds is not None else "none")

    out = Path(cfg.train.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / "run_config.yaml")

    tokenizer = load_tokenizer(cfg.model.name, cfg.model.trust_remote_code)
    trainer = SFTTrainer(
        model=cfg.model.name,
        args=build_sft_config(cfg, prec, eval_ds is not None),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=build_peft_config(cfg),
        quantization_config=quantization_config(cfg.model.quantization, prec),
    )

    trainable, total = count_parameters(trainer.model)
    logger.info("Trainable params: %s / %s (%.3f%%) — that's the LoRA adapter", human(trainable), human(total), 100 * trainable / total)

    start = time.time()
    result = trainer.train()
    minutes = (time.time() - start) / 60

    metrics = dict(result.metrics)
    if eval_ds is not None:
        ev = trainer.evaluate()
        metrics.update(ev)
        if "eval_loss" in ev:
            metrics["eval_perplexity"] = math.exp(min(ev["eval_loss"], 20))

    adapter_dir = out / "adapter"
    trainer.save_model(str(adapter_dir))  # saves only the LoRA weights (a few MB) + adapter config
    tokenizer.save_pretrained(str(adapter_dir))

    summary = {
        "base_model": cfg.model.name,
        "quantization": cfg.model.quantization,
        "lora": {"r": cfg.lora.r, "alpha": cfg.lora.alpha, "target_modules": cfg.lora.target_modules},
        "trainable_params": trainable,
        "total_params": total,
        "train_examples": len(train_ds),
        "eval_examples": len(eval_ds) if eval_ds is not None else 0,
        "minutes": round(minutes, 2),
        "device": prec.device,
        "dtype": prec.dtype_name,
        "metrics": metrics,
        "log_history": trainer.state.log_history,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    _plot_loss(trainer.state.log_history, out / "loss.png")

    if cfg.train.push_to_hub:
        trainer.push_to_hub()
        logger.info("Pushed adapter to https://huggingface.co/%s", cfg.train.hub_model_id)

    logger.info("Done in %.1f min. Adapter saved to %s", minutes, adapter_dir)
    logger.info("Try it:  python -m lora_ft.chat --adapter %s --compare", adapter_dir)
    return summary


def _plot_loss(history: list[dict], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    tr = [(h["step"], h["loss"]) for h in history if "loss" in h and "step" in h]
    ev = [(h["step"], h["eval_loss"]) for h in history if "eval_loss" in h and "step" in h]
    if not tr:
        return
    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    ax.plot(*zip(*tr, strict=True), label="train loss", linewidth=2)
    if ev:
        ax.plot(*zip(*ev, strict=True), label="eval loss", linewidth=2, marker="o")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="LoRA / QLoRA supervised fine-tuning")
    p.add_argument("--config", "-c", required=True, help="Path to a YAML config in configs/")
    p.add_argument("--set", "-s", nargs="*", default=[], metavar="KEY=VALUE", help="Override config values, e.g. train.max_steps=20 lora.r=8")
    args = p.parse_args(argv)
    setup_logging()
    cfg = load_config(args.config, args.set)
    train(cfg)


if __name__ == "__main__":
    main()
