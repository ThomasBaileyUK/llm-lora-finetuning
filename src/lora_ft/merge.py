"""Merge a LoRA adapter into its base model to get a standalone model.

The merged model needs no PEFT at inference time and can be served with vLLM, TGI,
`transformers serve`, or converted to GGUF for llama.cpp / Ollama.

    python -m lora_ft.merge --adapter outputs/qwen3-0.6b-lora/adapter --out outputs/qwen3-0.6b-merged
"""

from __future__ import annotations

import argparse

from .chat import base_model_name
from .utils import detect_precision, load_tokenizer, logger, setup_logging


def merge(adapter: str, out: str, base: str | None = None, dtype: str = "auto", push_to: str | None = None) -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    prec = detect_precision(dtype)
    base = base or base_model_name(adapter)
    # Always merge into an un-quantised base: merging into 4-bit weights loses precision.
    logger.info("Loading %s in %s", base, prec.dtype_name)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=prec.torch_dtype)
    model = PeftModel.from_pretrained(model, adapter)
    model = model.merge_and_unload()
    model.save_pretrained(out)
    load_tokenizer(adapter).save_pretrained(out)
    logger.info("Merged model saved to %s", out)
    if push_to:
        model.push_to_hub(push_to)
        load_tokenizer(adapter).push_to_hub(push_to)
        logger.info("Pushed to https://huggingface.co/%s", push_to)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Merge a LoRA adapter into the base model")
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--base")
    p.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    p.add_argument("--push-to", help="Optional Hugging Face repo id, e.g. your-name/my-merged-model")
    args = p.parse_args(argv)
    setup_logging()
    merge(args.adapter, args.out, args.base, args.dtype, args.push_to)


if __name__ == "__main__":
    main()
