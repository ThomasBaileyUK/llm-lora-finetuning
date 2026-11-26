"""Chat with your fine-tuned adapter (optionally side-by-side with the base model).

    python -m lora_ft.chat --adapter outputs/qwen3-0.6b-lora/adapter
    python -m lora_ft.chat --adapter outputs/qwen3-0.6b-lora/adapter --compare --prompt "What's an FDE?"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .utils import chat_prompt, detect_precision, load_tokenizer, setup_logging


def base_model_name(adapter_dir: str) -> str:
    cfg = json.loads((Path(adapter_dir) / "adapter_config.json").read_text(encoding="utf-8"))
    return cfg["base_model_name_or_path"]


def run_system_prompt(adapter: str | None) -> str | None:
    """The system prompt the adapter was trained with (saved next to it by lora_ft.train)."""
    if not adapter:
        return None
    path = Path(adapter).parent / "run_config.yaml"
    if not path.exists():
        return None
    import yaml

    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return (cfg.get("data") or {}).get("system_prompt")


def load_model(adapter: str | None, base: str | None = None, quantization: str = "none", dtype: str = "auto"):
    """Load base model (+ LoRA adapter). Returns (model, tokenizer, precision)."""
    from transformers import AutoModelForCausalLM

    from .utils import quantization_config

    prec = detect_precision(dtype)
    base = base or base_model_name(adapter)
    kw = {"dtype": prec.torch_dtype}
    q = quantization_config(quantization, prec)
    if q is not None:
        kw["quantization_config"] = q
    if prec.device == "cuda":
        kw["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(base, **kw)
    if prec.device == "mps":
        model.to("mps")
    tokenizer = load_tokenizer(adapter or base)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer, prec


def generate(model, tokenizer, messages: list[dict], max_new_tokens: int = 384, temperature: float = 0.7) -> str:
    import torch

    text = chat_prompt(tokenizer, messages)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=0.9 if temperature > 0 else None,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def generate_base(model, tokenizer, messages, **kw) -> str:
    """Same model with the LoRA adapter switched off = the original base model."""
    if hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            return generate(model, tokenizer, messages, **kw)
    return generate(model, tokenizer, messages, **kw)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Chat with a LoRA fine-tuned model")
    p.add_argument("--adapter", help="Path (or Hub id) of the trained adapter")
    p.add_argument("--base", help="Base model (defaults to the one recorded in the adapter)")
    p.add_argument("--prompt", help="Ask a single question instead of an interactive chat")
    p.add_argument("--system", help="Optional system prompt")
    p.add_argument("--compare", action="store_true", help="Also show the base model's answer")
    p.add_argument("--quantization", default="none", choices=["none", "4bit", "8bit"])
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--temperature", type=float, default=0.7)
    args = p.parse_args(argv)
    if not args.adapter and not args.base:
        p.error("give --adapter (and/or --base)")
    setup_logging()

    model, tok, _ = load_model(args.adapter, args.base, args.quantization)
    kw = {"max_new_tokens": args.max_new_tokens, "temperature": args.temperature}
    system = args.system or run_system_prompt(args.adapter)
    history: list[dict] = [{"role": "system", "content": system}] if system else []

    def ask(q: str) -> None:
        history.append({"role": "user", "content": q})
        if args.compare and args.adapter:
            print("\n── base model ──────────────────────────────")
            print(generate_base(model, tok, history, **kw))
            print("\n── fine-tuned ──────────────────────────────")
        answer = generate(model, tok, history, **kw)
        print(answer, "\n")
        history.append({"role": "assistant", "content": answer})

    if args.prompt:
        ask(args.prompt)
        return
    print("Chat started. Type 'reset' to clear history, 'exit' to quit.\n")
    while True:
        try:
            q = input("you › ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit"}:
            break
        if q.lower() == "reset":
            del history[int(bool(system)):]
            continue
        if q:
            ask(q)


if __name__ == "__main__":
    main()
