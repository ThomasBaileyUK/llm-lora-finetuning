"""Compare the base model and the fine-tuned adapter on held-out conversations.

Produces a Markdown report with:
  * perplexity on the assistant replies (lower = the model finds the target answers more likely)
  * side-by-side generations for each eval prompt, next to the reference answer

    python -m lora_ft.evaluate --adapter outputs/qwen3-0.6b-lora/adapter \
        --data data/recruiter_assistant/eval.jsonl --out outputs/qwen3-0.6b-lora/eval_report.md
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from .chat import generate, generate_base, load_model, run_system_prompt
from .data import normalise, read_local
from .utils import chat_prompt, logger, setup_logging


def answer_nll(model, tokenizer, messages: list[dict]) -> tuple[float, int]:
    """Summed negative log-likelihood of the final assistant reply, and its token count."""
    import torch

    prompt = chat_prompt(tokenizer, messages[:-1])
    full = prompt + messages[-1]["content"] + (tokenizer.eos_token or "")
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    ids = tokenizer(full, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
    labels = ids.clone()
    labels[:, : len(p_ids)] = -100
    n = int((labels != -100).sum())
    with torch.no_grad():
        loss = model(input_ids=ids, labels=labels).loss.item()
    return loss * n, n


def perplexity(model, tokenizer, convs, use_adapter: bool) -> float:
    tot, count = 0.0, 0
    for c in convs:
        if use_adapter or not hasattr(model, "disable_adapter"):
            nll, n = answer_nll(model, tokenizer, c)
        else:
            with model.disable_adapter():
                nll, n = answer_nll(model, tokenizer, c)
        tot, count = tot + nll, count + n
    return math.exp(tot / max(count, 1))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Base vs fine-tuned evaluation report")
    p.add_argument("--adapter", required=True)
    p.add_argument("--data", required=True, help="Local .jsonl/.json/.csv of held-out conversations")
    p.add_argument("--out", help="Where to write the Markdown report (default: next to the adapter)")
    p.add_argument("--limit", type=int, default=20, help="Max examples to generate for")
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument("--quantization", default="none", choices=["none", "4bit", "8bit"])
    args = p.parse_args(argv)
    setup_logging()

    system = run_system_prompt(args.adapter)  # evaluate with the same system prompt used in training
    convs = [r["messages"] for r in normalise(read_local(args.data), system, source=args.data)][: args.limit]
    model, tok, _ = load_model(args.adapter, quantization=args.quantization)

    logger.info("Scoring %d conversations…", len(convs))
    ppl_base = perplexity(model, tok, convs, use_adapter=False)
    ppl_tuned = perplexity(model, tok, convs, use_adapter=True)

    kw = {"max_new_tokens": args.max_new_tokens, "temperature": 0}
    lines = [
        "# Evaluation report",
        "",
        f"Adapter: `{args.adapter}` · data: `{args.data}` · {len(convs)} examples",
        "",
        "| | Base model | Fine-tuned | Change |",
        "|---|---:|---:|---:|",
        f"| Perplexity on reference answers (lower is better) | {ppl_base:.2f} | {ppl_tuned:.2f} | {100 * (ppl_tuned - ppl_base) / ppl_base:+.1f}% |",
        "",
    ]
    for i, c in enumerate(convs, 1):
        logger.info("Generating %d/%d", i, len(convs))
        ctx = c[:-1]
        q = ctx[-1]["content"]
        lines += [
            f"## {i}. {q[:90]}{'…' if len(q) > 90 else ''}",
            "",
            "**Base model**",
            "",
            generate_base(model, tok, ctx, **kw),
            "",
            "**Fine-tuned**",
            "",
            generate(model, tok, ctx, **kw),
            "",
            "<details><summary>Reference answer</summary>",
            "",
            c[-1]["content"],
            "",
            "</details>",
            "",
        ]
    out = Path(args.out or Path(args.adapter).parent / "eval_report.md")
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Perplexity base %.2f → fine-tuned %.2f. Report: %s", ppl_base, ppl_tuned, out)


if __name__ == "__main__":
    main()
