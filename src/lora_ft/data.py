"""Dataset loading and normalisation.

Every supported input format is converted to the conversational format TRL expects:

    {"messages": [{"role": "system"|"user"|"assistant", "content": "..."}, ...]}

Supported inputs (auto-detected per row):
  * messages      {"messages": [{"role", "content"}, ...]}
  * ShareGPT      {"conversations": [{"from": "human"|"gpt"|"system", "value": "..."}, ...]}
  * Alpaca        {"instruction", "input"?, "output"}
  * prompt/resp   {"prompt", "completion"|"response"}  (strings or message lists)
  * question/ans  {"question", "answer"}

The pure-Python helpers here don't import `datasets`, so they're easy to unit test.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROLE_MAP = {
    "system": "system",
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "bot": "assistant",
    "model": "assistant",
}


class DataFormatError(ValueError):
    pass


def _as_messages(value: Any, default_role: str) -> list[dict[str, str]]:
    if isinstance(value, str):
        return [{"role": default_role, "content": value}]
    if isinstance(value, list):
        return [_clean_msg(m) for m in value]
    raise DataFormatError(f"Expected a string or list of messages, got {type(value).__name__}")


def _clean_msg(m: Any) -> dict[str, str]:
    if not isinstance(m, dict):
        raise DataFormatError("Each message must be an object")
    role = m.get("role", m.get("from"))
    content = m.get("content", m.get("value"))
    if role is None or content is None:
        raise DataFormatError(f"Message needs role/content (or from/value): {m}")
    role = ROLE_MAP.get(str(role).lower())
    if role is None:
        raise DataFormatError(f"Unknown role '{m.get('role', m.get('from'))}'")
    return {"role": role, "content": str(content)}


def detect_format(row: dict[str, Any]) -> str:
    if "messages" in row:
        return "messages"
    if "conversations" in row:
        return "sharegpt"
    if "instruction" in row and "output" in row:
        return "alpaca"
    if "prompt" in row and ("completion" in row or "response" in row):
        return "prompt_completion"
    if "question" in row and "answer" in row:
        return "qa"
    raise DataFormatError(f"Can't recognise row format from keys {sorted(row)}. See README → Data format.")


def to_messages(row: dict[str, Any], system_prompt: str | None = None) -> dict[str, list[dict[str, str]]]:
    fmt = detect_format(row)
    if fmt == "messages":
        msgs = [_clean_msg(m) for m in row["messages"]]
    elif fmt == "sharegpt":
        msgs = [_clean_msg(m) for m in row["conversations"]]
    elif fmt == "alpaca":
        instr = str(row["instruction"]).strip()
        extra = str(row.get("input") or "").strip()
        user = f"{instr}\n\n{extra}" if extra else instr
        msgs = [{"role": "user", "content": user}, {"role": "assistant", "content": str(row["output"])}]
    elif fmt == "prompt_completion":
        answer = row.get("completion", row.get("response"))
        msgs = _as_messages(row["prompt"], "user") + _as_messages(answer, "assistant")
    else:  # qa
        msgs = [{"role": "user", "content": str(row["question"])}, {"role": "assistant", "content": str(row["answer"])}]

    if system_prompt and not any(m["role"] == "system" for m in msgs):
        msgs = [{"role": "system", "content": system_prompt}] + msgs
    validate_messages(msgs)
    return {"messages": msgs}


def validate_messages(msgs: list[dict[str, str]]) -> None:
    if not msgs:
        raise DataFormatError("Conversation is empty")
    body = msgs[1:] if msgs[0]["role"] == "system" else msgs
    if any(m["role"] == "system" for m in body):
        raise DataFormatError("A system message is only allowed as the first message")
    if not body or body[0]["role"] != "user":
        raise DataFormatError("Conversation must start with a user message (after an optional system message)")
    if body[-1]["role"] != "assistant":
        raise DataFormatError("Conversation must end with an assistant message (that's what the model learns)")
    for a, b in zip(body, body[1:], strict=False):
        if a["role"] == b["role"]:
            raise DataFormatError("Roles must alternate user/assistant")
    if any(not m["content"].strip() for m in body):
        raise DataFormatError("Messages must not be empty")


def read_local(path: str | Path) -> list[dict[str, Any]]:
    """Read .jsonl, .json (list) or .csv into a list of dicts."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix == ".jsonl":
        rows = []
        with p.open(encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        raise DataFormatError(f"{p}:{i}: invalid JSON ({e.msg})") from e
        return rows
    if p.suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise DataFormatError(f"{p} must contain a JSON list of examples")
        return data
    if p.suffix == ".csv":
        import csv

        with p.open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    raise DataFormatError(f"Unsupported file type '{p.suffix}' (use .jsonl, .json or .csv)")


def normalise(rows: Iterable[dict[str, Any]], system_prompt: str | None = None, source: str = "data") -> list[dict]:
    out = []
    for i, row in enumerate(rows):
        try:
            out.append(to_messages(row, system_prompt))
        except DataFormatError as e:
            raise DataFormatError(f"{source} row {i}: {e}") from e
    return out


def is_local(spec: str) -> bool:
    return Path(spec).suffix in {".jsonl", ".json", ".csv"} or Path(spec).exists()


def load_datasets(cfg):
    """Return (train_dataset, eval_dataset | None) as `datasets.Dataset` objects in messages format."""
    from datasets import Dataset, load_dataset

    d = cfg.data

    def load(spec: str, split: str):
        if is_local(spec):
            return Dataset.from_list(normalise(read_local(spec), d.system_prompt, spec))
        ds = load_dataset(spec, d.hf_config, split=split)
        cols = ds.column_names
        ds = ds.map(lambda r: to_messages(r, d.system_prompt), remove_columns=cols, desc=f"Formatting {spec}")
        return ds

    train = load(d.train, d.train_split)
    evals = None
    if d.eval:
        evals = load(d.eval, d.eval_split or d.train_split)
    elif d.eval_split and not is_local(d.train):
        evals = load(d.train, d.eval_split)
    elif d.eval_fraction > 0:
        split = train.train_test_split(test_size=d.eval_fraction, seed=d.seed)
        train, evals = split["train"], split["test"]

    train = train.shuffle(seed=d.seed)
    if d.max_train_samples:
        train = train.select(range(min(d.max_train_samples, len(train))))
    if evals is not None and d.max_eval_samples:
        evals = evals.select(range(min(d.max_eval_samples, len(evals))))
    return train, evals
