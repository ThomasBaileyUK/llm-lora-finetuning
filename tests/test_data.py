import json
from pathlib import Path

import pytest

from lora_ft.data import DataFormatError, detect_format, normalise, read_local, to_messages

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "recruiter_assistant"


@pytest.mark.parametrize(
    "row, fmt",
    [
        ({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}, "messages"),
        ({"conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "hello"}]}, "sharegpt"),
        ({"instruction": "Say hi", "input": "", "output": "hi"}, "alpaca"),
        ({"prompt": "Q", "completion": "A"}, "prompt_completion"),
        ({"prompt": "Q", "response": "A"}, "prompt_completion"),
        ({"question": "Q", "answer": "A"}, "qa"),
    ],
)
def test_formats_are_detected_and_normalised(row, fmt):
    assert detect_format(row) == fmt
    msgs = to_messages(row)["messages"]
    assert msgs[0]["role"] == "user"
    assert msgs[-1]["role"] == "assistant"


def test_alpaca_input_is_appended():
    msgs = to_messages({"instruction": "Summarise", "input": "Some text", "output": "Short"})["messages"]
    assert msgs[0]["content"] == "Summarise\n\nSome text"


def test_conversational_prompt_completion():
    row = {"prompt": [{"role": "user", "content": "Q"}], "completion": [{"role": "assistant", "content": "A"}]}
    assert to_messages(row)["messages"] == [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]


def test_system_prompt_added_once():
    row = {"question": "Q", "answer": "A"}
    msgs = to_messages(row, system_prompt="Be brief")["messages"]
    assert msgs[0] == {"role": "system", "content": "Be brief"}
    row2 = {"messages": [{"role": "system", "content": "Own"}, {"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}
    assert to_messages(row2, system_prompt="Be brief")["messages"][0]["content"] == "Own"


def test_multi_turn_ok():
    row = {"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}, {"role": "assistant", "content": "d"}]}
    assert len(to_messages(row)["messages"]) == 4


@pytest.mark.parametrize(
    "row, msg",
    [
        ({"foo": "bar"}, "recognise"),
        ({"messages": []}, "empty"),
        ({"messages": [{"role": "user", "content": "only a question"}]}, "end with an assistant"),
        ({"messages": [{"role": "assistant", "content": "x"}, {"role": "user", "content": "y"}]}, "start with a user"),
        ({"messages": [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}, {"role": "assistant", "content": "c"}]}, "alternate"),
        ({"messages": [{"role": "wizard", "content": "a"}]}, "Unknown role"),
        ({"messages": [{"role": "user", "content": " "}, {"role": "assistant", "content": "c"}]}, "empty"),
    ],
)
def test_bad_rows_raise_helpful_errors(row, msg):
    with pytest.raises(DataFormatError, match=msg):
        to_messages(row)


def test_normalise_reports_row_number():
    with pytest.raises(DataFormatError, match="row 1"):
        normalise([{"question": "Q", "answer": "A"}, {"nope": 1}], source="x.jsonl")


def test_read_local_jsonl_json_csv(tmp_path):
    rows = [{"question": "Q1", "answer": "A1"}, {"question": "Q2", "answer": "A2"}]
    (tmp_path / "d.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n\n")
    (tmp_path / "d.json").write_text(json.dumps(rows))
    (tmp_path / "d.csv").write_text("question,answer\nQ1,A1\nQ2,A2\n")
    for name in ["d.jsonl", "d.json", "d.csv"]:
        assert len(normalise(read_local(tmp_path / name))) == 2


def test_read_local_bad_json_line(tmp_path):
    (tmp_path / "bad.jsonl").write_text('{"question": "Q", "answer": "A"}\n{oops}\n')
    with pytest.raises(DataFormatError, match=":2"):
        read_local(tmp_path / "bad.jsonl")


@pytest.mark.parametrize("name", ["train.jsonl", "eval.jsonl"])
def test_sample_dataset_is_valid(name):
    rows = normalise(read_local(SAMPLE / name), source=name)
    assert len(rows) >= 5
    for r in rows:
        assert len(r["messages"][-1]["content"]) > 40


def test_no_eval_leakage():
    train_q = {r["messages"][0]["content"] for r in normalise(read_local(SAMPLE / "train.jsonl"))}
    eval_q = {r["messages"][0]["content"] for r in normalise(read_local(SAMPLE / "eval.jsonl"))}
    assert not train_q & eval_q
