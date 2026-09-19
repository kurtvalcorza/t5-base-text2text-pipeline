"""Offline tests for the public validation and evaluation stage helpers (DAT24 / EVAL21)."""

from __future__ import annotations

import pytest

from t5_base_text2text_pipeline import (
    DECISION_RULE,
    DEFAULT_MAX_NEW_TOKENS,
    INPUT_SCHEMA,
    MAX_INPUT_TOKENS,
    MAX_NEW_TOKENS,
    MAX_NUM_BEAMS,
    MAX_TEXT_CHARS,
    MODEL_ID,
    MODEL_REVISION,
    TASK_PREFIXES,
    evaluation_report,
    known_prefix,
    validate_inputs,
)

TRANSLATE = "translate English to German: The house is wonderful."
SUMMARIZE = "summarize: A passage worth condensing, written out at some length."


def _result(generated_tokens: int = 7, num_beams: int = 1) -> dict:
    return {
        "text": "Das Haus ist wunderbar.",
        "generated_tokens": generated_tokens,
        "input_tokens": 11,
        "stopped_by": "eos",
        "known_prefix": "translate English to German: ",
        "generation": {
            "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
            "num_beams": num_beams,
            "do_sample": False,
            "decision_rule": DECISION_RULE,
        },
    }


def test_validate_inputs_returns_manifest_with_schema_and_identity() -> None:
    manifest = validate_inputs([TRANSLATE, SUMMARIZE], names=["input00", "input01"])
    assert manifest["verdict"] == "accepted"
    assert manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["text_chars"] == [1, MAX_TEXT_CHARS]
    assert manifest["schema"]["input_tokens"] == [1, MAX_INPUT_TOKENS]
    assert manifest["schema"]["max_new_tokens"] == [1, MAX_NEW_TOKENS]
    assert manifest["schema"]["num_beams"] == [1, MAX_NUM_BEAMS]
    assert manifest["schema"]["task_prefixes"] == list(TASK_PREFIXES)
    assert manifest["schema"]["decision_rule"] == DECISION_RULE
    assert manifest["inputs"] == [
        {"id": "input00", "chars": len(TRANSLATE), "known_prefix": "translate English to German: "},
        {"id": "input01", "chars": len(SUMMARIZE), "known_prefix": "summarize: "},
    ]
    assert (manifest["max_new_tokens"], manifest["num_beams"]) == (DEFAULT_MAX_NEW_TOKENS, 1)
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_validate_inputs_default_ids_and_unknown_prefix_is_recorded_not_rejected() -> None:
    manifest = validate_inputs(["no trained prefix here"], max_new_tokens=8, num_beams=4)
    assert [entry["id"] for entry in manifest["inputs"]] == ["input00"]
    assert manifest["inputs"][0]["known_prefix"] is None
    assert manifest["verdict"] == "accepted"
    assert (manifest["max_new_tokens"], manifest["num_beams"]) == (8, 4)
    assert known_prefix(TRANSLATE) == "translate English to German: "


def test_validate_inputs_rejects_like_generate() -> None:
    with pytest.raises(TypeError, match="not a single string"):
        validate_inputs(TRANSLATE)
    with pytest.raises(ValueError, match="at least one item"):
        validate_inputs([])
    with pytest.raises(TypeError, match="text must be str"):
        validate_inputs([None])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="text is empty"):
        validate_inputs(["   "])
    with pytest.raises(ValueError, match="MAX_TEXT_CHARS"):
        validate_inputs(["x" * (MAX_TEXT_CHARS + 1)])
    with pytest.raises(TypeError, match="max_new_tokens must be an int"):
        validate_inputs([TRANSLATE], max_new_tokens=True)
    with pytest.raises(ValueError, match="num_beams must be between"):
        validate_inputs([TRANSLATE], num_beams=MAX_NUM_BEAMS + 1)
    with pytest.raises(ValueError, match="names must have one entry per text"):
        validate_inputs([TRANSLATE], names=["a", "b"])


def test_evaluation_report_is_not_measurable_without_references() -> None:
    report = evaluation_report(_result())
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == []
    assert report["baselines"] == []
    assert report["n_generated_tokens"] == 7
    assert report["sample_kind"] == "synthetic"
    assert "no reference output" in report["reason"]
    assert "ROUGE-1/2/L" in report["needs"]
    assert DECISION_RULE in report["score_semantics"]
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_evaluation_report_scores_one_item_as_sample_sanity() -> None:
    result = {**_result(12, num_beams=4), "text": "Das Haus ist wunderbar."}
    report = evaluation_report(result, ["Das Haus ist wunderbar.", "other"], sample_kind="BYOD upload")
    assert report["verdict"] == "sample-sanity"
    assert {m["id"]: m["value"] for m in report["metrics"]} == {
        "rouge1": 100.0,
        "rouge2": 100.0,
        "rougeL": 100.0,
    }
    assert report["sample_kind"] == "BYOD upload"
    assert report["n_generated_tokens"] == 12
    assert "single item" in report["reason"]
    with pytest.raises(ValueError, match="at least one non-empty"):
        evaluation_report(result, [" "])
