import hashlib
import json
import re
from pathlib import Path

import pytest

from t5_base_text2text_pipeline import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_WEIGHTS_DIR,
    MAX_INPUT_TOKENS,
    MAX_NEW_TOKENS,
    MAX_NUM_BEAMS,
    MAX_TEXT_CHARS,
    MODEL_ID,
    MODEL_KEY,
    MODEL_LICENSE,
    MODEL_REVISION,
    TASK_PREFIXES,
    T5BaseText2TextPipeline,
    stage_missing_files,
    verify_snapshot,
)

HEX40 = re.compile(r"^[0-9a-f]{40}$")


def _fake_count(text: str) -> int:
    return len(text.split()) + 1  # words + EOS, stands in for the SentencePiece count


def _fake_runner(text: str, max_new_tokens: int, num_beams: int) -> tuple[str, int, str]:
    words = text.split()[:max_new_tokens]
    return (
        " ".join(w.upper() for w in words),
        len(words),
        "eos" if len(words) < max_new_tokens else "max_new_tokens",
    )


def _pipeline() -> T5BaseText2TextPipeline:
    return T5BaseText2TextPipeline(_fake_runner, _fake_count, "cpu", "injected")


def _write_snapshot(root: Path, payload: bytes = b"weights") -> Path:
    (root / "model.safetensors").write_bytes(payload)
    manifest = {
        "modelKey": MODEL_KEY,
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {
                "path": "model.safetensors",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
    }
    path = root / "dimer-base-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_identity_constants_are_40_hex_and_named():
    assert HEX40.match(MODEL_REVISION)
    assert MODEL_ID == "google-t5/t5-base"
    assert MODEL_LICENSE == "apache-2.0"
    assert DEFAULT_WEIGHTS_DIR.name == MODEL_KEY
    assert DEFAULT_WEIGHTS_DIR.parent.name == "weights"
    assert MAX_INPUT_TOKENS == 512
    assert 1 <= DEFAULT_MAX_NEW_TOKENS <= MAX_NEW_TOKENS


def test_identity_matches_local_manifest_when_present():
    manifest_path = DEFAULT_WEIGHTS_DIR / "dimer-base-manifest.json"
    if not manifest_path.is_file():
        pytest.skip("local snapshot manifest not staged")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["modelId"] == MODEL_ID
    assert manifest["revision"] == MODEL_REVISION
    assert manifest["modelKey"] == MODEL_KEY


def test_task_prefixes_match_local_config_when_present():
    config_path = DEFAULT_WEIGHTS_DIR / "config.json"
    if not config_path.is_file():
        pytest.skip("local snapshot config not staged")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["n_positions"] == MAX_INPUT_TOKENS
    upstream = {task["prefix"] for task in config["task_specific_params"].values()}
    assert upstream == set(TASK_PREFIXES)


def test_verify_snapshot_accepts_matching_manifest(tmp_path: Path):
    _write_snapshot(tmp_path)
    result = verify_snapshot(tmp_path)
    assert result["revision"] == MODEL_REVISION
    assert result["path"] == str(tmp_path)


def test_verify_snapshot_rejects_tampered_digest(tmp_path: Path):
    manifest_path = _write_snapshot(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = manifest["files"][0]["sha256"]
    manifest["files"][0]["sha256"] = ("0" if digest[0] != "0" else "1") + digest[1:]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(tmp_path)


def test_verify_snapshot_rejects_tampered_bytes_and_missing_file(tmp_path: Path):
    _write_snapshot(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"weightz")
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"short")
    with pytest.raises(ValueError, match="size"):
        verify_snapshot(tmp_path)
    (tmp_path / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError):
        verify_snapshot(tmp_path)


def test_verify_snapshot_rejects_wrong_identity(tmp_path: Path):
    manifest_path = _write_snapshot(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["revision"] = "0" * 40
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(tmp_path)
    with pytest.raises(FileNotFoundError):
        verify_snapshot(tmp_path / "missing")


def test_from_pretrained_refuses_without_snapshot_or_download(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="allow_download=False"):
        T5BaseText2TextPipeline.from_pretrained(weights_dir=tmp_path, allow_download=False)


def test_stage_missing_files_fetches_only_absent_entries_then_verifies(tmp_path: Path):
    """Fresh-clone shape: manifest committed, weight file absent. allow_download fetches exactly that file."""
    payload = b"weights-bytes"
    (tmp_path / "config.json").write_bytes(b"{}")
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {"path": "config.json", "bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()},
            {"path": "model.bin", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()},
        ],
    }
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(tmp_path)
    fetched = []

    def fake_download(relative_path, root):
        fetched.append(relative_path)
        (root / relative_path).write_bytes(payload)

    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == ["model.bin"]
    assert fetched == ["model.bin"]
    assert len(verify_snapshot(tmp_path)["files"]) == 2
    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == []


def test_stage_missing_files_refuses_foreign_manifest(tmp_path: Path):
    manifest = {"modelId": "someone/else", "revision": MODEL_REVISION, "files": []}
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(tmp_path, allow_download=True, downloader=lambda *_: None)


def test_generate_rejects_bad_inputs():
    pipe = _pipeline()
    with pytest.raises(TypeError):
        pipe.generate(b"bytes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="empty"):
        pipe.generate("   ")
    with pytest.raises(ValueError, match="MAX_TEXT_CHARS"):
        pipe.generate("x" * (MAX_TEXT_CHARS + 1))
    with pytest.raises(ValueError, match="MAX_INPUT_TOKENS"):
        pipe.generate(" ".join(["w"] * MAX_INPUT_TOKENS))  # MAX_INPUT_TOKENS words + EOS > ceiling
    with pytest.raises(ValueError, match="max_new_tokens"):
        pipe.generate("summarize: ok", max_new_tokens=0)
    with pytest.raises(ValueError, match="max_new_tokens"):
        pipe.generate("summarize: ok", max_new_tokens=MAX_NEW_TOKENS + 1)
    with pytest.raises(TypeError):
        pipe.generate("summarize: ok", max_new_tokens=2.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        pipe.generate("summarize: ok", max_new_tokens=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="num_beams"):
        pipe.generate("summarize: ok", num_beams=0)
    with pytest.raises(ValueError, match="num_beams"):
        pipe.generate("summarize: ok", num_beams=MAX_NUM_BEAMS + 1)


def test_generate_output_fields_and_prefix_detection():
    pipe = _pipeline()
    result = pipe.generate("translate English to German: The house is wonderful.", max_new_tokens=3)
    assert result["text"] == "TRANSLATE ENGLISH TO"
    assert result["generated_tokens"] == 3
    assert result["input_tokens"] == 9  # 8 words + EOS under the fake counter
    assert result["stopped_by"] == "max_new_tokens"
    assert result["known_prefix"] == "translate English to German: "
    assert result["generation"] == {
        "max_new_tokens": 3,
        "num_beams": 1,
        "do_sample": False,
        "decision_rule": pipe.generate("summarize: a")["generation"]["decision_rule"],
    }
    assert result["model_id"] == MODEL_ID
    assert result["model_revision"] == MODEL_REVISION
    assert result["device"] == "cpu"
    assert result["source"] == "injected"
    unprefixed = pipe.generate("The house is wonderful.")
    assert unprefixed["known_prefix"] is None
    assert unprefixed["stopped_by"] == "eos"
    assert unprefixed["generation"]["max_new_tokens"] == DEFAULT_MAX_NEW_TOKENS


def test_generate_accepts_ceiling_boundaries():
    pipe = _pipeline()
    result = pipe.generate(
        " ".join(["w"] * (MAX_INPUT_TOKENS - 1)), max_new_tokens=MAX_NEW_TOKENS, num_beams=MAX_NUM_BEAMS
    )
    assert result["input_tokens"] == MAX_INPUT_TOKENS
    assert result["generation"]["num_beams"] == MAX_NUM_BEAMS
