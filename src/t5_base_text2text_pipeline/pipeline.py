"""Text-to-text generation with the pinned ``google-t5/t5-base`` checkpoint.

The class loads weights only from a digest-verified local snapshot (``weights/t5-base/``) or, when
explicitly allowed, from the Hugging Face Hub at the pinned revision. The caller supplies the task
prefix (``summarize: ``, ``translate English to German: `` ...); this module never adds one.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_ID = "google-t5/t5-base"
MODEL_REVISION = "a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "t5-base"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"

MAX_INPUT_TOKENS = 512  # ``n_positions`` in the snapshot config.json; longer inputs are rejected, not cut
MAX_NEW_TOKENS = 512  # ceiling on decoder steps per call
DEFAULT_MAX_NEW_TOKENS = 64
MAX_TEXT_CHARS = 20_000  # pre-tokenisation guard on the input string
MAX_NUM_BEAMS = 8
DECISION_RULE = "greedy argmax per decoding step (num_beams=1); beam search when num_beams > 1; no sampling"
# The four prefixes the upstream checkpoint was trained on, read from ``task_specific_params`` in the
# snapshot config.json. Reported back as ``known_prefix``; the pipeline does not prepend any of them.
TASK_PREFIXES = (
    "summarize: ",
    "translate English to German: ",
    "translate English to French: ",
    "translate English to Romanian: ",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"snapshot manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        return json.load(fh)


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local snapshot against its manifest; raise naming the first mismatch."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest = _read_manifest(root)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    for entry in manifest.get("files", []):
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {"path": str(root), **manifest}


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest = _read_manifest(root)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


INPUT_SCHEMA: dict[str, Any] = {
    "input": "one non-empty str that already carries its task prefix; the pipeline prepends none",
    "text_chars": [1, MAX_TEXT_CHARS],
    "input_tokens": [1, MAX_INPUT_TOKENS],
    "max_new_tokens": [1, MAX_NEW_TOKENS],
    "num_beams": [1, MAX_NUM_BEAMS],
    "task_prefixes": list(TASK_PREFIXES),
    "decision_rule": DECISION_RULE,
    "preprocessing": (
        "SentencePiece encoding with no prefix added and no truncation: an input over "
        "MAX_INPUT_TOKENS is rejected with a ValueError naming the count, never cut"
    ),
}


def _check_inputs(text: Any, max_new_tokens: Any, num_beams: Any) -> str:
    """Raise TypeError/ValueError naming the first violated ceiling; return the text.

    The encoder-token ceiling is not checked here because it needs the loaded tokenizer;
    ``_check_input_tokens`` applies it inside the pipeline once the count is known.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    if not text.strip():
        raise ValueError("text is empty")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"text has {len(text)} chars; ceiling is MAX_TEXT_CHARS={MAX_TEXT_CHARS}")
    for name, value, ceiling in (
        ("max_new_tokens", max_new_tokens, MAX_NEW_TOKENS),
        ("num_beams", num_beams, MAX_NUM_BEAMS),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an int")
        if not 1 <= value <= ceiling:
            raise ValueError(f"{name} must be between 1 and {ceiling}, got {value}")
    return text


def _check_input_tokens(n_input: int) -> int:
    """The encoder-token ceiling, applied once the tokenizer has counted."""
    if n_input > MAX_INPUT_TOKENS:
        raise ValueError(f"input is {n_input} tokens; ceiling is MAX_INPUT_TOKENS={MAX_INPUT_TOKENS}")
    return n_input


def known_prefix(text: str) -> str | None:
    """Which trained task prefix ``text`` starts with, or ``None``; nothing is prepended."""
    return next((prefix for prefix in TASK_PREFIXES if text.startswith(prefix)), None)


def validate_inputs(
    texts: Sequence[str],
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    num_beams: int = 1,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, per-input observations, verdict).

    Rejection is reported by raising exactly as ``generate`` would: both route through
    ``_check_inputs``. ``generate`` takes one text per call, so ``texts`` is the batch the notebook
    will loop over and every entry is validated with the same settings. An input that starts with
    no trained prefix is **not** rejected — the pipeline does not refuse it either — but the
    manifest records ``known_prefix: null`` so the caller can see it. The encoder-token ceiling
    (``MAX_INPUT_TOKENS``) needs the loaded tokenizer and is enforced inside ``generate``.
    """
    if isinstance(texts, str | bytes) or not isinstance(texts, Sequence):
        raise TypeError("texts must be a sequence of str, not a single string")
    if not texts:
        raise ValueError("texts must hold at least one item")
    checked = [_check_inputs(text, max_new_tokens, num_beams) for text in texts]
    if names is not None and len(names) != len(checked):
        raise ValueError("names must have one entry per text")
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [
            {
                "id": names[i] if names else f"input{i:02d}",
                "chars": len(text),
                "known_prefix": known_prefix(text),
            }
            for i, text in enumerate(checked)
        ],
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def evaluation_report(
    result: Mapping[str, Any], references: Sequence[str] | None = None, *, sample_kind: str = "synthetic"
) -> dict[str, Any]:
    """Evaluation stage: a machine-readable report even though no metric exists here.

    The repository ships no metric helper, so the verdict is always ``not-measurable`` (EVAL9).
    ``references`` exists for interface parity with the fleet's other pipelines and is recorded in
    ``reason`` rather than scored: ROUGE and BLEU need a scorer and enough referenced items to state
    a dispersion, and manufacturing a number from a proxy such as length ratio or copy rate would
    misrepresent a plumbing check as a quality measurement.
    """
    generation = result.get("generation", {})
    supplied = references is not None
    return {
        "task": "caller-prefixed text-to-text generation (summarisation, translation)",
        "score_semantics": (
            "the pipeline emits no probability, confidence or score: generated_tokens, input_tokens "
            "and stopped_by are counts and flags, and "
            f"{generation.get('decision_rule', DECISION_RULE)} produces some token at every step "
            "with no minimum-probability cut-off and no shipped acceptance threshold"
        ),
        "sample_kind": sample_kind,
        "n_generated_tokens": int(result.get("generated_tokens", 0)),
        "metrics": [],
        "baselines": [],
        "verdict": "not-measurable",
        "reason": (
            "the repository ships no metric helper and a generation has no ground truth here"
            + (
                "; references were supplied but no metric helper exists to score them, and one "
                "reference is not a dispersion"
                if supplied
                else "; the evaluated sample has no reference outputs"
            )
        ),
        "needs": (
            "reference outputs from the deployment domain — a reference summary per document, a "
            "reference translation per sentence — over enough items to state a dispersion, scored "
            "with the caller's own ROUGE-1/2/L or BLEU/chrF implementation, excluding or re-running "
            "outputs whose stopped_by is max_new_tokens; no proxy such as length ratio or copy rate "
            "substitutes for that"
        ),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


@dataclass
class T5BaseText2TextPipeline:
    """``_runner(text, max_new_tokens, num_beams)`` -> ``(generated_text, generated_tokens, stopped_by)``;
    ``_count_tokens(text)`` -> encoder token count incl. EOS. Both injectable so tests run offline."""

    _runner: Callable[[str, int, int], tuple[str, int, str]]
    _count_tokens: Callable[[str], int]
    device: str = "cpu"
    source: str = "injected"

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> T5BaseText2TextPipeline:
        import torch
        from transformers import T5ForConditionalGeneration, T5TokenizerFast

        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        if (root / MANIFEST_NAME).is_file():
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
            location, kwargs, source = str(root), dict(local_files_only=True), "local-snapshot"
        elif allow_download:
            location, kwargs, source = MODEL_ID, dict(revision=MODEL_REVISION), "hf-hub"
        else:
            raise FileNotFoundError(f"no verified snapshot at {root} and allow_download=False")
        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        tokenizer = T5TokenizerFast.from_pretrained(location, trust_remote_code=False, **kwargs)
        model = T5ForConditionalGeneration.from_pretrained(
            location, dtype=torch.float32, trust_remote_code=False, **kwargs
        )
        model = model.to(resolved_device).eval()
        eos_id, pad_id = model.config.eos_token_id, model.config.pad_token_id

        def count_tokens(text: str) -> int:
            return len(tokenizer(text, truncation=False)["input_ids"])

        def runner(text: str, max_new_tokens: int, num_beams: int) -> tuple[str, int, str]:
            enc = tokenizer(text, return_tensors="pt", truncation=False).to(resolved_device)
            with torch.inference_mode():
                out = model.generate(
                    **enc, max_new_tokens=max_new_tokens, num_beams=num_beams, do_sample=False
                )
            ids = out[0].tolist()
            content = [t for t in ids if t not in (eos_id, pad_id)]
            stopped_by = "eos" if eos_id in ids else "max_new_tokens"
            return tokenizer.decode(content, skip_special_tokens=True), len(content), stopped_by

        return cls(runner, count_tokens, resolved_device, source)

    def _validate(self, text: Any, max_new_tokens: Any, num_beams: Any) -> int:
        text = _check_inputs(text, max_new_tokens, num_beams)
        return _check_input_tokens(self._count_tokens(text))

    def generate(
        self, text: str, *, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS, num_beams: int = 1
    ) -> dict[str, Any]:
        """Run one prefixed input through encoder-decoder generation; the caller owns the task prefix."""
        n_input = self._validate(text, max_new_tokens, num_beams)
        generated, n_generated, stopped_by = self._runner(text, max_new_tokens, num_beams)
        if not isinstance(generated, str) or not isinstance(n_generated, int):
            raise RuntimeError("runner must return (str, int, str)")
        prefix = known_prefix(text)
        return {
            "text": generated,
            "generated_tokens": n_generated,
            "input_tokens": n_input,
            "stopped_by": stopped_by,
            "known_prefix": prefix,
            "generation": {
                "max_new_tokens": max_new_tokens,
                "num_beams": num_beams,
                "do_sample": False,
                "decision_rule": DECISION_RULE,
            },
            "device": self.device,
            "source": self.source,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }
