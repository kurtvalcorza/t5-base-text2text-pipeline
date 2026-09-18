"""Text-to-text generation with the pinned ``google-t5/t5-base`` checkpoint.

The class loads weights only from a digest-verified local snapshot (``weights/t5-base/``) or, when
explicitly allowed, from the Hugging Face Hub at the pinned revision. The caller supplies the task
prefix (``summarize: ``, ``translate English to German: `` ...); this module never adds one at inference.

The adaptation contract (``evaluate``, ``adapt``, ``save_artifact``, ``from_artifact``) teaches a
caller-chosen prefix: it fine-tunes the last decoder blocks on a validated ``{id, source, targets}`` dataset
whose sources are prepended with that prefix, selects the epoch by validation ROUGE-L, and exports the
trained tensors as a safetensors adapter bound to the pinned base weights. The inference contract above is
unchanged by it.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
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
WEIGHT_FILE = "model.safetensors"
WEIGHT_SHA256 = (
    "a90903540cc02cbeb7ff9f823f1a80eb778c7e22426a0e620b01c77a5ec8f5b4"  # manifest digest of WEIGHT_FILE
)
PARAMETER_COUNT = 222_903_552
DECODER_LAYERS = 12  # config.json num_decoder_layers
DEFAULT_TRAINABLE_DECODER_LAYERS = 4  # the last four decoder blocks (37,757,952 parameters)
MAX_TRAIN_SOURCE_TOKENS = 512  # prefixed-source truncation ceiling during adaptation (never at inference)
MAX_TRAIN_TARGET_TOKENS = 64  # target truncation ceiling during adaptation
MAX_PREFIX_CHARS = 64
MAX_EVAL_RECORDS = 2_000
MIN_SCORED_RECORDS = 50  # below this a scored set is labelled a small sample
ARTIFACT_FORMAT = "org.valcorza.t5-base.adapter.v1"
ARTIFACT_FORMAT_VERSION = "1.0"
ARTIFACT_WEIGHTS_NAME = "adapter.safetensors"
ARTIFACT_MANIFEST_NAME = "manifest.json"
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


def _check_prefix(prefix: Any) -> str:
    """A task prefix is a short non-empty string ending in ': ' (the upstream convention)."""
    if not isinstance(prefix, str):
        raise TypeError("prefix must be str")
    if not prefix.strip():
        raise ValueError("prefix is empty")
    if len(prefix) > MAX_PREFIX_CHARS:
        raise ValueError(f"prefix has {len(prefix)} chars; ceiling is MAX_PREFIX_CHARS={MAX_PREFIX_CHARS}")
    if not prefix.endswith(": "):
        raise ValueError("prefix must end with ': ' like the trained prefixes (e.g. 'paper title: ')")
    return prefix


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
    """Evaluation stage: a machine-readable report for one ``generate`` result.

    With ``references`` (one or more reference outputs for the same input) the report carries the
    ROUGE-1/2/L F1 of that single output against its best-matching reference (`metrics.py`) with the
    verdict ``sample-sanity`` — one item is a plumbing check, not a quality measurement; the corpus-level
    stage is ``T5BaseText2TextPipeline.evaluate``. Without references the verdict is ``not-measurable``.
    """
    from .metrics import rouge_scores

    generation = result.get("generation", {})
    supplied = references is not None
    metrics: list[dict[str, Any]] = []
    if supplied:
        refs = [str(r) for r in references if str(r).strip()]
        if not refs:
            raise ValueError("references must hold at least one non-empty output")
        scores = rouge_scores(str(result.get("text", "")), refs)
        metrics = [
            {"id": name, "value": 100.0 * value, "estimation": "single item, best of the references"}
            for name, value in scores.items()
        ]
    return {
        "task": "caller-prefixed text-to-text generation (summarisation, translation, or a taught prefix)",
        "score_semantics": (
            "the pipeline emits no probability, confidence or score: generated_tokens, input_tokens "
            "and stopped_by are counts and flags, and "
            f"{generation.get('decision_rule', DECISION_RULE)} produces some token at every step "
            "with no minimum-probability cut-off and no shipped acceptance threshold; ROUGE, when "
            "references are supplied, is n-gram agreement with those references (own implementation), "
            "not faithfulness"
        ),
        "sample_kind": sample_kind,
        "n_generated_tokens": int(result.get("generated_tokens", 0)),
        "known_prefix": result.get("known_prefix"),
        "metrics": metrics,
        "baselines": [],
        "verdict": "sample-sanity" if supplied else "not-measurable",
        "reason": (
            "ROUGE-1/2/L are computed for one item against its reference outputs with the repository's own "
            "implementation; a single item states no dispersion and is not a quality measurement"
            if supplied
            else "no reference output was supplied, so ROUGE cannot be computed; "
            "a generation has no ground truth here"
        ),
        "needs": (
            "reference outputs from the deployment domain — a reference summary per document, a reference "
            "translation per sentence, a reference title per abstract — over enough items to state a "
            "dispersion, scored with `evaluate` (ROUGE-1/2/L, own implementation; BLEU/chrF for translation "
            "need the caller's own scorer), excluding or re-running outputs whose stopped_by is "
            "max_new_tokens; no proxy such as length ratio or copy rate substitutes for that"
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
    adapter: dict[str, Any] | None = field(default=None, repr=False)
    _model: Any = field(default=None, repr=False)
    _tokenizer: Any = field(default=None, repr=False)

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> T5BaseText2TextPipeline:
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        if (root / MANIFEST_NAME).is_file():
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
            location, kwargs, source = str(root), dict(local_files_only=True), "local-snapshot"
        elif allow_download:
            location, kwargs, source = MODEL_ID, dict(revision=MODEL_REVISION), "hf-hub"
        else:
            raise FileNotFoundError(f"no verified snapshot at {root} and allow_download=False")
        # Refuse invalid snapshots before importing model libraries.
        import torch
        from transformers import T5ForConditionalGeneration, T5TokenizerFast

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

        return cls(runner, count_tokens, resolved_device, source, _model=model, _tokenizer=tokenizer)

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

    # ---- adaptation -----------------------------------------------------------------------------------

    def _require_model(self) -> tuple[Any, Any]:
        if self._model is None or self._tokenizer is None:
            raise ValueError(
                "this operation needs a pipeline built with from_pretrained() or from_artifact()"
            )
        return self._model, self._tokenizer

    def evaluate(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        prefix: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        num_beams: int = 1,
    ) -> dict[str, Any]:
        """Generate from every record's prefixed source and score the outputs against its references
        (ROUGE-1/2/L). The prefix is the caller's; nothing checks it is one the model knows."""
        from .metrics import text_metrics
        from .samples import validate_dataset

        prefix = _check_prefix(prefix)
        checked = validate_dataset(records, min_records=1, max_records=MAX_EVAL_RECORDS)["records"]
        started = time.perf_counter()
        hypotheses, truncated = [], 0
        for record in checked:
            result = self.generate(
                prefix + record["source"], max_new_tokens=max_new_tokens, num_beams=num_beams
            )
            hypotheses.append(result["text"])
            truncated += result["stopped_by"] == "max_new_tokens"
        metrics = text_metrics(hypotheses, [r["targets"] for r in checked])
        metrics.update(
            {
                "prefix": prefix,
                "known_prefix": known_prefix(prefix),
                "hit_token_ceiling": truncated,
                "generation": {"max_new_tokens": max_new_tokens, "num_beams": num_beams, "do_sample": False},
                "verdict": "measured" if len(checked) >= MIN_SCORED_RECORDS else "measured-small-sample",
                "adapted": self.adapter is not None,
                "seconds": round(time.perf_counter() - started, 3),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
            }
        )
        return metrics

    def _trainable_names(self, trainable_decoder_layers: int) -> list[str]:
        if (
            not isinstance(trainable_decoder_layers, int)
            or not 1 <= trainable_decoder_layers <= DECODER_LAYERS
        ):
            raise ValueError(f"trainable_decoder_layers must be an int in 1..{DECODER_LAYERS}")
        model, _ = self._require_model()
        first = DECODER_LAYERS - trainable_decoder_layers
        prefixes = tuple(f"decoder.block.{k}." for k in range(first, DECODER_LAYERS))
        return [name for name, _p in model.named_parameters() if name.startswith(prefixes)]

    def adapt(
        self,
        train: Sequence[Mapping[str, Any]],
        val: Sequence[Mapping[str, Any]] | None = None,
        *,
        prefix: str,
        epochs: int = 2,
        lr: float = 5e-4,
        batch_size: int = 8,
        trainable_decoder_layers: int = DEFAULT_TRAINABLE_DECODER_LAYERS,
        seed: int = 0,
        progress: Callable[[dict[str, Any]], None] | None = None,
        eval_generation: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Bounded supervised fine-tuning that teaches `prefix` on a validated text-to-text dataset.

        Every training source is prepended with `prefix`; only the last `trainable_decoder_layers` decoder
        blocks train (4 by default; the encoder, the shared embeddings, the earlier decoder blocks and the
        tied output projection stay frozen). Teacher-forced cross-entropy on the first reference, AdamW at
        a fixed learning rate with gradient clipping at 1.0, prefixed sources truncated to
        MAX_TRAIN_SOURCE_TOKENS and targets to MAX_TRAIN_TARGET_TOKENS **during training only**. Epoch 0
        records the frozen model's validation ROUGE under the same prefix; every epoch is scored on the
        validation split with `eval_generation` (the pipeline defaults unless given), and the epoch with the
        highest validation ROUGE-L is kept."""
        from .samples import validate_dataset

        prefix = _check_prefix(prefix)
        if not isinstance(epochs, int) or not 1 <= epochs <= 20:
            raise ValueError("epochs must be an int in 1..20")
        if not (0.0 < lr <= 1e-2):
            raise ValueError("lr must be in (0, 1e-2]")
        if not isinstance(batch_size, int) or not 1 <= batch_size <= 32:
            raise ValueError("batch_size must be an int in 1..32")
        names = self._trainable_names(trainable_decoder_layers)
        train_checked = validate_dataset(train)["records"]
        val_checked = (
            validate_dataset(val, min_records=1, max_records=MAX_EVAL_RECORDS)["records"] if val else []
        )
        generation = dict(eval_generation or {})
        import torch

        torch.manual_seed(seed)
        model, tokenizer = self._require_model()
        started = time.perf_counter()
        wanted = set(names)
        for name, param in model.named_parameters():
            param.requires_grad_(name in wanted)
        params = [p for p in model.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in params)
        optimiser = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        device = torch.device(self.device)

        def score_val() -> dict[str, Any] | None:
            if not val_checked:
                return None
            model.eval()
            return {
                k: v
                for k, v in self.evaluate(val_checked, prefix=prefix, **generation).items()
                if k in ("rouge1", "rouge2", "rougeL", "n", "mean_output_words", "hit_token_ceiling")
            }

        history: list[dict[str, Any]] = []
        entry: dict[str, Any] = {"epoch": 0, "train_loss": None, "val": score_val(), "note": "frozen model"}
        history.append(entry)
        if progress:
            progress(entry)
        best_rouge = entry["val"]["rougeL"] if entry["val"] else -math.inf
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in wanted}
        best_epoch = 0
        generator = torch.Generator().manual_seed(seed)
        for epoch in range(1, epochs + 1):
            model.train()
            order = torch.randperm(len(train_checked), generator=generator).tolist()
            losses = []
            for start in range(0, len(order), batch_size):
                batch = [train_checked[i] for i in order[start : start + batch_size]]
                encoded = tokenizer(
                    [prefix + r["source"] for r in batch],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MAX_TRAIN_SOURCE_TOKENS,
                )
                labels = tokenizer(
                    text_target=[r["targets"][0] for r in batch],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MAX_TRAIN_TARGET_TOKENS,
                )["input_ids"]
                labels[labels == tokenizer.pad_token_id] = -100
                out = model(
                    input_ids=encoded["input_ids"].to(device),
                    attention_mask=encoded["attention_mask"].to(device),
                    labels=labels.to(device),
                )
                optimiser.zero_grad(set_to_none=True)
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimiser.step()
                losses.append(float(out.loss.detach()))
            model.eval()
            entry = {"epoch": epoch, "train_loss": sum(losses) / len(losses), "val": score_val()}
            history.append(entry)
            if progress:
                progress(entry)
            current = entry["val"]["rougeL"] if entry["val"] else math.inf
            if current > best_rouge or not entry["val"]:
                best_rouge = current
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in wanted}
                best_epoch = epoch
        merged = dict(model.state_dict())
        merged.update(best_state)
        model.load_state_dict(merged, strict=True)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self.adapter = {
            "prefix": prefix,
            "trainable_decoder_layers": trainable_decoder_layers,
            "trainable_names": names,
            "n_trainable": n_trainable,
            "n_total": sum(p.numel() for p in model.parameters()),
            "epochs": epochs,
            "best_epoch": best_epoch,
            "selection": "highest validation ROUGE-L" if val_checked else "final epoch (no validation split)",
            "lr": lr,
            "batch_size": batch_size,
            "max_train_source_tokens": MAX_TRAIN_SOURCE_TOKENS,
            "max_train_target_tokens": MAX_TRAIN_TARGET_TOKENS,
            "eval_generation": generation,
            "n_train": len(train_checked),
            "n_val": len(val_checked),
            "seed": seed,
            "history": history,
            "seconds": round(time.perf_counter() - started, 2),
        }
        return dict(self.adapter)

    # ---- artifacts ------------------------------------------------------------------------------------

    def save_artifact(self, output_dir: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        """Write the adapted decoder tensors as safetensors with a manifest naming the base and the prefix."""
        if self.adapter is None:
            raise ValueError("nothing to save: call adapt() first")
        model, _ = self._require_model()
        from safetensors.torch import save_file

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        names = set(self.adapter["trainable_names"])
        tensors = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items() if k in names}
        weights_path = out / ARTIFACT_WEIGHTS_NAME
        save_file(tensors, str(weights_path), metadata={"format": "pt"})
        manifest = {
            "format": ARTIFACT_FORMAT,
            "format_version": ARTIFACT_FORMAT_VERSION,
            "base_model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "key": MODEL_KEY,
                "weight_file": WEIGHT_FILE,
                "weight_sha256": WEIGHT_SHA256,
            },
            "adapter": {k: v for k, v in self.adapter.items() if k not in ("history", "trainable_names")},
            "history": self.adapter["history"],
            "tensors": sorted(tensors),
            "files": [
                {
                    "path": ARTIFACT_WEIGHTS_NAME,
                    "bytes": weights_path.stat().st_size,
                    "sha256": _sha256(weights_path),
                }
            ],
            "metadata": dict(metadata or {}),
        }
        (out / ARTIFACT_MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return out

    def load_artifact(self, artifact_dir: str | Path) -> dict[str, Any]:
        """Verify an adapter's manifest and digest, then overwrite exactly the tensors it carries."""
        root = Path(artifact_dir)
        manifest = json.loads((root / ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
        if manifest.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"artifact format {manifest.get('format')!r} != {ARTIFACT_FORMAT!r}")
        base = manifest.get("base_model", {})
        if (base.get("id"), base.get("revision"), base.get("weight_sha256")) != (
            MODEL_ID,
            MODEL_REVISION,
            WEIGHT_SHA256,
        ):
            raise ValueError("artifact was adapted from a different base model, revision or weight file")
        entry = manifest["files"][0]
        weights_path = root / entry["path"]
        if not weights_path.is_file():
            raise FileNotFoundError(f"artifact weights missing: {weights_path}")
        if _sha256(weights_path) != entry["sha256"] or weights_path.stat().st_size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: digest or size mismatch; refusing to load")
        model, _ = self._require_model()
        from safetensors.torch import load_file

        tensors = load_file(str(weights_path))
        if sorted(tensors) != manifest["tensors"]:
            raise ValueError("artifact tensor names differ from its manifest")
        state = model.state_dict()
        for key, value in tensors.items():
            if key not in state or not key.startswith("decoder.block."):
                raise ValueError(
                    f"artifact tensor {key} is not an adaptable decoder tensor of the base model"
                )
            if tuple(value.shape) != tuple(state[key].shape):
                raise ValueError(
                    f"artifact tensor {key} has shape {tuple(value.shape)}, "
                    f"base has {tuple(state[key].shape)}"
                )
        merged = dict(state)
        merged.update({k: v.to(state[k].dtype) for k, v in tensors.items()})
        model.load_state_dict(merged, strict=True)
        model.eval()
        self.adapter = {
            **manifest["adapter"],
            "trainable_names": manifest["tensors"],
            "history": manifest.get("history", []),
        }
        return manifest

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        *,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> T5BaseText2TextPipeline:
        pipeline = cls.from_pretrained(device=device, weights_dir=weights_dir, allow_download=allow_download)
        pipeline.load_artifact(artifact_dir)
        return pipeline
