# T5-Base Text2Text Pipeline

DIMER-oriented inference and fine-tuning wrapper for **google-t5/t5-base** (the original 220 M-parameter T5, not FLAN), pinned to an immutable Hugging Face revision. The repository exposes prefixed text-to-text generation — the caller supplies `summarize: `, `translate English to German: `, `translate English to French: ` or `translate English to Romanian: ` — with deterministic greedy decoding by default, a supply-chain check of the local weight snapshot, machine-readable provenance, and a bounded adaptation contract that teaches a **new task prefix**: a digest-pinned real referenced corpus (SciTLDR-A abstracts paired with their paper titles), corpus ROUGE-1/2/L with a Lead-N baseline, supervised fine-tuning of the last decoder blocks with validation-ROUGE-L epoch selection, and a safetensors adapter that records its prefix and reloads onto the digest-verified base.

## Upstream alignment

- Model: `google-t5/t5-base`
- Revision: `a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1`
- Upstream weight license: Apache-2.0
- Upstream task: text-to-text generation (summarisation and En→De/Fr/Ro translation via task prefixes; pre-trained on C4 plus a supervised multi-task mixture)
- Repository adaptation: bounded supervised fine-tuning of the last *k* decoder blocks (`adapt`, default 4 of 12 = 37,757,952 of 222,903,552 parameters) that teaches a caller-chosen prefix on caller-supplied or pinned SciTLDR-A records; the encoder, embeddings and tied output projection are never modified; the adapter carries only the trained tensors, records the prefix, and is bound to the base `model.safetensors` SHA-256

## Quick start

```python
from t5_base_text2text_pipeline import T5BaseText2TextPipeline

pipe = T5BaseText2TextPipeline.from_pretrained()          # verifies weights/t5-base first
result = pipe.generate("translate English to German: The house is wonderful.", max_new_tokens=32)
print(result["text"])            # 'Das Haus ist wunderbar.'  (CPU smoke, greedy)
print(result["generated_tokens"], result["stopped_by"], result["known_prefix"])
```

Teaching a new prefix on the pinned SciTLDR-A sample (CPU, about six minutes including the validation passes):

```python
from t5_base_text2text_pipeline import T5BaseText2TextPipeline, DEFAULT_PREFIX, fetch_sample_dataset, check_split_disjoint, lead_baseline

GEN = {'max_new_tokens': 24, 'num_beams': 4}
splits = fetch_sample_dataset()            # three pinned JSON-Lines files (5.5 MB), digest-verified, cached under weights/scitldr/
check_split_disjoint(splits)               # 300 / 50 / 100 abstract–title records from SciTLDR-A's own paper-disjoint members
pipe = T5BaseText2TextPipeline.from_pretrained()
print(lead_baseline(splits['test'])['rougeL'], pipe.evaluate(splits['test'], prefix=DEFAULT_PREFIX, **GEN)['rougeL'])   # 14.08, 0.71 in the recorded run
pipe.adapt(splits['train'], splits['validation'], prefix=DEFAULT_PREFIX, eval_generation=GEN)                        # last 4 decoder blocks, 2 epochs, best validation ROUGE-L kept
print(pipe.evaluate(splits['test'], prefix=DEFAULT_PREFIX, **GEN)['rougeL'])                                        # 29.97 in the recorded run
artifact = pipe.save_artifact('outputs/adapter')                                                                    # adapter.safetensors (151 MB) + manifest.json with the prefix
again = T5BaseText2TextPipeline.from_artifact(artifact)                                                             # verifies base digest + artifact digest before applying
```

`generate(text, *, max_new_tokens=64, num_beams=1)` takes one non-empty string of at most 20,000 characters (`MAX_TEXT_CHARS`) that tokenises to at most 512 SentencePiece tokens (`MAX_INPUT_TOKENS`; longer inputs are rejected, not truncated), `max_new_tokens` in 1..512 (`MAX_NEW_TOKENS`) and `num_beams` in 1..8 (`MAX_NUM_BEAMS`). The pipeline never adds a prefix; `TASK_PREFIXES` lists the four the checkpoint was trained on and each result reports which one the input started with as `known_prefix` (or `None`). Upstream `task_specific_params` (min_length, length_penalty, no_repeat_ngram_size) are not applied. Every result carries `text`, `generated_tokens`, `input_tokens`, `stopped_by`, `generation` settings, `device`, `source`, `model_id` and `model_revision`. `evaluate(records, prefix=..., **generation)` prepends the prefix to a validated `{id, source, targets}` dataset and reports corpus ROUGE-1/2/L with the mean output and reference lengths (`measured` / `measured-small-sample`); `evaluation_report(result, references)` scores one item as `sample-sanity` and still returns `not-measurable` without references; `adapt(train, val, *, prefix, epochs=2, lr=5e-4, batch_size=8, trainable_decoder_layers=4, seed=0, eval_generation=...)` fine-tunes the last decoder blocks under the prefix and keeps the best-validation-ROUGE-L epoch; `save_artifact` / `from_artifact` export and reload the trained tensors as safetensors with a manifest bound to the base weight digest. Dataset helpers (`fetch_corpus`, `read_corpus`, `build_sample_dataset`, `validate_dataset`, `split_dataset`, `check_split_disjoint`, `load_byod_dataset`, `write_dataset_csv`) live in `samples.py`; ROUGE and `lead_baseline` in `metrics.py`; records are 8–20,000 mappings with un-prefixed sources and one or more references each, the prefix is a short string ending in `: `, and every inference ceiling is a refusal, never a silent cut (training truncates prefixed sources to 512 and targets to 64 tokens).

## Weights layout

```
weights/t5-base/
  config.json  generation_config.json  model.safetensors  spiece.model  tokenizer.json
  README.md  dimer-base-manifest.json  (no tokenizer_config.json at this revision)
```

`from_pretrained()` calls `stage_missing_files()` (fetches absent manifest entries at the pinned revision, only with `allow_download=True`) then `verify_snapshot()` (size + SHA-256 of every entry), and loads `T5ForConditionalGeneration` + `T5TokenizerFast` with `local_files_only=True` and `trust_remote_code=False`. Without a manifest it raises unless `allow_download=True`. See `docs/WEIGHTS.md`.

## Tests

```
pip install -e . --no-deps
pytest -q -o addopts= tests
```

Tests are offline: they use an injected fake runner, token counter and corpus fetcher plus temporary manifests, never the weights (33 tests plus 5 notebook-parity tests). `tests/test_model_backed.py` (2 tests: `evaluate` under a new prefix, a one-epoch adaptation of the last decoder block with an artifact round trip) runs only when `weights/t5-base/` is staged.

## Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/t5-base-text2text-pipeline/blob/main/tutorials/t5_base_text2text_colab.ipynb)

`tutorials/t5_base_text2text_colab.ipynb` is declared `E2E` (mode `GUIDED`) under DIMER Notebook Specification 2.0 and is **standalone** (§4): generated by `tools/build_notebook.py`, it carries the three pipeline modules, model identity, manifest digests and runtime pins, so the exported notebook runs without this repository (parity enforced by `tests/test_notebook_parity.py`; see `tutorials/README.md`). Its default path fetches the three pinned SciTLDR-A files (5.5 MB, Apache-2.0, each refused on any digest mismatch), pairs abstracts with their titles and draws 300 / 50 / 100 source-disjoint records with four refusal probes, stages the git-ignored `model.safetensors` with `stage_missing_files(..., allow_download=True)` and digest-verifies the snapshot, exercises the inference contract on two trained-prefix inputs and probes the new prefix `paper title: ` on one abstract, scores the Lead-1 baseline and the frozen model under the new prefix and under `summarize: ` on the test split (ROUGE-L 14.08 / 0.71 / 18.99 in the recorded run — an unknown prefix draws one-word answers), fine-tunes the last four decoder blocks for two epochs with validation-ROUGE-L epoch selection (284.2 s on CPU), re-scores the test split (ROUGE-L 29.97 at 6.5 words against 7.4-word titles), titles four unseen abstracts with a `measured-small-sample` verdict, re-generates the German translation to show the adapter's effect on a trained prefix, exports a 151 MB safetensors adapter that records its prefix, and reloads it with 6/6 identical outputs. Every number is one seeded split with no dispersion estimate. BYOD (`{id, source, target}` / `{id, source, targets}` as CSV, JSON or JSONL plus the caller's `PREFIX`) is optional and gated off by default. See `docs/release-verification.md` for the release gate.

## Release status

**Release-grade** — the `E2E` notebook blob `d2f10303` (committed at `3220429`) executed top-to-bottom in a clean Kaggle Tesla T4 runtime on 2026-09-19 (11/11 ok (1 restart after install cell), 477.4 s); the record is in `docs/release-verification.md` and `STATUS.md`. Static and unit checks — including the standalone generator parity checks — are necessary but were never the evidence; the hosted run is. A later change to the carried modules or the notebook returns the status to Candidate until re-verified.

## Licensing

This repository's code is Apache-2.0 (`LICENSE`). The packaged upstream weights are Apache-2.0; see `docs/WEIGHTS.md` and `MODEL_CARD.md`.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
