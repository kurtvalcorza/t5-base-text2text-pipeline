# Tutorials

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/kurtvalcorza/t5-base-text2text-pipeline)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/t5-base-text2text-pipeline/blob/main/tutorials/t5_base_text2text_colab.ipynb)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-google--t5%2Ft5--base-ffcc4d?style=flat)](https://huggingface.co/google-t5/t5-base)
[![Upstream](https://img.shields.io/badge/Upstream-google--research%2Ftext--to--text--transfer--transformer-181717?style=flat&logo=github&logoColor=white)](https://github.com/google-research/text-to-text-transfer-transformer)
[![arXiv](https://img.shields.io/badge/arXiv-1910.10683-b31b1b.svg)](https://arxiv.org/abs/1910.10683)

Notebook specification: **DIMER Notebook Specification 1.0**

| Notebook | Profile | Capability | Default runtime | BYOD | Release status |
|---|---|---|---|---|---|
| `t5_base_text2text_colab.ipynb` | `TASK-INFERENCE` | caller-prefixed text-to-text generation (summarisation, En→De/Fr/Ro translation) with `google-t5/t5-base` (original T5, not FLAN); greedy decoding by default, beam search on request, no sampling; `generated_tokens`/`input_tokens`/`stopped_by`/`known_prefix` per call; no metric (ROUGE/BLEU need references the sample lacks) | CPU (CUDA used automatically when available) | one UTF-8 text file, one already-prefixed input per line, gated off by default; no references, so no metric | **Candidate** — static checks pass; the clean-runtime execution run is pending and will be recorded in `../docs/release-verification.md`, which must be reviewed for the exact notebook revision before promotion |

## Conformance notes

- The notebook exercises `T5BaseText2TextPipeline` from the repository public API rather than reimplementing model loading; model acquisition goes through the package: `stage_missing_files(WEIGHTS_DIR, allow_download=True)` fetches only the manifest entries a fresh clone lacks (the git-ignored `model.safetensors`), at the pinned revision, `verify_snapshot` re-hashes every entry, and `from_pretrained(weights_dir=WEIGHTS_DIR)` loads the verified files (`local_files_only=True`, `trust_remote_code=False`); the notebook never calls `huggingface_hub` or `transformers` directly.
- The default sample is synthetic: two already-prefixed inputs authored in code (the card-pass smoke's `translate English to German: The house is wonderful.` and a four-sentence `summarize: ` passage); the pipeline invents no prefix and the notebook says so. The sample has no reference outputs, so no ROUGE/BLEU is computed and none is manufactured; the sanity checks are plumbing checks only.
- Decision rule and score semantics: greedy per-step argmax (`num_beams=1`, `do_sample=False`) stated as the default, beam search as the alternative; the pipeline emits no probability or confidence and ships no acceptance threshold — the caller owns it. `stopped_by == 'max_new_tokens'` is explained as a truncated output.
- Ceilings `MAX_TEXT_CHARS` (20000), `MAX_INPUT_TOKENS` (512, rejected not truncated), `MAX_NEW_TOKENS` (512), `MAX_NUM_BEAMS` (8), `DEFAULT_MAX_NEW_TOKENS` (64), `DECISION_RULE` and `TASK_PREFIXES` are printed before the model runs; instruction following, sampling, batching, other language pairs and the upstream `task_specific_params` are named as out of scope.
- CPU is documented as adequate with the card-measured smoke numbers (4.55 s load + verify, 0.29 s / 0.83 s per call on an Intel Core Ultra 9 275HX).
- `USE_BYOD` defaults to `False` so the sample path never opens an upload dialog.
- The pinned snapshot is a 6-file manifest: upstream ships no `tokenizer_config.json` at this revision, so `T5TokenizerFast` is built from `tokenizer.json` with class defaults (documented in the card, README and `docs/WEIGHTS.md`); the notebook's Section 4 says so before the load.
- `tools/validate_release_assets.py` performs source validation only. It does not satisfy the
  clean-runtime execution requirement; a release review must confirm that a recorded clean run in
  `docs/release-verification.md` matches the notebook revision under review before the status is
  promoted to `Release-grade`.
