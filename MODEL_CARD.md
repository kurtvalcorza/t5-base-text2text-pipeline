---
license: apache-2.0
model_card_spec: "1.1"
pipeline_tag: text2text-generation
base_model: google-t5/t5-base
date_published: "2019-10"
date_published_source: "google-research/text-to-text-transfer-transformer initial release 2019-10-23 (arXiv:1910.10683 v1 same day); Hub history begins 2019-12-12"
---

# T5-Base (DIMER package v0.1.0) — Text-to-Text Transfer Transformer (Text2Text Generation)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-google--t5%2Ft5--base-ffcc4d?style=flat)](https://huggingface.co/google-t5/t5-base)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-google--research%2Ftext--to--text--transfer--transformer-181717?style=flat&logo=github&logoColor=white)](https://github.com/google-research/text-to-text-transfer-transformer)
[![arXiv Paper](https://img.shields.io/badge/arXiv-1910.10683-b31b1b.svg)](https://arxiv.org/abs/1910.10683)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This pipeline provides a ready-to-run interactive Google Colab notebook that exercises the repository's public API end to end — bootstrap a fresh runtime, stage and verify the pinned upstream revision, validate an input, run the task, and inspect and export the outputs:

- **Task Inference Tutorial**:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/t5-base-text2text-pipeline/blob/main/tutorials/t5_base_text2text_colab.ipynb) [`t5_base_text2text_colab.ipynb`](https://github.com/kurtvalcorza/t5-base-text2text-pipeline/blob/main/tutorials/t5_base_text2text_colab.ipynb)  
  *Caller-prefixed text-to-text generation with the pinned `google-t5/t5-base` weights (`translate English to German: `, `summarize: `): greedy deterministic decoding with an explicit token budget; the pinned revision ships no `tokenizer_config.json` (6-file manifest); no metric is reported.*

---

#### Description

`google-t5/t5-base` is the 220-million-parameter checkpoint of the original Text-To-Text Transfer Transformer released by Raffel et al. (JMLR 21(140), 2020; arXiv:1910.10683), pinned here to revision `a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1`. It is a standard encoder-decoder Transformer: 12 encoder and 12 decoder layers, `d_model` 768, 12 attention heads of width 64, feed-forward width 3072, relative position biases with 32 buckets instead of absolute position embeddings, and a shared SentencePiece vocabulary of 32,128 pieces (`spiece.model`, byte-identical to T5-Small's, 100 of them sentinel `<extra_id_N>` tokens) — all read from the snapshot `config.json` and `tokenizer.json` (this snapshot ships no `tokenizer_config.json`; `T5TokenizerFast` loads from `tokenizer.json` with its class defaults, and the smoke run confirmed pad 0 / EOS 1 decoding). The checkpoint was pre-trained with a span-corruption denoising objective on C4 and simultaneously on a supervised multi-task mixture in which every task is cast as text in, text out, with a plain-language prefix naming the task (upstream README, "Training Details"); this is the pre-FLAN T5, not an instruction-tuned model. At inference the encoder reads the prefixed input once, then the decoder emits one SentencePiece token per step, starting from the pad token (`decoder_start_token_id` 0) and stopping at `</s>` (id 1) or a step ceiling; nothing is adapted, fine-tuned or conditioned beyond the text the caller supplies. What this repository adds is packaging: the `T5BaseText2TextPipeline` class in `src/t5_base_text2text_pipeline/pipeline.py`, digest verification of the local snapshot (`verify_snapshot`, `stage_missing_files`), input validation with named ceilings, a fixed output contract, and no task prefixes of its own — the caller writes `summarize: ` or `translate English to German: ` in front of the text.

#### Intended Use and Limitations

###### Primary Intended Uses

The task is conditional text generation: input one UTF-8 string that begins with a task prefix the checkpoint was trained on, output one generated string plus its token count and the generation settings used. The upstream `config.json` names four such prefixes in `task_specific_params` — `summarize: `, `translate English to German: `, `translate English to French: `, `translate English to Romanian: ` — and the pipeline exposes the same list as `TASK_PREFIXES`, echoing which one (if any) the input started with as `known_prefix`. Envisioned applications are single-document abstractive-leaning summarisation of short English passages and English-to-German/French/Romanian sentence translation in research prototypes, teaching material, and as a mid-size deterministic baseline against which larger seq2seq or instruction-tuned models are compared. In a larger system the pipeline is an inference component that produces candidate text for a human or a downstream scorer; it is the second-smallest member of the T5 family and is chosen where an 892 MB float32 weight file and roughly 2–3× the T5-Small CPU latency (measured below) are acceptable in exchange for the larger model (the `t5-small-text2text-pipeline` sibling is the 60 M-parameter step down; which one produces better text on your documents is something only your own references can settle).

###### Primary Intended Users

The intended users are machine-learning engineers, NLP researchers and application developers integrating a seq2seq baseline into research prototypes, internal tooling, or the DIMER model workbench. The pipeline assumes its users understand that the model follows only the trained prefixes and not free-form instructions, that greedy decoding returns one deterministic candidate rather than a probability over outputs, that generated text can be fluent and wrong, that the model reads at most 512 SentencePiece tokens of input, and that any quality claim on their own documents needs reference outputs and a ROUGE/BLEU-style scorer they supply. It is not designed for hobbyist "chat with the model" use; there is no chat template and no instruction following.

###### Out-of-scope use cases

1. **Capability boundary:** not an instruction-following or chat model — prompts other than the four trained prefixes (questions, "rewrite this", other language pairs, other source languages) produce copies of the input, fragments, or unrelated text; not a classifier, embedder or masked-LM (the GLUE/SuperGLUE tasks in its training mixture are not exposed here); not multilingual beyond English source text with German, French or Romanian targets (upstream README `language` list). For fill-mask or sentence embeddings use `bert-masked-lm-pipeline`; for open-ended generation use `gpt2-text-generation-pipeline`.
2. **Input boundary:** only `str` is accepted (`TypeError` otherwise); empty or whitespace-only text, text over `MAX_TEXT_CHARS = 20000` characters, or text that tokenises to more than `MAX_INPUT_TOKENS = 512` pieces including `</s>` is rejected with `ValueError` — the pipeline refuses rather than silently truncating; `max_new_tokens` is capped at `MAX_NEW_TOKENS = 512` and `num_beams` at `MAX_NUM_BEAMS = 8`; one input per call, no batching.
3. **Decision boundary:** not for producing text that is acted on without a human reading it — contract or medical translation, summaries that feed automated decisions, content published under a person's name — and not for any use where a fabricated sentence causes harm before it is checked.

#### Factors

###### Groups

The pipeline is human-adjacent rather than human-centric: it consumes and emits natural-language text that routinely mentions people, but it carries no demographic labels and its outputs are strings, not decisions about a person. Its pre-training corpus C4 is filtered English web text whose demographic composition the upstream authors did not audit; the pinned upstream README answers "Bias, Risks, and Limitations" with "More information needed", and Dodge et al. (2021) document that C4's block-list filtering disproportionately removed text about and by minority groups. Neither the upstream card nor this repository reports any group-level quality breakdown for summaries or translations (for example by dialect, named-entity origin, or gendered referents in German/French/Romanian output, all of which require grammatical gender the English source may not specify). The fairness audit therefore transfers to the operator: before deployment, score outputs on a labelled sample of your own inputs stratified by the groups that matter to your application, and treat systematic gender defaulting or degraded quality on a group as a blocker.

###### Instrumentation

There is no physical sensor: the training data was produced by software. C4 was built from the April 2019 Common Crawl web snapshot by rule-based filtering — keeping lines ending in terminal punctuation, dropping pages with fewer than five sentences, pages containing words from a block list, pages with placeholder or code-like text, and duplicate three-sentence spans (paper §2.2) — and the supervised tasks were assembled from published benchmark datasets listed in the upstream README (CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE, CB, COPA, WiC, MultiRC, ReCoRD, BoolQ) plus the translation and summarisation corpora behind the four config prefixes. The instrument characteristics that reach the model are text encoding and tokenisation: the pipeline lower-cases nothing, so casing, punctuation, Unicode normalisation, HTML residue and line breaks in the caller's input become SentencePiece tokens and shift the output; the vocabulary was fit on C4 English (with some German, French and Romanian), so characters outside it become `<unk>` (id 2) and vanish from the output. The pipeline does not detect encoding damage, boilerplate, or a change in input domain; it only rejects the type, length and token-count violations listed above.

###### Environment

Operating environment: Python 3.12 with `torch==2.14.0`, `transformers==4.57.6`, `tokenizers==0.22.2`, `sentencepiece==0.2.2` (exact pins in `pyproject.toml`); `from_pretrained` picks `cuda:0` when available, else CPU, and loads the weights in float32 on both. Measured on this repository's smoke run (Windows venv `dimer-next16`, `CUDA_VISIBLE_DEVICES=-1`, `device="cpu"` passed explicitly, Intel Core Ultra 9 275HX): loading and digest-verifying the 894 MB snapshot took 4.55 s; `translate English to German: The house is wonderful.` (11 input tokens) generated 5 tokens in 0.29 s greedy and 0.21 s with `num_beams=4`, both yielding `Das Haus ist wunderbar.`; a 94-token `summarize: ` input (`max_new_tokens=48`) generated 40 tokens in 0.83 s and stopped at `</s>` with a two-sentence output. The same three calls through the `t5-small-text2text-pipeline` sibling on the same CPU took 0.16 s, 0.31 s and 0.09 s (T5-Base is 1.8×, 2.7× and 2.3× slower) after a 3.89 s load of its 244 MB snapshot; per generated token the summarisation step cost 21 ms here against 6.5 ms for T5-Small. CUDA and the Hub-download path were not executed. Data environment: inputs are assumed to be well-formed English prose of the kind found on the filtered web (news, encyclopaedic, expository text) and, for translation, single sentences; summaries of text much shorter than a paragraph degenerate into near-copies, inputs above 512 tokens are refused, and text far from web English (code, tables, transcripts, other languages) yields output the pipeline cannot flag as degraded.

#### Metrics

###### Performance Measures

The pipeline reports no performance measure, and ships no metric helper. Each result carries only descriptive fields: `text`, `generated_tokens`, `input_tokens`, `stopped_by` (`eos` or `max_new_tokens`) and `known_prefix`. Summarisation and translation quality are conventionally scored with ROUGE-1/2/L and BLEU (or chrF) against human reference outputs; those references do not exist for an arbitrary input, and a reference-free proxy (length ratio, copy rate) would say nothing about faithfulness, so none is manufactured here. A caller who needs a number must supply references for a held-out sample of their own inputs and a scorer; `stopped_by` tells them which outputs hit the token ceiling and should be excluded or re-run with a larger `max_new_tokens` before scoring. The public `evaluation_report()` helper makes that absence machine-readable rather than silent: it always returns a report whose `verdict` is `not-measurable` with an empty `metrics` list, and whose `needs` field names the reference outputs and caller-side ROUGE/BLEU scorer that would be required to score this model. Upstream reports per-task scores for T5-Base in Table 14 of the paper (arXiv:1910.10683); this pipeline has not reproduced any of them and reports no quality figure of its own.

###### Decision thresholds

The default decision rule is greedy decoding: at every step the token with the highest logit is emitted (`num_beams=1`, `do_sample=False`), exposed as `DECISION_RULE` and reported in `generation.decision_rule`; this is an implicit argmax threshold with no minimum-probability cut-off, so the model always produces some token. `num_beams` up to 8 replaces the per-step argmax with a beam search over whole sequences. Generation stops at `</s>` or at `max_new_tokens` (default `DEFAULT_MAX_NEW_TOKENS = 64`, ceiling 512). The upstream `task_specific_params` block also carries `min_length`, `length_penalty`, `no_repeat_ngram_size` and `early_stopping` values for each prefix; the pipeline does **not** apply them — `model.generate` reads only the snapshot's `generation_config.json`, which sets nothing but the start, pad and EOS ids — so a caller who wants the upstream summarisation settings (`num_beams=4`, `min_length=30`, `no_repeat_ngram_size=3`) gets only `num_beams` from this interface. No acceptance threshold on output quality is shipped because the model emits no confidence: the trade-off the operator sets is between over-long or repetitive output (raise `num_beams`, lower `max_new_tokens`) and truncated output (raise `max_new_tokens`), judged on their own references.

###### Approaches to uncertainty and variability

This pipeline reports no metric, so there is no estimation procedure or dispersion to state; the upstream Table 14 figures are single evaluations by the upstream authors with no reported interval. Inference is deterministic for a given input, settings, weights, device and library versions: sampling is disabled (`do_sample=False`), dropout is off under `model.eval()`, and no seed is needed — the smoke run produced the same German sentence under greedy and 4-beam decoding. Beam search and greedy decoding can still differ from each other, and float32 kernel differences between CPU and GPU can flip near-tied tokens and change the rest of the sequence from that point on. The pipeline emits no probability or confidence at all: `generated_tokens` and `stopped_by` are counts, not scores. A caller who needs calibrated confidence in an output must compute it themselves (for example, sequence log-probabilities via `transformers` `output_scores`, then a calibration map fit on their own labelled data); nothing in this repository does so.

#### Ethical considerations and biases

###### Data

Upstream discloses that the checkpoint was pre-trained on C4 (the Colossal Clean Crawled Corpus, filtered Common Crawl web text) and Wiki-DPR for the unsupervised objective and on the named GLUE/SuperGLUE datasets for the supervised mixture (pinned upstream README, "Training Data"), with the translation and summarisation corpora behind the four prefixes described in the paper; disclosure stops there — no per-document licensing, consent status or personal-data audit is given for C4, and web crawl text is known to contain names, contact details and other personal data of identifiable people, so the presence of personal data is not ruled out and should be assumed. This repository distributes code, tests and documentation; the 892 MB `model.safetensors` and the tokenizer files are git-ignored and staged locally under `weights/t5-base/` with a manifest, and no sample data is shipped. The operator must audit the text they submit for personal, confidential or proprietary content; the pipeline performs no such check and keeps no log.

###### Human Life

The pipeline is not intended for decisions in health, safety, criminal justice, employment, credit, housing or any other domain central to human life, and it has not been validated or certified for any of them by anyone. Its only validation is the offline unit suite (13 tests) and the CPU smoke run recorded in this repository. Where a sensitive use is foreseeable — summarising clinical notes, translating a legal notice, condensing a job application — it is admissible only with a human reading the generated text against the source before it is used, an independent evaluation on representative documents with references, and whatever regulatory clearance the domain requires; a T5-Base summary or translation must never be the record of what a source document said.

###### Mitigations

Implemented and inspectable in `src/t5_base_text2text_pipeline/pipeline.py`: (1) supply chain — `MODEL_REVISION` is a 40-hex commit; `stage_missing_files` refuses a manifest whose `modelId`/`revision` differ from the constants and fetches absent files only with `allow_download=True`, one file at a time at that revision; `verify_snapshot` re-hashes every file in `weights/t5-base/dimer-base-manifest.json` and raises on the first size or SHA-256 mismatch before any weight is loaded; loading is `local_files_only=True` from the verified directory, or from the Hub with `revision=MODEL_REVISION` only when explicitly allowed; `trust_remote_code=False` always. (2) Input integrity — `_validate` rejects non-`str`, empty, over-length (`MAX_TEXT_CHARS`) and over-token (`MAX_INPUT_TOKENS`, counted with the real tokenizer) inputs and out-of-range `max_new_tokens`/`num_beams` before the model runs. The public `validate_inputs()` helper applies the same checks through the same private `_check_inputs()` function and returns an input manifest (schema, ceilings, per-input observations including `known_prefix`, verdict, findings), so a caller can record exactly what was accepted or rejected without duplicating the validation logic; the token ceiling stays inside the pipeline because it needs the loaded tokenizer. (3) Reproducibility — exact `==` dependency pins, `model.eval()`, float32, deterministic decoding, and `model_id`/`model_revision`/`generation` in every result. (4) Refusals — no sampling, no batching, no prefix injection, no training or fine-tuning API; a missing snapshot with `allow_download=False` raises `FileNotFoundError`. No statistical mitigation (data re-balancing, output filtering) is applied because the pipeline does not train and ships no content filter.

###### Risks and harms

Fabrication: the decoder produces fluent text whether or not it is entailed by the input — a translation can add or drop a negation, a summary can state a fact the source never made — and the reader who trusts it bears the harm; likelihood rises with input length and with distance from web-English news prose. Prefix misuse: an unrecognised prefix returns something rather than an error (`known_prefix` is `None`, not a refusal), so a caller who mistypes a prefix gets plausible-looking garbage. Bias amplification: C4's filtering and composition (Dodge et al., 2021) and gendered target languages mean summaries and translations can default to majority phrasing and masculine forms; data subjects and third parties bear that harm when the text is published. Automation bias: fluent output is checked less carefully than a rough one. Silent quality loss: text near the 512-token limit is summarised from a lopsided reading of the document, and `max_new_tokens` truncation cuts mid-sentence (the T5-Small sibling's smoke summary ran to the ceiling; T5-Base's stopped at `</s>` after 40 of 48 allowed tokens, but a shorter `max_new_tokens` would have cut it). Data exposure: submitted text is processed in memory only, but the operator's surrounding system may log it. Magnitude ranges from an awkward sentence to a materially false statement attributed to a source.

###### Use cases

The pipeline must not be used for surveillance, profiling or social scoring — for example summarising intercepted or scraped personal communications to characterise an individual — nor to generate text that is presented as a person's own words or as a faithful record of a document without disclosure. It must not support unlawful discrimination in employment, housing, credit, insurance, education or healthcare access, nor deceptive, manipulative or predatory applications such as mass-producing misleading summaries of public documents or fabricated "translations" of statements. Any use that violates the Apache-2.0 terms of the upstream weights or the DIMER deployment terms is prohibited. The developers identify no further prohibited use beyond these because the model's capability is limited to short prefixed seq2seq text.

## Immutable provenance

- Model: `google-t5/t5-base`
- Revision: `a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1`
- Snapshot manifest: `weights/t5-base/dimer-base-manifest.json`, 6 files (no `tokenizer_config.json` upstream at this revision), `totalBytes` 893837231
- `model.safetensors` SHA-256: `a90903540cc02cbeb7ff9f823f1a80eb778c7e22426a0e620b01c77a5ec8f5b4` (891646390 bytes)
- `config.json` SHA-256: `46dd7cb62d29c81fb551e0ef1ea274c24a46ba441eeb948897706252933df033` (1208 bytes)
- `spiece.model` SHA-256: `d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86` (791656 bytes; byte-identical to the `t5-small` snapshot's)
- Weight format: SafeTensors; loader `T5ForConditionalGeneration.from_pretrained(<verified dir>, dtype=torch.float32, local_files_only=True, trust_remote_code=False)` with `T5TokenizerFast` from the same directory (built from `tokenizer.json`; no `tokenizer_config.json` in this snapshot)

## Input/output contract

- `T5BaseText2TextPipeline.from_pretrained(device=None, weights_dir=None, allow_download=False)`
- `generate(text, *, max_new_tokens=64, num_beams=1)` — `text`: one `str`, non-empty, ≤ 20000 chars, ≤ 512 SentencePiece tokens incl. `</s>`, beginning with the caller's own task prefix; `max_new_tokens` 1–512; `num_beams` 1–8. Returns `{"text", "generated_tokens", "input_tokens", "stopped_by", "known_prefix", "generation": {"max_new_tokens", "num_beams", "do_sample", "decision_rule"}, "device", "source", "model_id", "model_revision"}`.
- `TASK_PREFIXES` — the four upstream prefixes from `config.json`; informational, never applied.
- `verify_snapshot(path=None)` — returns the manifest dict with `path`; raises `FileNotFoundError` / `ValueError`.
- `stage_missing_files(path=None, *, allow_download=False, downloader=None)` — returns the list of manifest entries fetched.
- No metric helper: ROUGE/BLEU need references the caller must supply.

## Runtime

- Pins: `torch==2.14.0`, `transformers==4.57.6`, `tokenizers==0.22.2`, `sentencepiece==0.2.2`, `huggingface-hub==0.36.2`, `safetensors==0.8.0`, `numpy==2.5.3`; Python 3.12.
- Precision: float32 on both CPU and CUDA; decoding greedy by default, beam search on request, sampling disabled; upstream `task_specific_params` (min_length, length_penalty, no_repeat_ngram_size) not applied.
- Measured (Windows venv `dimer-next16`, torch 2.14.0+cu130 build, `CUDA_VISIBLE_DEVICES=-1`, `HF_HUB_OFFLINE=1`): device `cpu`, source `local-snapshot`, load + verify 4.55 s; translate 11 → 5 tokens in 0.29 s (greedy) / 0.21 s (4 beams), output `Das Haus ist wunderbar.`; summarise 94 → 40 tokens in 0.83 s, stopped by `eos`; total 5.88 s; no loader warnings.
- Tests: `pytest -q -o addopts= tests` — 13 passed, offline, no weights required; `ruff check src tests` clean.

## References

- Raffel, Shazeer, Roberts, Lee, Narang, Matena, Zhou, Li, Liu. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. JMLR 21(140):1–67, 2020. https://arxiv.org/abs/1910.10683
- Dodge, Sap, Marasović, Agnew, Ilharco, Groeneveld, Mitchell, Gardner. Documenting Large Webtext Corpora: A Case Study on the Colossal Clean Crawled Corpus. EMNLP 2021. https://arxiv.org/abs/2104.08758
- Upstream code: https://github.com/google-research/text-to-text-transfer-transformer
- Upstream card: https://huggingface.co/google-t5/t5-base
