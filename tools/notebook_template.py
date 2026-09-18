"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules (pipeline.py, samples.py, metrics.py), and the model pin/stage/verify cells are produced by
the generator from repository sources so they cannot drift from the package.

This template configures an E2E text-to-text workflow: the pinned T5-base snapshot is digest-verified
and loaded, a digest-pinned real corpus (SciTLDR abstracts paired with their paper titles) is fetched,
validated and split, two prefixed inputs are generated through the inference contract, the frozen model is
scored under a **new task prefix** it was never trained on beside the Lead-1 baseline and its own
`summarize: ` prefix, a bounded fine-tuning of the last decoder blocks teaches the prefix in the kernel,
the held-out split is scored again, and the adapter is exported and reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "t5_base_text2text_pipeline",
    "repo_name": "t5-base-text2text-pipeline",
    "stem": "t5_base_text2text",
    "notebook_name": "t5_base_text2text_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies, stages and digest-verifies the "
        "pinned T5-base snapshot (safetensors, 892 MB), fetches the three digest-pinned SciTLDR-A files from the project "
        "repository (5.5 MB, no credential), pairs abstracts with their paper titles and draws 300 / 50 / 100 training, "
        "validation and test records from the release's own paper-disjoint members, generates from two prefixed inputs "
        "through the inference contract with an input manifest and a rejection probe, scores the frozen model on the test "
        "abstracts under the new prefix `paper title: ` with ROUGE-1/2/L beside the Lead-1 baseline and the trained "
        "`summarize: ` prefix, runs a bounded fine-tuning of the last four decoder blocks that teaches the prefix with "
        "validation-ROUGE-L epoch selection, scores the held-out split again, generates titles for new abstracts with the "
        "adapted model, exports the adapter as safetensors with a manifest, and reloads that artifact into a fresh pipeline "
        "to verify output parity. The default path needs no repository clone, no DIMER worker or service, no credential, no "
        "upload dialog and no configuration edit (NOTEBOOK_SPEC 2.0 §5). On CPU the whole path takes about eight minutes of "
        "model time after the downloads; a CUDA runtime is used automatically when present."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to supply your own "
        "input–output pairs as a CSV (columns `id`, `source`, `target`), a JSON array or a JSONL file of `{{id, source, "
        "target}}` or `{{id, source, targets: [...]}}` records — sources **without** a prefix; set `PREFIX` to the task name "
        "you want to teach. They pass through the same validation, seeded source-disjoint split, baselines, fine-tuning, "
        "held-out evaluation, inference, artifact export and reload-parity cells as the SciTLDR sample. The expected schema "
        "and the ceilings are stated in the Prerequisites and in Section 4, and uploaded files stay inside this runtime. BYOD "
        "is optional and never part of the default path."
    ),
    "pipeline_class": "T5BaseText2TextPipeline",
    "weights_key": "t5-base",
    "modules": ["pipeline.py", "samples.py", "metrics.py"],
    "entry_module": "pipeline.py",
    "identity_names": {},
    "runtime_imports": ["torch", "transformers"],
    "title": "T5-base — DIMER E2E text-to-text fine-tuning tutorial: teaching a new task prefix (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/t5-base-text2text-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/t5-base-text2text-pipeline/blob/main/tutorials/t5_base_text2text_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-google--t5%2Ft5--base-ffcc4d?style=flat",
            "https://huggingface.co/google-t5/t5-base",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-google--research%2Ftext--to--text--transfer--transformer-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/google-research/text-to-text-transfer-transformer",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-1910.10683-b31b1b.svg", "https://arxiv.org/abs/1910.10683"),
    ],
    "capability": "caller-prefixed text-to-text generation (summarisation and English→German/French/Romanian translation) and bounded supervised fine-tuning of the last decoder blocks that teaches a new task prefix on a referenced corpus, using the pinned `google-t5/t5-base` weights",
    "intro": (
        "T5 casts every task as text-to-text: the caller writes a **task prefix** (`summarize: `, `translate English to "
        "German: `) in front of the input, the 223 M-parameter encoder-decoder reads the prefixed text once, and the "
        "decoder writes the output token by token under **greedy decoding** (`num_beams` 1, the default) or beam search "
        "(`num_beams` up to 8). The pinned checkpoint knows the four prefixes in `TASK_PREFIXES`; the carried module never "
        "prepends one, reports which known prefix an input starts with (`known_prefix`, `None` otherwise), verifies the "
        "snapshot manifest, validates inputs and settings against named ceilings (an input over `MAX_INPUT_TOKENS` is "
        "rejected, not truncated), and returns a fixed output contract with `generated_tokens`, `input_tokens` and "
        "`stopped_by`. **The pipeline emits no score, probability or quality metric** — a generation is free text.\n\n"
        "What this notebook adds to inference is **adaptation to a new prefix**. The dataset is real: SciTLDR-A (Cachola "
        "et al., 2020; Apache-2.0) ships, for 3,229 computer-science papers, the abstract and the paper's title — three "
        "digest-pinned JSON-Lines files fetched from the project repository at a pinned commit. The tutorial pairs each "
        "abstract with its title under the prefix `paper title: `, which the pinned checkpoint has **never seen**: the "
        "frozen model answers it with whatever its trained tasks make of the words (the build record saw `True` and "
        "`False`, scoring ROUGE-L 0.7), and the fine-tuning question is whether a bounded adaptation of the last decoder "
        "blocks teaches the prefix on held-out papers. Three metrics are implemented in the carried `metrics.py` (corpus "
        "**ROUGE-1/2/L** F1, rouge-score-style, not rouge-score-identical) and two reference points frame the result: the "
        "**Lead-1 baseline** (the abstract's first sentence as the title) and the frozen model's own **`summarize: ` "
        "prefix** scored against the titles. Nothing here is a quality claim about your task: it is one seeded split of "
        "one corpus."
    ),
    "learning_objectives": (
        "install the pinned runtime; read what the carried pipeline, dataset and metrics modules guarantee; stage and "
        "digest-verify the immutable upstream snapshot; fetch a digest-pinned referenced corpus and validate and split it "
        "without leakage; generate through the public API with explicit `max_new_tokens`/`num_beams` and read "
        "`known_prefix`, the token counts and `stopped_by` correctly; score the frozen model under a new prefix beside the "
        "Lead-1 baseline and a trained prefix and read why an unknown prefix means nothing until it is taught; run a "
        "bounded fine-tuning with explicit hyperparameters and validation-based epoch selection; evaluate on an independent "
        "test split; generate for new abstracts; and export a safetensors adapter that reloads against the pinned base with "
        "verified parity."
    ),
    "exclusions": (
        "sampling-based or diverse decoding, tasks the checkpoint was not trained on before they are taught here, "
        "multi-document or long-document (chunked) generation, full-model or encoder fine-tuning, classification or scoring "
        "(the sibling `bart-mnli-zero-shot-classification-pipeline` covers zero-shot classification), any faithfulness or "
        "factuality score, and any claim that a SciTLDR title split stands in for your task. The repository exposes none of "
        "these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU (float32) and uses CUDA automatically when available. CPU is adequate: the build record measured 5.5 s to load and digest-verify the 892 MB snapshot, about 0.3–0.6 s per abstract for 4-beam title-length generation (26–64 s for the 100-abstract test split) and about 100 s per training epoch over 300 abstracts plus a 50-abstract validation pass per epoch. The pinned `torch==2.14.0` install and the 892 MB checkpoint are the large downloads of the run.",
        "- **Knowledge:** basic Python; what an encoder-decoder (seq2seq) model is; what a task prefix does in T5 and why an untrained prefix is just words; what greedy and beam decoding do; what ROUGE measures and why it is neither faithfulness nor a human judgement.",
        "- **Data contract:** records are `{{id, source, targets}}` — an input **without** any prefix and one or more reference outputs (`{{id, source, target}}` with a single string is accepted and normalised), the source 1..20,000 characters and, with the prefix, at most 512 BPE tokens at inference, each reference 1..400 characters, ids matching `[A-Za-z0-9_.:-]{{1,64}}` and unique; a dataset needs 8..20,000 records; sources are de-duplicated case-insensitively before splitting so the same input never sits in two splits; the prefix is a short string ending in `: ` (at most 64 characters); during training only, prefixed sources are truncated to 512 and targets to 64 BPE tokens (inference never truncates — it rejects). BYOD accepts CSV, JSON or JSONL in that shape.",
        "- **Validation is structural, not semantic:** nothing checks that a reference is a good output for its source or that the prefix describes the task — a mislabelled corpus is fine-tuned on without complaint.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — an internal document set with its reference outputs is exactly that. The default path uploads nothing.",
        "- **External access (data):** besides the Hub, the default path fetches three pinned objects (`train.jsonl` 3,155,015 bytes, `dev.jsonl` 1,124,865 bytes, `test.jsonl` 1,204,107 bytes; SHA-256 `b222771d…` / `3191fa98…` / `fb42dd6c…`) from `raw.githubusercontent.com` at the pinned `allenai/scitldr` commit over HTTPS, each refused on any mismatch before it is read; SciTLDR is Apache-2.0 (Cachola et al., 2020).",
    ],
    "cells": [
        {
            "md": (
                "## 4. Referenced corpus, validation and split\n\n"
                "`fetch_corpus` downloads the three pinned SciTLDR-A files (or reads them from the cache), refuses a "
                "byte-size or SHA-256 mismatch per file before it is parsed, and `read_corpus` flattens each JSON-Lines "
                "member into records whose `source` is the abstract's sentences joined by a space and whose single "
                "`targets` entry is the paper's title. `build_sample_dataset` keeps abstracts of 200..1,600 characters with "
                "a title, drops repeated abstracts and repeated titles, and draws 300 training records from the `train` "
                "member, 50 validation records from `dev` and 100 test records from `test` by a seeded shuffle — the "
                "release's own paper-disjoint partition. `validate_dataset` then checks every record against the contract, "
                "`check_split_disjoint` asserts no abstract appears in two splits, and the training split is written to "
                "`outputs/{stem}_train.csv` in the shape BYOD expects. `PREFIX` is the task name every later cell "
                "prepends; the default is one the checkpoint has never seen.\n\n"
                "Look for: 1,992 + 619 + 618 raw papers, three digests, splits 300 / 50 / 100, titles of 2..17 words, and "
                "four refusal probes — a duplicate id, an empty reference list, a missing field and a dataset too small to "
                "split — each rejected before `torch` does anything."
            ),
            "code": (
                "import hashlib\n"
                "import io\n"
                "import json\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "SPLIT_SEED = 42  # @param {{type:\"integer\"}}\n"
                "PREFIX = 'paper title: '  # @param {{type:\"string\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    file_name, payload = next(iter(uploaded.items()))\n"
                "    byod_path = Path('work') / file_name\n"
                "    byod_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "    byod_path.write_bytes(payload)\n"
                "    records = load_byod_dataset(byod_path)\n"
                "    splits = split_dataset(records, seed=SPLIT_SEED)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "    raw_papers = {{'byod': len(records)}}\n"
                "else:\n"
                "    corpus = read_corpus(fetch_corpus(cache_dir='weights/scitldr'))\n"
                "    raw_papers = {{name: len(part) for name, part in corpus.items()}}\n"
                "    splits = build_sample_dataset(corpus, seed=SPLIT_SEED)\n"
                "    data_source = f'{{CORPUS_NAME}} ({{CORPUS_RELEASE}}; {{CORPUS_LICENSE}})'\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "dataset_manifests = {{name: validate_dataset(part) for name, part in splits.items()}}\n"
                "disjoint = check_split_disjoint(splits)\n"
                "write_dataset_csv(train_records, 'outputs/{stem}_train.csv')\n"
                "print({{'data_source': data_source, 'prefix': PREFIX, 'prefix_is_trained': known_prefix(PREFIX) is not None, 'raw_papers': raw_papers, 'splits': disjoint, 'file_sha256': {{k: v[2][:12] + '...' for k, v in CORPUS_FILES.items()}}}})\n"
                "for name, manifest in dataset_manifests.items():\n"
                "    print({{name: {{'n': manifest['n_records'], 'unique_sources': manifest['unique_sources'], 'source_chars': manifest['source_chars'], 'target_words': manifest['target_words'], 'digest': manifest['digest'][:16] + '...'}}}})\n"
                "print({{'example': {{'id': train_records[0]['id'], 'source': train_records[0]['source'][:200] + '...', 'targets': train_records[0]['targets']}}}})\n\n"
                "probes = {{\n"
                "    'duplicate id': [{{**r, 'id': 'same'}} for r in train_records[:8]],\n"
                "    'empty reference list': [{{**train_records[0], 'targets': []}}, *train_records[1:8]],\n"
                "    'missing field': [{{'id': r['id'], 'source': r['source']}} for r in train_records[:8]],\n"
                "    'too small': train_records[:3],\n"
                "}}\n"
                "for name, probe in probes.items():\n"
                "    try:\n"
                "        validate_dataset(probe)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                "## 5. Generate through the inference contract\n\n"
                "Before any adaptation, the inference contract is exercised as it always was, on two prefixed inputs "
                "authored in this cell — a German translation and a summary, both under trained prefixes. `validate_inputs` "
                "applies exactly the checks `generate` applies (text type and character ceiling, `max_new_tokens` and "
                "`num_beams` within their ceilings) and returns an input manifest that records each input's `known_prefix`; "
                "the encoder-token ceiling `MAX_INPUT_TOKENS` (512) needs the real tokenizer and is enforced inside "
                "`generate`, which **rejects with a `ValueError` naming the count, never truncates**. An out-of-range "
                "`num_beams` is validated too and its rejection recorded as a finding. `generate` returns the text with "
                "`generated_tokens`, `input_tokens`, `stopped_by` (`eos`, or `max_new_tokens` when the output was cut) and "
                "`known_prefix`, and echoes the settings. **Score semantics:** the pipeline emits **no probability, confidence "
                "or score of any kind**. The third call uses the new prefix on one test abstract so the frozen model's answer "
                "to words it was never trained on is visible before any metric is read."
            ),
            "code": (
                "import time\n\n"
                "GEN_MAX_NEW_TOKENS = 24  # @param {{type:\"integer\"}}\n"
                "NUM_BEAMS = 4  # @param {{type:\"integer\"}}\n\n"
                "GEN = {{'max_new_tokens': GEN_MAX_NEW_TOKENS, 'num_beams': NUM_BEAMS}}\n"
                "texts = ['translate English to German: The house is wonderful.', 'summarize: ' + test_records[0]['source']]\n"
                "item_ids = [f'input{{index:02d}}' for index in range(len(texts))]\n"
                "ceilings = {{'MAX_TEXT_CHARS': MAX_TEXT_CHARS, 'MAX_INPUT_TOKENS': MAX_INPUT_TOKENS, 'MAX_NEW_TOKENS': MAX_NEW_TOKENS, 'MAX_NUM_BEAMS': MAX_NUM_BEAMS, 'DEFAULT_MAX_NEW_TOKENS': DEFAULT_MAX_NEW_TOKENS, 'MAX_PREFIX_CHARS': MAX_PREFIX_CHARS}}\n"
                "print(ceilings)\n"
                "print({{'decision_rule': DECISION_RULE, 'task_prefixes': list(TASK_PREFIXES)}})\n"
                "input_manifest = validate_inputs(texts, max_new_tokens=GEN_MAX_NEW_TOKENS, num_beams=NUM_BEAMS, names=item_ids)\n"
                "try:\n"
                "    validate_inputs(texts, max_new_tokens=GEN_MAX_NEW_TOKENS, num_beams=MAX_NUM_BEAMS + 1)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'num-beams-ceiling-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "results = []\n"
                "for item_id, text in zip(item_ids, texts, strict=True):\n"
                "    started = time.perf_counter()\n"
                "    result = pipe.generate(text, **GEN)\n"
                "    results.append({{'id': item_id, 'input': text, 'seconds': round(time.perf_counter() - started, 3), **result}})\n"
                "    print({{'id': item_id, 'known_prefix': result['known_prefix'], 'input_tokens': result['input_tokens'], 'generated_tokens': result['generated_tokens'], 'stopped_by': result['stopped_by'], 'seconds': results[-1]['seconds'], 'text': result['text']}})\n"
                "checks = {{\n"
                "    'one_result_per_input': len(results) == len(texts),\n"
                "    'generated_within_ceiling': all(r['generated_tokens'] <= GEN_MAX_NEW_TOKENS for r in results),\n"
                "    'input_within_ceiling': all(r['input_tokens'] <= MAX_INPUT_TOKENS for r in results),\n"
                "    'every_input_has_known_prefix': all(r['known_prefix'] is not None for r in results),\n"
                "    'settings_echoed': all(r['generation']['max_new_tokens'] == GEN_MAX_NEW_TOKENS and r['generation']['num_beams'] == NUM_BEAMS and r['generation']['do_sample'] is False for r in results),\n"
                "}}\n"
                "if not all(checks.values()):\n"
                "    raise RuntimeError(f'generate output failed a sanity check: {{checks}}')\n"
                "new_prefix_probe = pipe.generate(PREFIX + test_records[0]['source'], **GEN)\n"
                "print({{'checks': checks, 'findings': len(input_manifest['findings']), 'no_score': 'the pipeline emits no probability or quality score'}})\n"
                "print({{'frozen_model_under_new_prefix': {{'prefix': PREFIX, 'known_prefix': new_prefix_probe['known_prefix'], 'text': new_prefix_probe['text'], 'reference_title': test_records[0]['targets'][0]}}}})"
            ),
        },
        {
            "md": (
                "## 6. Baselines and the frozen model's score on the test split\n\n"
                "Three numbers frame the adaptation, all under the settings of Section 5. The **Lead-1 baseline** submits "
                "the first sentence of each abstract as its title: what a system that does no modelling gets. The **frozen "
                "model under the new prefix** generates from `paper title: ` + abstract — an unknown prefix is just words, "
                "so expect a near-zero score and one-word outputs. The **frozen model under `summarize: `**, its closest "
                "trained task, is scored against the same titles as a second reference point: a news-style summary of the "
                "abstract overlaps the title more than nothing, but it is not a title. All three use corpus **ROUGE-1**, "
                "**ROUGE-2** and **ROUGE-L** F1 (rouge-score-style, not rouge-score-identical); read `mean_output_words` "
                "beside every score. About a minute and a half on CPU."
            ),
            "code": (
                "baseline_lead1 = lead_baseline(test_records, n_sentences=1)\n"
                "print({{'lead1_baseline': {{'rouge1': round(baseline_lead1['rouge1'], 2), 'rouge2': round(baseline_lead1['rouge2'], 2), 'rougeL': round(baseline_lead1['rougeL'], 2), 'mean_output_words': round(baseline_lead1['mean_output_words'], 1), 'n': baseline_lead1['n']}}}})\n"
                "t0 = time.perf_counter()\n"
                "frozen_test = pipe.evaluate(test_records, prefix=PREFIX, **GEN)\n"
                "print({{'frozen_model_new_prefix_test': {{'rouge1': round(frozen_test['rouge1'], 2), 'rouge2': round(frozen_test['rouge2'], 2), 'rougeL': round(frozen_test['rougeL'], 2), 'mean_output_words': round(frozen_test['mean_output_words'], 1), 'mean_reference_words': round(frozen_test['mean_reference_words'], 1), 'hit_token_ceiling': frozen_test['hit_token_ceiling'], 'known_prefix': frozen_test['known_prefix'], 'n': frozen_test['n'], 'verdict': frozen_test['verdict']}}, 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "t0 = time.perf_counter()\n"
                "frozen_summarize = pipe.evaluate(test_records, prefix='summarize: ', **GEN)\n"
                "print({{'frozen_model_summarize_test': {{'rouge1': round(frozen_summarize['rouge1'], 2), 'rouge2': round(frozen_summarize['rouge2'], 2), 'rougeL': round(frozen_summarize['rougeL'], 2), 'mean_output_words': round(frozen_summarize['mean_output_words'], 1), 'known_prefix': frozen_summarize['known_prefix']}}, 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "print({{'definitions': frozen_test['definitions']}})\n"
                "for record in test_records[:2]:\n"
                "    print({{'frozen_new_prefix': pipe.generate(PREFIX + record['source'], **GEN)['text'], 'frozen_summarize': pipe.generate('summarize: ' + record['source'], **GEN)['text'], 'reference': record['targets'][0]}})\n"
                "assert baseline_lead1['rouge1'] > 0.0"
            ),
        },
        {
            "md": (
                "## 7. Bounded fine-tuning that teaches the prefix\n\n"
                "`pipe.adapt` prepends `PREFIX` to every training abstract and trains only the last `TRAINABLE_DECODER_LAYERS` "
                "decoder blocks — four by default, 37,757,952 of 222,903,552 parameters; the encoder, the shared embeddings, "
                "the tied output projection and the earlier decoder blocks stay frozen — with teacher-forced cross-entropy on "
                "the title, AdamW at a fixed learning rate, gradient clipping at 1.0, seeded shuffling and no scheduler. "
                "Prefixed sources are truncated to 512 and targets to 64 BPE tokens **during training only**. Epoch 0 records "
                "the frozen model's validation ROUGE under the same prefix and settings; every epoch is scored the same way, "
                "and the epoch with the highest validation ROUGE-L is kept.\n\n"
                "Watch validation ROUGE-L go from near zero to the thirties and `mean_output_words` settle near title length "
                "within two epochs (about 100 s of training plus a validation pass per epoch on CPU). The build record's "
                "counter-examples: two decoder blocks at a lower learning rate reached ROUGE-L 16 in two epochs and still "
                "produced `False` for some abstracts — the new prefix needs more capacity than an in-domain tweak."
            ),
            "code": (
                "EPOCHS = 2  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 5e-4  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 8  # @param {{type:\"integer\"}}\n"
                "TRAINABLE_DECODER_LAYERS = 4  # @param {{type:\"integer\"}}\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4)}}\n"
                "    if entry.get('val'):\n"
                "        row['val_rouge1'] = round(entry['val']['rouge1'], 2)\n"
                "        row['val_rouge2'] = round(entry['val']['rouge2'], 2)\n"
                "        row['val_rougeL'] = round(entry['val']['rougeL'], 2)\n"
                "        row['val_mean_output_words'] = round(entry['val']['mean_output_words'], 1)\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, prefix=PREFIX, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, trainable_decoder_layers=TRAINABLE_DECODER_LAYERS, eval_generation=GEN, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'prefix': adapt_result['prefix'], 'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation\n\n"
                "The test split was never used for training or epoch selection, and no abstract in it appears in the "
                "training or validation splits. The adapted model is scored under the new prefix exactly as the frozen model "
                "was in Section 6, and the four numbers are put side by side. Look for a ROUGE-L in the high twenties or "
                "thirties — above the Lead-1 baseline and above the trained `summarize: ` prefix — with `mean_output_words` "
                "near the reference length; the cell asserts the adapted ROUGE-L is above the frozen ROUGE-L under the same "
                "prefix. One hundred abstracts from one seeded split of one corpus give no dispersion estimate; the deltas are "
                "sample-sanity evidence that the adaptation contract works, not a benchmark, and a taught prefix on paper "
                "abstracts says nothing about your task until you measure it there. The adapter changes only the last "
                "decoder blocks, so the trained prefixes are affected too; Section 9 checks the German translation once "
                "more after adaptation."
            ),
            "code": (
                "adapted_test = pipe.evaluate(test_records, prefix=PREFIX, **GEN)\n"
                "adapted_val = pipe.evaluate(val_records, prefix=PREFIX, **GEN)\n"
                "comparison = {{\n"
                "    metric: {{'lead1': round(baseline_lead1[metric], 2), 'frozen_summarize': round(frozen_summarize[metric], 2), 'frozen_new_prefix': round(frozen_test[metric], 2), 'adapted': round(adapted_test[metric], 2)}}\n"
                "    for metric in ('rouge1', 'rouge2', 'rougeL', 'mean_output_words')\n"
                "}}\n"
                "comparison['delta_vs_frozen'] = {{metric: round(adapted_test[metric] - frozen_test[metric], 2) for metric in ('rouge1', 'rouge2', 'rougeL')}}\n"
                "for metric, row in comparison.items():\n"
                "    print({{metric: row}})\n"
                "for record in test_records[:2]:\n"
                "    print({{'adapted': pipe.generate(PREFIX + record['source'], **GEN)['text'], 'reference': record['targets'][0]}})\n"
                "evaluation_report_payload = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'data_source': data_source,\n"
                "    'prefix': PREFIX,\n"
                "    'dataset_digests': {{name: manifest['digest'] for name, manifest in dataset_manifests.items()}},\n"
                "    'splits': disjoint,\n"
                "    'generation': frozen_test['generation'],\n"
                "    'baselines': {{'lead1': baseline_lead1, 'frozen_summarize_prefix': frozen_summarize}},\n"
                "    'frozen_test': frozen_test,\n"
                "    'validation_metrics': adapted_val,\n"
                "    'test_metrics': adapted_test,\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report_payload, f, indent=2, ensure_ascii=False)\n"
                "assert adapted_test['rougeL'] > frozen_test['rougeL']\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json'}})"
            ),
        },
        {
            "md": (
                "## 9. Generate for new abstracts, export the adapter and reload it\n\n"
                "Four abstracts that were in none of the splits are given titles by the adapted model through the same "
                "`generate` contract as Section 5 and scored with `pipe.evaluate` (a `measured-small-sample` verdict, because "
                "four documents carry no dispersion estimate); the single-document `evaluation_report` helper — the "
                "inference-stage helper, which now scores supplied references as `sample-sanity` — is written for the first "
                "of them, and the German translation from Section 5 is generated once more so the effect of the adapter on a "
                "trained prefix is visible.\n\n"
                "`pipe.save_artifact` writes the trained tensors — the last four decoder blocks, about 151 MB — as "
                "`adapter.safetensors`, with a `manifest.json` recording the artifact format, the base model id and revision, "
                "the digest of the base `model.safetensors`, the prefix it was trained for, the tensor names, the file size "
                "and SHA-256, the training configuration and the epoch history (OUT8). `T5BaseText2TextPipeline.from_artifact` "
                "re-verifies the base snapshot, checks the artifact manifest and digest **before** deserialising, refuses any "
                "tensor that is not an adaptable decoder tensor, and overlays the tensors onto a freshly loaded base — a new "
                "object from files, not the in-memory model (VER2). The cell asserts identical outputs (VER4)."
            ),
            "code": (
                "import csv\n"
                "import shutil\n\n"
                "if USE_BYOD:\n"
                "    new_records = [{{**r, 'id': f'new-{{i:02d}}'}} for i, r in enumerate(test_records[:4])]\n"
                "else:\n"
                "    used = {{r['source'].lower() for part in splits.values() for r in part}}\n"
                "    new_records = [{{**r, 'id': f'new-{{i:02d}}'}} for i, r in enumerate([r for r in filter_records(corpus['dev']) if r['source'].lower() not in used][:4])]\n"
                "new_metrics = pipe.evaluate(new_records, prefix=PREFIX, **GEN)\n"
                "new_results = []\n"
                "for record in new_records:\n"
                "    item = pipe.generate(PREFIX + record['source'], **GEN)\n"
                "    new_results.append({{'id': record['id'], 'output': item['text'], 'reference': record['targets'][0], 'generated_tokens': item['generated_tokens'], 'input_tokens': item['input_tokens'], 'stopped_by': item['stopped_by']}})\n"
                "    print({{k: new_results[-1][k] for k in ('id', 'output', 'reference')}})\n"
                "single_report = evaluation_report(pipe.generate(PREFIX + new_records[0]['source'], **GEN), new_records[0]['targets'], sample_kind='one unseen SciTLDR abstract' if not USE_BYOD else 'one BYOD test record')\n"
                "german_after = pipe.generate(texts[0], **GEN)\n"
                "print({{'new_abstracts': {{'n': new_metrics['n'], 'rouge1': round(new_metrics['rouge1'], 2), 'rougeL': round(new_metrics['rougeL'], 2), 'verdict': new_metrics['verdict']}}, 'single_document_report_verdict': single_report['verdict'], 'german_translation_before_after': [results[0]['text'], german_after['text']]}})\n"
                "with open('outputs/{stem}_generations.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.DictWriter(handle, fieldnames=list(new_results[0]))\n"
                "    writer.writeheader()\n"
                "    writer.writerows(new_results)\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'prefix': artifact_manifest['adapter']['prefix'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...'}})\n\n"
                "reloaded = T5BaseText2TextPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "before = [pipe.generate(PREFIX + r['source'], **GEN)['text'] for r in test_records[:6]]\n"
                "after = [reloaded.generate(PREFIX + r['source'], **GEN)['text'] for r in test_records[:6]]\n"
                "parity = {{'identical_outputs': sum(a == b for a, b in zip(before, after, strict=True)), 'of': len(before)}}\n"
                "print({{'reload_parity': parity, 'reloaded_prefix': reloaded.adapter['prefix'], 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['identical_outputs'] == parity['of']\n\n"
                "weight_entry = next(entry for entry in snapshot['files'] if entry['path'] == WEIGHT_FILE)\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'snapshot': {{'path': str(WEIGHTS_DIR), 'files': len(snapshot['files']), 'total_bytes': snapshot.get('totalBytes'), 'fetched_this_run': fetched, 'weight_file': WEIGHT_FILE, 'weight_format': 'safetensors, digest-verified', 'weight_sha256': weight_entry['sha256']}},\n"
                "    'data_source': data_source,\n"
                "    'prefix': PREFIX,\n"
                "    'corpus': {{'name': CORPUS_NAME, 'release': CORPUS_RELEASE, 'base_url': CORPUS_BASE_URL, 'files': {{k: {{'name': v[0], 'bytes': v[1], 'sha256': v[2]}} for k, v in CORPUS_FILES.items()}}, 'license': CORPUS_LICENSE}},\n"
                "    'inference_contract': {{'input_manifest': input_manifest, 'sanity_checks': checks, 'items': [{{k: r[k] for k in ('id', 'input', 'text', 'known_prefix', 'input_tokens', 'generated_tokens', 'stopped_by', 'seconds')}} for r in results], 'frozen_new_prefix_probe': new_prefix_probe['text'], 'german_after_adaptation': german_after['text']}},\n"
                "    'comparison': comparison,\n"
                "    'new_abstracts': new_metrics,\n"
                "    'single_document_report': single_report,\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes'], 'tensors': len(artifact_manifest['tensors'])}},\n"
                "    'reload_parity': parity,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'device': pipe.device, 'dtype': 'float32', 'source': pipe.source}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(result_payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "An unknown prefix means nothing to the frozen model — `paper title: ` produced one-word answers scoring near zero, "
        "while its trained `summarize: ` prefix overlapped the titles about as much as the abstract's first sentence — and "
        "a bounded fine-tuning of the last four decoder blocks on 300 abstract–title pairs teaches the prefix in a few "
        "minutes on CPU, lifting held-out ROUGE-L above both reference points with outputs at title length, with a 151 MB "
        "adapter that reloads to identical outputs. That is the claim: the adaptation contract can teach a new task prefix "
        "end to end on a real referenced corpus, and the numbers it produces are read against a Lead baseline and the "
        "frozen model's nearest trained task rather than in isolation.\n\n"
        "The test split is 100 abstracts from one seeded split of one corpus, the metrics are three n-gram overlap scores "
        "(own implementation, not rouge-score-identical, and none a judgement of whether a title is good), and titles are "
        "short and formulaic. So a gain here says the contract works, not that the adapted model writes good titles for "
        "your papers, that it handles long or technical inputs, or that its outputs are faithful. The adapter changes the "
        "last decoder blocks, which every prefix shares, so the trained tasks are affected too — Section 9 shows the German "
        "translation before and after — and nothing here measures that beyond one sentence.\n\n"
        "Three things to carry to real data. **References first:** the Lead baseline and the frozen model under its nearest "
        "trained prefix on *your* references are the numbers to read before any adapted one. **Leakage:** de-duplicate "
        "sources across splits (the contract does this case-insensitively) and split by document collection or author when "
        "your pairs come from one. **Ceilings:** prefixed inputs over `MAX_INPUT_TOKENS` are refused at inference and "
        "truncated to 512 tokens only during training — long-document tasks are out of scope.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone "
        "notebook, can acquire and digest-verify the pinned model snapshot, fetch and digest-verify a real referenced corpus, "
        "validate the demonstrated dataset contract without leakage, execute the inference contract and a bounded "
        "fine-tuning that teaches a new prefix, evaluate against a trivial baseline and the frozen model on an independent "
        "split, and emit the shown machine-readable artifacts — without the repository being reachable. It does **not** "
        "establish benchmark superiority, output quality or faithfulness on any other task, a usable acceptance threshold, "
        "or production fitness.\n\n"
        "**Optional experiments (they do not affect the default path):** set `TRAINABLE_DECODER_LAYERS = 2` and watch the "
        "prefix fail to take in two epochs; set `PREFIX = 'summarize: '` to re-target a trained prefix to titles and compare "
        "the frozen and adapted scores; set `NUM_BEAMS = 1` and read the greedy scores; or bring your own input–output pairs "
        "and prefix through BYOD and read the Lead baseline before the adapted number.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/t5-base-text2text-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/t5-base-text2text-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/t5-base-text2text-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/google-research/text-to-text-transfer-transformer\n"
        "- Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (Raffel et al., JMLR 2020): https://arxiv.org/abs/1910.10683\n"
        "- TLDR: Extreme Summarization of Scientific Documents (Cachola et al., EMNLP Findings 2020; SciTLDR, Apache-2.0): https://arxiv.org/abs/2004.15011\n"
        "- ROUGE: A Package for Automatic Evaluation of Summaries (Lin, 2004): https://aclanthology.org/W04-1013\n"
        "- DIMER Notebook Specification 2.0 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)"
    ),
}
