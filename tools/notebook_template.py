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
    "isolated_runtime": True,
    "infrastructure_labels": True,
    # The fleet's uv isolated-environment mechanism (bioclip2-biodiversity-pipeline): managed CPython, a size- and
    # SHA-256-verified uv wheel, and a lock compiled from the pyproject pins with
    # `uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes
    # --only-binary :all: -o tutorials/requirements-colab.lock.txt`.
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    # T5B-m2: the default path also reaches raw.githubusercontent.com for the SciTLDR files, so the generated access
    # bullet names both hosts instead of "the Hugging Face Hub only".
    "external_access": (
        "the Hugging Face Hub, to fetch the pinned `{MODEL_ID}` snapshot (~{total_mb:.0f} MB in total) at revision "
        "`{MODEL_REVISION:.12}…`, and `raw.githubusercontent.com`, for the three pinned SciTLDR-A files named in the data "
        "bullet above. No credentials are required; nothing is installed from this repository."
    ),
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime builds an isolated environment from the hash-locked pins (nothing is "
        "installed into the notebook's own Python, so no restart is needed and Run all completes in one pass), stages and digest-verifies the "
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
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and either set `BYOD_PATH` to the file in the "
        "runtime (Colab, Kaggle or Jupyter) or leave it empty to upload one file in Colab, then choose **Run after** from that "
        "cell, to supply your own input–output pairs as a CSV (columns `id`, `source`, `target`; UTF-8, with or without a "
        "byte-order mark), a JSON array or a JSONL file of `{id, source, target}` or `{id, source, targets: [...]}` records — "
        "sources **without** a prefix (a source that already starts with a trained prefix or with `PREFIX` is refused); set "
        "`PREFIX` to the task name you want to teach. After the split the training split needs eight records and the "
        "validation split one, so the **effective minimum is 12 distinct sources** (split 8 / 2 / 2). Section 4 first puts the "
        "model back to the pinned base, so Sections 5 and 6 score the frozen model. They pass through the same validation, seeded source-disjoint split, baselines, fine-tuning, "
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
    "guided": {
        "opening": [
            (
                "**Who this notebook is for.** A learner who knows basic Python, has used Colab or Jupyter, and wants to see how a text-to-text model is steered by a task prefix, what happens when the prefix is one it has never seen, and how to teach it a new one with a small referenced set without fooling themselves. No prior experience with T5 or fine-tuning is assumed; each term is explained where it first matters and again in the **Glossary** at the end. CPU is adequate (about eight minutes of model time); a GPU runtime is faster.\n\n**Input → Model → Output.**\n\n| | Generation | Scored titling | Bounded fine-tuning |\n|---|---|---|---|\n| Input | `prefix` + text (at most 512 tokens) | `paper title: ` + an abstract | abstract–title pairs (300 training and 50 validation in the sample) |\n| Model | T5-base encoder-decoder, beam search with explicit `max_new_tokens` and `num_beams` | the same model | the last four decoder blocks, trained with teacher-forced cross-entropy; everything else frozen |\n| Output | the generated text, token counts and `stopped_by` — no score | the title, scored by ROUGE-1/2/L against the paper's title | a 151 MB safetensors adapter, and held-out ROUGE beside a Lead-1 baseline |\n\n**How to use this notebook.** Choose a runtime (CPU works; **Runtime → Change runtime type → T4 GPU** is faster), then **Runtime → Run all**. Run all completes in one pass: Section 1 installs nothing into the notebook's own Python, so no restart is needed. Sections 1–3 are **infrastructure** — the isolated environment, the carried package and the model snapshot — and their cells are collapsed; you may run them without studying them. The learning path starts in Section 4. Form fields (`# @param`) are the only values meant to be edited, and the defaults reproduce the recorded run. Before each principal result the notebook asks you to **Predict**; after it come **What to notice** and a collapsible **Check your reasoning** with a worked answer from the recorded run (the Kaggle T4 run of 19 September 2026). Section 10 is a **change-one-thing experiment**, off by default. **Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. Writing your predictions down is optional.\n\n**Roadmap:** 1–3 infrastructure → 4 the referenced corpus, validation and the split *(evaluation practice)* → 5 the inference contract *(core concept: prefixes, no score)* → 6 the Lead-1 baseline and the frozen model under a new and a trained prefix *(evaluation practice)* → 7 bounded fine-tuning that teaches the prefix *(core concept)* → 8 held-out evaluation → 9 new abstracts, export and reload *(engineering)* → 10 change one thing (optional) → conclude."
            )
        ]
    },
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
        "- **Learner:** basic Python and Colab or Jupyter familiarity; no prior experience with T5 or fine-tuning. The notebook explains task prefixes, encoder-decoder generation, beam search, teacher forcing, ROUGE-1/2/L, the Lead-N baseline and the adapter where they are first used; the Glossary repeats them.",
        "- **Runtime:** a fresh supported **Linux x86_64** runtime (Google Colab, Kaggle or Linux Jupyter). Section 1 builds its own Python 3.12.12 environment from a hash-locked list of manylinux wheels, so the kernel's own Python version does not matter and nothing is installed into it. The default path runs on CPU (float32) and uses CUDA automatically when available. CPU is adequate: the build record measured 5.5 s to load and digest-verify the 892 MB snapshot, about 0.3–0.6 s per abstract for 4-beam title-length generation (26–64 s for the 100-abstract test split) and about 100 s per training epoch over 300 abstracts plus a 50-abstract validation pass per epoch. The pinned `torch==2.14.0` install and the 892 MB checkpoint are the large downloads of the run.",
        "- **Knowledge:** basic Python; what an encoder-decoder (seq2seq) model is; what a task prefix does in T5 and why an untrained prefix is just words; what greedy and beam decoding do; what ROUGE measures and why it is neither faithfulness nor a human judgement.",
        "- **Data contract:** records are `{id, source, targets}` — an input **without** any prefix and one or more reference outputs (`{id, source, target}` with a single string is accepted and normalised), the source 1..20,000 characters and, with the prefix, at most 512 BPE tokens at inference, each reference 1..400 characters, ids matching `[A-Za-z0-9_.:-]{1,64}` and unique; a dataset needs 8..20,000 records, and after the split the training split needs eight and the validation split one, so the effective BYOD minimum is **12 distinct sources** (`min_byod_records()` computes it); sources are de-duplicated case-insensitively before splitting so the same input never sits in two splits; the prefix is a short string ending in `: ` (at most 64 characters); during training only, prefixed sources are truncated to 512 and targets to 64 BPE tokens (inference never truncates — it rejects). BYOD accepts CSV, JSON or JSONL in that shape.",
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
                "split — each rejected before `torch` does anything.\n\n"
                "*Evaluation practice.* **Bring your own data (optional):** set `USE_BYOD = True` and either `BYOD_PATH` (the CSV, "
                "JSON or JSONL file, as a path in this runtime — this works on Colab, Kaggle and Jupyter) or leave `BYOD_PATH` "
                "empty to upload exactly one file through the Colab dialog; then choose **Run after** from this cell. This cell "
                "first puts the model back to the pinned base, so Sections 5 and 6 score the frozen model. The effective minimum "
                "is 12 distinct sources, and sources must not carry a prefix.\n\n"
                "**Predict before running:** the three splits come from the release's own train / dev / test files, which are "
                "disjoint by paper. Can the same *title* still appear in two splits?"
            ),
            "code": (
                "import hashlib\n"
                "import io\n"
                "import json\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_PATH = ''  # @param {{type:\"string\"}}\n"
                "SPLIT_SEED = 42  # @param {{type:\"integer\"}}\n"
                "PREFIX = 'paper title: '  # @param {{type:\"string\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if pipe.adapter is not None:\n"
                "    # A re-run after Section 7 (BYOD, a new prefix or split): Sections 5 and 6 must score the frozen model, not the adapted one.\n"
                "    print({{'restored_pinned_base': len(pipe.restore_base()), 'note': 'the adapted weights were removed; Sections 5-6 score the frozen model again'}})\n"
                "if USE_BYOD:\n"
                "    if BYOD_PATH.strip():\n"
                "        byod_path = Path(BYOD_PATH.strip()).expanduser()\n"
                "        if not byod_path.is_file():\n"
                "            raise FileNotFoundError(f'BYOD_PATH {{BYOD_PATH!r}} is not a file (relative paths start at {{Path.cwd()}}): give the .csv, .json or .jsonl file.')\n"
                "        file_name = byod_path.name\n"
                "    else:\n"
                "        try:\n"
                "            from google.colab import files\n"
                "        except ImportError:\n"
                "            raise RuntimeError('USE_BYOD is True but BYOD_PATH is empty, and the upload dialog exists only in Google Colab: on Kaggle or Jupyter put the file in the runtime and set BYOD_PATH to its path.') from None\n"
                "        uploaded = files.upload() or {{}}\n"
                "        if len(uploaded) != 1:\n"
                "            raise ValueError(f'Upload exactly one .csv, .json or .jsonl file (received {{len(uploaded)}}; a cancelled dialog sends none): run this cell again.')\n"
                "        file_name, payload = next(iter(uploaded.items()))\n"
                "        byod_path = Path('work') / Path(file_name).name\n"
                "        byod_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "        byod_path.write_bytes(payload)\n"
                "    records = load_byod_dataset(byod_path, prefix=PREFIX)\n"
                "    splits = split_dataset(records, seed=SPLIT_SEED)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "    raw_papers = {{'byod': len(records), 'duplicate_sources_dropped': len(records) - sum(len(part) for part in splits.values()), 'effective_minimum': min_byod_records()['total']}}\n"
                "    if len(splits['test']) < 20:\n"
                "        print({{'caution': f\"only {{len(splits['test'])}} held-out test records: ROUGE carries no dispersion estimate at this size; add pairs before reading it\"}})\n"
                "else:\n"
                "    corpus = read_corpus(fetch_corpus(cache_dir='weights/scitldr'))\n"
                "    raw_papers = {{name: len(part) for name, part in corpus.items()}}\n"
                "    splits = build_sample_dataset(corpus, seed=SPLIT_SEED)\n"
                "    data_source = f'{{CORPUS_NAME}} ({{CORPUS_RELEASE}}; {{CORPUS_LICENSE}})'\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "# The training split must hold MIN_RECORDS; validation and test only need one record each (split_dataset checks that).\n"
                "dataset_manifests = {{name: validate_dataset(part, min_records=MIN_RECORDS if name == 'train' else 1) for name, part in splits.items()}}\n"
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
                '**What to notice:** the raw paper counts, the three split sizes and digests, the `prefix_is_trained` flag (False for `paper title: `), and the four refusals.\n\n<details><summary>Check your reasoning</summary>Yes. `check_split_disjoint` compares abstracts, and no abstract is shared; but a review probe found one identical title in both the training and the test split — two papers with the same title. It is harmless at this size, and it shows that a leakage check is only as strong as the key it compares: decide which field must not repeat across splits on your own data.</details>'
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
                "to words it was never trained on is visible before any metric is read.\n\n"
                "*Core concept.* T5 learned a handful of tasks, each named by a **prefix** (`summarize: `, `translate English to "
                "German: `, …). The prefix is ordinary text: a prefix it never saw carries no instruction.\n\n"
                "**Predict before running:** what will the frozen model answer when the abstract is preceded by `paper title: `?"
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
                "**What to notice:** the German translation, the summary, `stopped_by`, and the frozen model's answer under the new prefix.\n\n<details><summary>Check your reasoning</summary>A single word. In the recorded run the answer to `paper title: ` + abstract was `True` — the prefix reads like a yes/no question, so the model produced the closest thing it knows. The trained prefixes behaved as expected (the card-pass smoke translated the sentence as `Das Haus ist wunderbar.`). No call returns a score: whether an output is any good is only measured against references, in Section 6.</details>"
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
                "beside every score. About a minute and a half on CPU.\n\n"
                "*Evaluation practice.* **Predict before running:** rank the three — Lead-1, the frozen model under `paper "
                "title: `, the frozen model under `summarize: ` — by ROUGE-L against the paper titles."
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
                "# A reported finding, not an assertion: on your data the first sentence may share no word with the references (a translation task, say).\n"
                "lead1_overlap = 'Lead-1 shares words with the references' if baseline_lead1['rouge1'] > 0.0 else 'Lead-1 shares no unigram with the references: ROUGE against this baseline is uninformative for this task'\n"
                "print({{'lead1_overlap': lead1_overlap}})"
            ),
        },
        {
            "md": (
                '**What to notice:** ROUGE-1/2/L for the three systems and `mean_output_words` beside each.\n\n<details><summary>Check your reasoning</summary>Summarize first, Lead-1 second, the new prefix last. In the recorded run ROUGE-L was 18.99 for `summarize: ` (16.7 words per output), 14.08 for Lead-1 (21.4 words) and 0.71 for `paper title: ` (1.03 words) against titles of about 7.4 words. The new prefix scores near zero because one word cannot overlap a title; read the length beside every ROUGE number.</details>'
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
                "produced `False` for some abstracts — the new prefix needs more capacity than an in-domain tweak.\n\n"
                "*Core concept.* Every call to `pipe.adapt` starts from the **pinned base**: tensors an earlier call changed are "
                "restored first, so epoch 0 is always the frozen model. Re-running this cell with a changed field is a fresh run, "
                "not continued training — but it replaces the default results that Sections 8 and 9 export; to compare a change "
                "side by side, use Section 10.\n\n"
                "**Predict before running:** how many epochs will it take before validation outputs are title-length?"
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
                "settings = {{'epochs': EPOCHS, 'lr': LEARNING_RATE, 'batch_size': BATCH_SIZE, 'trainable_decoder_layers': TRAINABLE_DECODER_LAYERS}}\n"
                "if settings != {{'epochs': 2, 'lr': 5e-4, 'batch_size': 8, 'trainable_decoder_layers': 4}}:\n"
                "    print({{'note': 'changed settings: this run starts again from the pinned base and replaces the default results of Sections 8-9; Section 10 compares a change side by side instead', 'settings': settings}})\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, prefix=PREFIX, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, trainable_decoder_layers=TRAINABLE_DECODER_LAYERS, eval_generation=GEN, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'prefix': adapt_result['prefix'], 'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'started_from': adapt_result['started_from'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                "**What to notice:** epoch 0 (the frozen model under the new prefix), the loss, `val_rougeL` and `val_mean_output_words`.\n\n<details><summary>Check your reasoning</summary>One. In the recorded runs validation ROUGE-L went from about 0.73 at epoch 0 to about 33 after the first epoch and 35 after the second, which was kept, with outputs near title length. Four decoder blocks and a learning rate of 5e-4 are enough to make an unknown prefix mean *write a title*; the build record's two-block variant at a lower rate reached only about 16.</details>"
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation\n\n"
                "The test split was never used for training or epoch selection, and no abstract in it appears in the "
                "training or validation splits. The adapted model is scored under the new prefix exactly as the frozen model "
                "was in Section 6, and the four numbers are put side by side. Look for a ROUGE-L in the high twenties or "
                "thirties — above the Lead-1 baseline and above the trained `summarize: ` prefix — with `mean_output_words` "
                "near the reference length; the cell reports a **verdict** — whether the adapted ROUGE-L `improved`, showed "
                "`no gain` or got `worse` against the frozen ROUGE-L under the same prefix — and records it; a fine-tune that does "
                "not help (on your data a trained prefix may already score well) is a finding, not an error, and Section 9 still "
                "exports, reloads and writes the result. One hundred abstracts from one seeded split of one corpus give no dispersion estimate; the deltas are "
                "sample-sanity evidence that the adaptation contract works, not a benchmark, and a taught prefix on paper "
                "abstracts says nothing about your task until you measure it there. The adapter changes only the last "
                "decoder blocks, so the trained prefixes are affected too; Section 9 checks the German translation once "
                "more after adaptation.\n\n"
                "**Predict before running:** will the adapted model also beat the trained `summarize: ` prefix and Lead-1?"
            ),
            "code": (
                "adapted_test = pipe.evaluate(test_records, prefix=PREFIX, **GEN)\n"
                "adapted_val = pipe.evaluate(val_records, prefix=PREFIX, **GEN)\n"
                "comparison = {{\n"
                "    metric: {{'lead1': round(baseline_lead1[metric], 2), 'frozen_summarize': round(frozen_summarize[metric], 2), 'frozen_new_prefix': round(frozen_test[metric], 2), 'adapted': round(adapted_test[metric], 2)}}\n"
                "    for metric in ('rouge1', 'rouge2', 'rougeL', 'mean_output_words')\n"
                "}}\n"
                "comparison['delta_vs_frozen'] = {{metric: round(adapted_test[metric] - frozen_test[metric], 2) for metric in ('rouge1', 'rouge2', 'rougeL')}}\n"
                "# Reported verdicts, not assertions: a non-improving fine-tune is a result to record, and export and reload still run.\n"
                "delta_rougeL = adapted_test['rougeL'] - frozen_test['rougeL']\n"
                "adaptation_verdict = 'improved' if delta_rougeL > 0 else ('no gain' if delta_rougeL == 0 else 'worse')\n"
                "comparison['verdicts'] = {{'adapted_vs_frozen_rougeL': adaptation_verdict, 'lead1_overlap': lead1_overlap}}\n"
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
                "print({{'report': 'outputs/{stem}_evaluation_report.json', 'verdicts': comparison['verdicts']}})"
            ),
        },
        {
            "md": (
                '**What to notice:** the four-way table, `delta_vs_frozen`, the `verdicts` row and `mean_output_words`.\n\n<details><summary>Check your reasoning</summary>Yes. In the recorded run the verdict was *improved*: ROUGE-L 0.71 → 28.21 (ROUGE-1 29.78, ROUGE-2 14.84), above `summarize: ` (18.99) and Lead-1 (14.08), with outputs of 6.93 words against titles of about 7.4. One seeded split of 100 abstracts carries no dispersion estimate, and ROUGE overlap is not a judgement of whether a title is good.</details>'
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
                "object from files, not the in-memory model (VER2). The cell asserts identical outputs (VER4) — a contract check, "
                "so it stays a hard check. With BYOD the four \"new\" records are taken from your test split (your file has no "
                "fourth slice), and the cell says so.\n\n"
                "**Predict before running:** the adapter changed decoder blocks that every prefix shares. Will the German "
                "translation from Section 5 come out the same?"
            ),
            "code": (
                "import csv\n"
                "import shutil\n\n"
                "if USE_BYOD:\n"
                "    print({{'note': 'BYOD: the four records below are your first four test records, renamed; they are not unseen data'}})\n"
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
        {
            "md": (
                '**What to notice:** the four new titles against their references, the German translation before and after, and `identical_outputs` against `of`.\n\n<details><summary>Check your reasoning</summary>Not necessarily: the record treats a changed translation as a finding to record, not a failure, because tuning shared decoder blocks for one prefix can move the others; one sentence is evidence, not a measurement. Reload parity held in the recorded run — 6 of 6 identical outputs — because the adapter carries the trained tensors exactly and beam search is deterministic on one device.</details>'
            ),
        },
        {
            "md": (
                "## 10. Change one thing: how many decoder blocks to train (optional)\n\n"
                "*Evaluation practice.* A **Predict → Change one thing → Run → Observe → Explain** activity, off by default so "
                "Run all is unaffected. Set `RUN_EXPERIMENT = True`, change **one** field — by default two decoder blocks train "
                "instead of four — and run this cell after Sections 4–9. The experiment loads its **own** pipeline from the "
                "verified snapshot, so it starts from the frozen model and never touches the default `pipe`; it writes only to "
                "`outputs/{stem}_experiment/`, prints the default and the changed run side by side, and checks that the default "
                "exports (adapter, evaluation report, result) are byte-identical afterwards. Expect a few minutes on CPU.\n\n"
                "**Predict:** with half the trainable blocks and the same two epochs, will the new prefix be taught as well?"
            ),
            "code": (
                "RUN_EXPERIMENT = False  # @param {{type:\"boolean\"}}\n"
                "EXPERIMENT_TRAINABLE_DECODER_LAYERS = 2  # @param {{type:\"integer\"}}\n"
                "EXPERIMENT_EPOCHS = 2  # @param {{type:\"integer\"}}\n"
                "EXPERIMENT_LEARNING_RATE = 5e-4  # @param {{type:\"number\"}}\n\n"
                "if not RUN_EXPERIMENT:\n"
                "    print({{'experiment': 'skipped (RUN_EXPERIMENT = False); the default path above is complete'}})\n"
                "else:\n"
                "    canonical_files = {{'adapter': artifact_dir / 'adapter.safetensors', 'evaluation_report': Path('outputs/{stem}_evaluation_report.json'), 'result': Path('outputs/{stem}_result.json')}}\n"
                "    canonical = {{name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in canonical_files.items()}}\n"
                "    experiment_dir = Path('outputs/{stem}_experiment')\n"
                "    shutil.rmtree(experiment_dir, ignore_errors=True)\n"
                "    experiment_dir.mkdir(parents=True)\n"
                "    # Its own pipeline from the verified snapshot: the experiment starts from the frozen model and the default pipe is untouched.\n"
                "    experiment_pipe = T5BaseText2TextPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "    experiment_result = experiment_pipe.adapt(train_records, val_records, prefix=PREFIX, epochs=EXPERIMENT_EPOCHS, lr=EXPERIMENT_LEARNING_RATE, batch_size=BATCH_SIZE, trainable_decoder_layers=EXPERIMENT_TRAINABLE_DECODER_LAYERS, eval_generation=GEN, progress=report)\n"
                "    experiment_test = experiment_pipe.evaluate(test_records, prefix=PREFIX, **GEN)\n"
                "    side_by_side = {{\n"
                "        'settings': {{'default': {{'trainable_decoder_layers': adapt_result['trainable_decoder_layers'], 'epochs': adapt_result['epochs'], 'lr': adapt_result['lr']}}, 'experiment': {{'trainable_decoder_layers': EXPERIMENT_TRAINABLE_DECODER_LAYERS, 'epochs': EXPERIMENT_EPOCHS, 'lr': EXPERIMENT_LEARNING_RATE}}}},\n"
                "        'trainable_parameters': {{'default': adapt_result['n_trainable'], 'experiment': experiment_result['n_trainable']}},\n"
                "        'epoch0_validation_rougeL (frozen model)': {{'default': round(adapt_result['history'][0]['val']['rougeL'], 2), 'experiment': round(experiment_result['history'][0]['val']['rougeL'], 2)}},\n"
                "        'best_epoch': {{'default': adapt_result['best_epoch'], 'experiment': experiment_result['best_epoch']}},\n"
                "        'test': {{metric: {{'frozen': round(frozen_test[metric], 2), 'default': round(adapted_test[metric], 2), 'experiment': round(experiment_test[metric], 2)}} for metric in ('rouge1', 'rouge2', 'rougeL', 'mean_output_words')}},\n"
                "    }}\n"
                "    for key, row in side_by_side.items():\n"
                "        print({{key: row}})\n"
                "    with open(experiment_dir / 'experiment_report.json', 'w', encoding='utf-8') as handle:\n"
                "        json.dump({{'side_by_side': side_by_side, 'history': experiment_result['history'], 'test_metrics': experiment_test}}, handle, indent=2, ensure_ascii=False)\n"
                "    unchanged = {{name: hashlib.sha256(path.read_bytes()).hexdigest() == canonical[name] for name, path in canonical_files.items()}}\n"
                "    if not all(unchanged.values()):\n"
                "        raise RuntimeError(f'the experiment changed a default export: {{unchanged}}')\n"
                "    print({{'default_exports_unchanged': unchanged, 'experiment_outputs': str(experiment_dir)}})\n"
                "    del experiment_pipe"
            ),
        },
        {
            "md": (
                "**Observe → Explain.** Compare the experiment's validation history and its `test` column with the default.\n\n"
                "<details><summary>Check your reasoning</summary>Epoch 0 of the experiment must equal epoch 0 of the default run "
                "(validation ROUGE-L about 0.73 in the recorded run): both are the frozen model, which the restart-from-base rule "
                "guarantees — before that rule, this experiment silently continued from the already-taught model and could not "
                "show anything. The build record's two-block variant, at a lower learning rate, reached ROUGE-L of about 16 in "
                "two epochs and still answered `False` for some abstracts; at the default rate, read your own run: if "
                "`mean_output_words` stays far from title length, the prefix has not taken.</details>"
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
        "**Optional experiments (off by default; each names its field and what to run):** Section 10 trains two decoder blocks "
        "in its own pipeline and prints it beside the default run — change `EXPERIMENT_TRAINABLE_DECODER_LAYERS`, "
        "`EXPERIMENT_EPOCHS` or `EXPERIMENT_LEARNING_RATE` there and run that cell again; every run starts from the frozen "
        "model. Setting `PREFIX = 'summarize: '` (re-target a trained prefix to titles) or `NUM_BEAMS = 1` (greedy scores) and "
        "choosing **Run after** from Section 4 starts from the pinned base — Section 4 restores it — but replaces the default "
        "results and exports. BYOD: `USE_BYOD`, `BYOD_PATH` and `PREFIX` in Section 4, then **Run after** from Section 4, and "
        "read the Lead baseline before the adapted number.\n\n"
        "## Troubleshooting\n\n"
'- **Section 1 stops with "This notebook needs a Linux x86_64 runtime"** — you are on Windows, macOS or an ARM machine. Use Google Colab, Kaggle or a Linux x86_64 Jupyter server.\n- **The uv wheel fails its size/SHA-256 check, or a download in Section 1 times out** — run Section 1 again; a complete environment is reused, an incomplete one is finished. If it repeats, the network is blocking or altering `files.pythonhosted.org` or `pypi.org`.\n- **"The isolated environment\'s Python process exited"** — usually out of memory. Restart the session and choose **Run all**; leave the optional experiment off on a small runtime.\n- **You re-ran Section 1 on its own** — nothing is lost: it keeps the running worker and every variable, so the cells after it keep working. After a session restart, run from the top.\n- **Section 3 reports a size or SHA-256 mismatch, or cannot reach the Hub** — the message names the file. Delete it from the snapshot folder Section 3 prints and run Section 3 again; the snapshot comes from `huggingface.co`.\n- **Section 4 cannot fetch a SciTLDR file, or one fails its digest** — `fetch_corpus` names it; the default path needs `raw.githubusercontent.com`. Run Section 4 again (cached files are re-hashed); delete `weights/scitldr/` if a cached file is corrupt.\n- **CUDA out of memory in Section 7 or 10** — set `BATCH_SIZE = 4` in Section 7 and choose **Run after** from Section 7 (the numbers will differ a little from the recorded run).\n- **BYOD: "BYOD_PATH … does not exist"** — the path is relative to the working directory printed in the message.\n- **BYOD: "the upload dialog exists only in Google Colab"** — on Kaggle or Jupyter, put the zip in the runtime (or attach it as a dataset) and set `BYOD_PATH`.\n- **BYOD: "Upload exactly one .zip file"** — the dialog was cancelled or several files were chosen; run the cell again.\n- **BYOD: "CSV is missing columns …"** — the file needs `id`, `source`, `target` (a UTF-8 file saved by Excel, with a byte-order mark, is fine).\n- **BYOD: "CSV has a header but no data rows"** — add one row per input–output pair.\n- **BYOD: "… the source already starts with the task prefix …"** — give sources without any prefix; the notebook prepends `PREFIX` itself.\n- **BYOD: "CSV line N: …" / "JSONL line N: …" / "JSON record [i]: …"** — that record breaks the rule named after the colon; fix it.\n- **BYOD: "… leave the training split with … supply at least 12 distinct sources"** — add pairs.\n'
        "## Glossary\n\n"
        "- **Encoder-decoder (seq2seq)** — the encoder reads the input; the decoder writes the output token by token.\n"
        "- **Task prefix** — the text before the input that names the task (`summarize: `); a prefix the model never saw is "
        "just words.\n"
        "- **Beam search / `num_beams`** — keeping the best few partial outputs at each step instead of only the most "
        "likely one; deterministic, no score attached.\n"
        "- **`max_new_tokens` / `stopped_by`** — the output budget, and whether the output ended naturally (`eos`) or was "
        "cut.\n"
        "- **Teacher forcing / cross-entropy** — training the decoder to predict each next token of the reference, given the "
        "reference so far.\n"
        "- **ROUGE-1 / ROUGE-2 / ROUGE-L** — overlap of words, word pairs and the longest common subsequence between an "
        "output and its reference, as F1 × 100; overlap, not quality or faithfulness.\n"
        "- **Lead-N baseline** — the first N sentences of the input, submitted as the output.\n"
        "- **Epoch / learning rate / validation selection** — one pass over the training pairs; the update step size; "
        "choosing the epoch on pairs that are neither trained on nor used for the final score.\n"
        "- **Held-out test split / leakage** — pairs never used for training or selection; the same source (or title) on "
        "both sides of a split.\n"
        "- **Adapter / reload parity** — the trained tensors only (safetensors) overlaid on the pinned base; the reloaded "
        "pipeline produces identical outputs.\n"
        "- **BYOD** — bring your own data: your input–output pairs and prefix through the same cells.\n\n"
        "## Conclusion (your notes)\n\n"
        "Optional — fill in from **your** run, not the recorded one:\n\n"
        "- The task was ___ (prefix `___`) on ___ test records.\n"
        "- Lead-1 scored ROUGE-L ___, the frozen model ___ under the new prefix and ___ under `summarize: `.\n"
        "- After fine-tuning, ROUGE-L moved from ___ to ___ at ___ words per output; the verdict was ___.\n"
        "- The German translation before and after was ___ / ___, because ___.\n"
        "- One reason not to trust this number on my own task yet: ___ (for example the size of the test split, ROUGE as "
        "overlap only, or the shared decoder).\n\n"
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
