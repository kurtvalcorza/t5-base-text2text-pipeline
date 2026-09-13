"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 1.1 §3.6 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
module, and the model pin/stage/verify cells are produced by the generator from repository
sources so they cannot drift from the package.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "t5_base_text2text_pipeline",
    "repo_name": "t5-base-text2text-pipeline",
    "stem": "t5_base_text2text",
    "notebook_name": "t5_base_text2text_colab.ipynb",
    "profile": "TASK-INFERENCE",
    "pipeline_class": "T5BaseText2TextPipeline",
    "weights_key": "t5-base",
    "runtime_imports": ["torch", "transformers"],
    "title": "T5-Base — DIMER text-to-text generation tutorial (standalone)",
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
    "capability": "caller-prefixed text-to-text generation (summarisation and English→German/French/Romanian translation) using the pinned `google-t5/t5-base` weights",
    "intro": (
        "`google-t5/t5-base` is the original 220 M-parameter T5 of Raffel et al. (2020) — **not FLAN-T5**, so it follows "
        "only the task prefixes it was trained on, not free-form instructions. At inference the encoder reads the whole "
        "prefixed input once and the decoder emits one SentencePiece token per step until `</s>` or a step ceiling; "
        "**greedy decoding** (per-step argmax, `num_beams=1`, `do_sample=False`) is the default decision rule and beam "
        "search is available on request — there is no sampling and no seed. The caller writes the task prefix — "
        "`summarize: `, `translate English to German: `, `translate English to French: ` or `translate English to "
        "Romanian: ` — in front of the text: **the pipeline invents no prefix** and only reports which known one an "
        "input started with (`known_prefix`, or `None`). **No adaptation occurs:** no training, fine-tuning, in-context "
        "conditioning, or preprocessing fitting — the pinned checkpoint is used as published. What the upstream "
        "checkpoint supplies is the model, the SentencePiece tokenizer and the prefix convention; what the carried "
        "pipeline module adds is manifest verification, input validation with named ceilings, a fixed output contract, "
        "the list of trained prefixes as `TASK_PREFIXES`, and the `validate_inputs` and `evaluation_report` stage "
        "helpers.\n\n"
        "**Snapshot note:** the pinned revision ships **no `tokenizer_config.json`** (a 6-file manifest), so "
        "`T5TokenizerFast` reads `tokenizer.json` and falls back to class defaults (pad 0 / EOS 1), as the model "
        "card documents; the card-pass smoke printed no tokenizer warning. Section 3 stages and digest-verifies "
        "those six files before the tokenizer or the model is constructed."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried pipeline module guarantees, author two already-prefixed "
        "inputs (or upload your own), stage and digest-verify the immutable upstream snapshot, surface the pipeline's "
        "ceilings and the decision rule and validate the inputs into an input manifest before any model work, generate "
        "through the public API with explicit `max_new_tokens`/`num_beams`, read `stopped_by`, `known_prefix` and the "
        "token counts correctly, read from the machine-readable evaluation report why **no metric is reported** and what "
        "references a ROUGE/BLEU evaluation would need, and export every generation with its identifier plus provenance."
    ),
    "exclusions": (
        "instruction following or chat, sampling-based decoding, batching, classification or embedding (the GLUE tasks "
        "in the training mixture are not exposed), source languages other than English or target languages other than "
        "German, French and Romanian, the upstream `task_specific_params` decoding settings (`min_length`, "
        "`length_penalty`, `no_repeat_ngram_size` are not applied by the pipeline), or any ROUGE/BLEU measurement. The "
        "repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU (float32) and uses CUDA automatically when available (float32 there too). CPU is adequate: the repository's model card records, for the Windows-venv smoke on an Intel Core Ultra 9 275HX, 4.55 s to load and digest-verify the 892 MB snapshot, 0.29 s for the 11-token translation and 0.83 s for a 94-token `summarize: ` input; the `t5-small-text2text-pipeline` sibling (60 M parameters, 244 MB snapshot) loaded in 3.89 s and took 0.16 s / 0.31 s for the same two calls on the same CPU in its own smoke, as both cards record. The pinned `torch==2.14.0` install and the 891 MB checkpoint are the large downloads of the run.",
        "- **Knowledge:** basic Python; what an encoder-decoder (seq2seq) model is; what greedy decoding and beam search do; why a generated sentence can be fluent and wrong.",
        "- **Data:** the default sample is **synthetic** — two already-prefixed inputs authored in code (one translation sentence, one four-sentence passage to summarise) — so nothing is downloaded and no private data is needed. It carries no reference outputs, so any number it produces is smoke/sanity evidence, never a quality measurement. Optional BYOD upload is gated off by default so the sample path can run top-to-bottom without interaction; each non-empty line is one input that must already start with its task prefix. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded text remains in the notebook runtime; this pipeline does not send it to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Author the synthetic sample or optional BYOD\n\n"
                "The default sample is **synthetic**: two inputs authored in this cell, each already carrying its task "
                "prefix because the pipeline invents no prefix. The first is the translation sentence the repository's "
                "card-pass smoke used (`translate English to German: The house is wonderful.`); the second is a "
                "four-sentence English passage about the C4 corpus, written here after the T5 paper's description, behind "
                "`summarize: `. Neither has a reference output — no human translation, no reference summary — so nothing "
                "in this notebook is a quality measurement; the card's smoke observation that the translation came back "
                "as `Das Haus ist wunderbar.` is one run on one machine, not an expected value this notebook asserts. "
                "The sample identity and a SHA-256 of its text are printed so an export can be tied to exactly these "
                "inputs.\n\n"
                "Two Colab form parameters fix the generation settings for every call: `GEN_MAX_NEW_TOKENS` (default 64, "
                "the package's `DEFAULT_MAX_NEW_TOKENS`) and `NUM_BEAMS` (default 1 = greedy). They are checked against "
                "the carried module's ceilings in the next section.\n\n"
                "BYOD is optional and disabled by default. Expected BYOD input: one UTF-8 text file in which every "
                "non-empty line is one input **that already starts with its task prefix** (for example `translate "
                "English to French: ...`); each line must be at most `MAX_TEXT_CHARS` characters and tokenise to at most "
                "`MAX_INPUT_TOKENS` SentencePiece pieces, which the pipeline enforces by rejecting, not by truncating. "
                "The upload stays inside this runtime. If you hold reference outputs for your lines, keep them outside "
                "the notebook — Section 7 explains what to compute with them."
            ),
            "code": (
                "import hashlib\n"
                "import io\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "GEN_MAX_NEW_TOKENS = 64  # @param {{type:\"integer\"}}\n"
                "NUM_BEAMS = 1  # @param {{type:\"integer\"}}\n\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    sample_name = next(iter(uploaded))\n"
                "    texts = [line.strip() for line in io.StringIO(uploaded[sample_name].decode('utf-8')) if line.strip()]\n"
                "    if not texts:\n"
                "        raise ValueError(f'{{sample_name}}: expected at least one non-empty line, each starting with its task prefix')\n"
                "    sample_kind = 'BYOD upload'\n"
                "else:\n"
                "    texts = [\n"
                "        'translate English to German: The house is wonderful.',\n"
                "        'summarize: The Colossal Clean Crawled Corpus, or C4, is a cleaned version of the April 2019 Common Crawl web snapshot. '\n"
                "        'The corpus was prepared by keeping only lines that end in terminal punctuation, dropping pages with fewer than five sentences, '\n"
                "        'and removing pages that contain words from a block list, placeholder text or source code. '\n"
                "        'Duplicate three-sentence spans were also removed so that boilerplate does not dominate the training signal. '\n"
                "        'It was used to pre-train the T5 family of models with a span-corruption denoising objective.',\n"
                "    ]\n"
                "    sample_name = 'synthetic_prefixed_inputs'\n"
                "    sample_kind = 'synthetic (authored in this cell)'\n"
                "item_ids = [f'input{{index:02d}}' for index in range(len(texts))]\n"
                "sample_sha256 = hashlib.sha256('\\n'.join(texts).encode('utf-8')).hexdigest()\n"
                "print({{'sample': sample_name, 'sample_kind': sample_kind, 'inputs': len(texts), 'text_sha256': sample_sha256, 'max_new_tokens': GEN_MAX_NEW_TOKENS, 'num_beams': NUM_BEAMS}})\n"
                "for item_id, text in zip(item_ids, texts, strict=True):\n"
                "    print(f'{{item_id}}: {{text[:110]}}' + ('...' if len(text) > 110 else ''))"
            ),
        },
        {
            "md": (
                "## 5. Validate the inputs → input manifest\n\n"
                "`validate_inputs` is the pipeline's public validation stage: it applies exactly the checks `generate` "
                "applies — both route through the same private `_check_inputs` — so the text type, non-emptiness, the "
                "character ceiling `MAX_TEXT_CHARS`, `max_new_tokens` in 1..`MAX_NEW_TOKENS` and `num_beams` in "
                "1..`MAX_NUM_BEAMS` are enforced identically. `generate` takes one text per call, so the helper "
                "validates the whole batch the notebook will loop over with the same settings and returns one **input "
                "manifest** naming the schema and ceilings, each input's identifier, character count and "
                "`known_prefix`, the settings in force, and the verdict; it is written to "
                "`outputs/{stem}_input_manifest.json`. An input that starts with no trained prefix is **not** rejected — "
                "the pipeline does not refuse it either — but `known_prefix` is `null` so a reader can see it, and the "
                "model then returns something plausible-looking rather than an error. `MAX_INPUT_TOKENS` (encoder tokens "
                "including `</s>`; the upstream `n_positions`) needs the real tokenizer and is therefore enforced inside "
                "`generate`, which **rejects with a `ValueError` naming the count, never silently cuts**; every result "
                "reports `input_tokens`. `DECISION_RULE` states the decoding rule in force (greedy per-step argmax, beam "
                "search when `num_beams > 1`, no sampling) and `TASK_PREFIXES` lists the four prefixes the checkpoint "
                "was trained on. To show what rejection looks like, the cell also validates an out-of-range `num_beams` "
                "and records the pipeline's own error message as a finding. Nothing here trims or alters the texts."
            ),
            "code": (
                "import json\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "ceilings = {{'MAX_TEXT_CHARS': MAX_TEXT_CHARS, 'MAX_INPUT_TOKENS': MAX_INPUT_TOKENS, 'MAX_NEW_TOKENS': MAX_NEW_TOKENS, 'MAX_NUM_BEAMS': MAX_NUM_BEAMS, 'DEFAULT_MAX_NEW_TOKENS': DEFAULT_MAX_NEW_TOKENS}}\n"
                "print(ceilings)\n"
                "print({{'decision_rule': DECISION_RULE}})\n"
                "print({{'task_prefixes': list(TASK_PREFIXES)}})\n"
                "input_manifest = validate_inputs(texts, max_new_tokens=GEN_MAX_NEW_TOKENS, num_beams=NUM_BEAMS, names=item_ids)\n"
                "# Demonstrate rejection on a setting that breaks a ceiling; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs(texts, max_new_tokens=GEN_MAX_NEW_TOKENS, num_beams=MAX_NUM_BEAMS + 1)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'num-beams-ceiling-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest, indent=2))\n"
                "unknown = [entry['id'] for entry in input_manifest['inputs'] if entry['known_prefix'] is None]\n"
                "if unknown:\n"
                "    print({{'warning': f'{{unknown}} start with no trained prefix; the model will still return text, but not a summary or translation'}})\n"
                "print({{'token_ceiling': 'enforced by generate() with the real tokenizer; reported as input_tokens'}})"
            ),
        },
        {
            "md": (
                "## 6. Generate and read the outputs correctly\n\n"
                "`generate(text, max_new_tokens=..., num_beams=...)` runs one prefixed input through the encoder-decoder "
                "and returns a dict: `text` (the decoded generation with special tokens removed), `generated_tokens` "
                "(decoder tokens emitted, excluding `</s>`/pad), `input_tokens` (encoder tokens including `</s>`, the "
                "number checked against `MAX_INPUT_TOKENS`), `stopped_by` (`eos` when the model ended the sequence "
                "itself, `max_new_tokens` when it hit the ceiling — such an output is cut mid-thought and should be "
                "re-run with a larger `GEN_MAX_NEW_TOKENS` before anyone reads it as a finished summary), `known_prefix` "
                "(which trained prefix the input started with, or `None`), the `generation` settings actually used "
                "(`max_new_tokens`, `num_beams`, `do_sample=False`, `decision_rule`), `device`, `source` and the model "
                "identity. **Score semantics:** the pipeline emits **no probability, confidence or score of any kind** — "
                "the counts above are counts, not scores; greedy decoding is an implicit per-step argmax with no "
                "minimum-probability cut-off, so some token is always produced, and the pipeline ships no acceptance "
                "threshold on output quality. Whoever deploys it owns any acceptance rule, judged on their own "
                "references. The run is deterministic for a given input, settings, weights, device and library versions "
                "(no sampling, `model.eval()`, no seed needed); float32 kernel differences between CPU and CUDA can flip "
                "a near-tied token and change the rest of the sequence from that point, and beam search can differ from "
                "greedy. The checks below are falsifiable plumbing checks — one result per input, every count within its "
                "ceiling, every default input recognised by its prefix — plus a per-call wall time measured on the "
                "runtime identified in Section 1 (the first call includes warm-up). Look for a short German sentence for "
                "`input00` and an English condensation for `input01`; whether they are *good* is exactly what no number "
                "here can tell you."
            ),
            "code": (
                "import time\n\n"
                "results = []\n"
                "for item_id, text in zip(item_ids, texts, strict=True):\n"
                "    started = time.perf_counter()\n"
                "    result = pipe.generate(text, max_new_tokens=GEN_MAX_NEW_TOKENS, num_beams=NUM_BEAMS)\n"
                "    elapsed = time.perf_counter() - started\n"
                "    results.append({{'id': item_id, 'input': text, 'seconds': round(elapsed, 3), **result}})\n"
                "    print(f\"{{item_id}} [{{result['known_prefix'] or 'no known prefix'}}] {{result['input_tokens']}} -> {{result['generated_tokens']}} tokens, stopped_by={{result['stopped_by']}}, {{elapsed:.2f}} s\")\n"
                "    print(f\"    {{result['text']}}\")\n"
                "checks = {{\n"
                "    'one_result_per_input': len(results) == len(texts),\n"
                "    'generated_within_ceiling': all(r['generated_tokens'] <= GEN_MAX_NEW_TOKENS for r in results),\n"
                "    'input_within_ceiling': all(r['input_tokens'] <= MAX_INPUT_TOKENS for r in results),\n"
                "    'settings_echoed': all(r['generation']['max_new_tokens'] == GEN_MAX_NEW_TOKENS and r['generation']['num_beams'] == NUM_BEAMS and r['generation']['do_sample'] is False for r in results),\n"
                "}}\n"
                "if not USE_BYOD:\n"
                "    checks['every_default_input_has_known_prefix'] = all(r['known_prefix'] is not None for r in results)\n"
                "if not all(checks.values()):\n"
                "    raise RuntimeError(f'generate output failed a sanity check: {{checks}}')\n"
                "print({{'checks': checks, 'decision_rule': results[0]['generation']['decision_rule'], 'hit_token_ceiling': [r['id'] for r in results if r['stopped_by'] == 'max_new_tokens']}})"
            ),
        },
        {
            "md": (
                "## 7. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and always produces a report — even, as "
                "here, when nothing is measurable. The repository ships **no metric helper and reports no performance "
                "measure**: summarisation and translation are conventionally scored with ROUGE-1/2/L and BLEU (or chrF) "
                "against human **reference outputs** — a reference summary per document, a reference translation per "
                "sentence — over enough items to state a dispersion, and the synthetic sample has none, so the verdict "
                "is always `not-measurable` and none is manufactured from a proxy such as length ratio or copy rate. "
                "Supplying a reference does not change the verdict, because no metric helper exists to score it and one "
                "reference is not a dispersion; the helper records that in `reason`. The report is written for the first "
                "generation and the verdict applies to every one of them equally — no metric exists for any. The "
                "upstream per-task figures in the paper's Table 14 are upstream claims, not measured here. It lands at "
                "`outputs/{stem}_evaluation_report.json`."
            ),
            "code": (
                "report = evaluation_report(results[0], sample_kind=sample_kind)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(report, indent=2))\n"
                "if report['verdict'] == 'not-measurable':\n"
                "    print('No metric is reported: the sample has no reference outputs, and the repository ships no metric helper; compute ROUGE/BLEU on your own referenced inputs.')"
            ),
        },
        {
            "md": (
                "## 8. Export the generations and provenance\n\n"
                "Two further files are written under `outputs/` beside the input manifest and the evaluation report: "
                "`outputs/{stem}_generations.csv` — one row per input with its identifier, the known prefix, the input "
                "text, the generated text, both token counts, `stopped_by` and the wall time, so every generation maps "
                "back to its input — and `outputs/{stem}_result.json`, which carries the same items plus the generation "
                "settings in force, the ceilings, the sanity checks, the input manifest, the evaluation report, the "
                "sample identity and digest, the notebook's source (repository, revision, embedded module digest, "
                "generator), the model identifier, the immutable model revision, the model licence, the verified "
                "snapshot summary, and the runtime identity (Python, `torch`, `transformers`, device, dtype). No "
                "credentials are involved in any step, so none can reach the export."
            ),
            "code": (
                "import csv\n\n"
                "items = [\n"
                "    {{'id': r['id'], 'known_prefix': r['known_prefix'], 'input': r['input'], 'output': r['text'], 'input_tokens': r['input_tokens'], 'generated_tokens': r['generated_tokens'], 'stopped_by': r['stopped_by'], 'seconds': r['seconds']}}\n"
                "    for r in results\n"
                "]\n"
                "with open('outputs/{stem}_generations.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.DictWriter(handle, fieldnames=list(items[0]))\n"
                "    writer.writeheader()\n"
                "    writer.writerows(items)\n"
                "payload = {{\n"
                "    'items': items,\n"
                "    'generation': results[0]['generation'],\n"
                "    'ceilings': ceilings,\n"
                "    'sanity_checks': checks,\n"
                "    'generations_file': 'outputs/{stem}_generations.csv',\n"
                "    'input_manifest': input_manifest,\n"
                "    'evaluation_report': report,\n"
                "    'sample': {{'name': sample_name, 'kind': sample_kind, 'inputs': len(texts), 'text_sha256': sample_sha256}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'snapshot': {{'path': str(WEIGHTS_DIR), 'files': len(snapshot['files']), 'total_bytes': snapshot.get('totalBytes'), 'fetched_this_run': fetched}},\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'transformers': transformers.__version__,\n"
                "        'device': pipe.device,\n"
                "        'dtype': 'float32',\n"
                "        'source': pipe.source,\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The generated strings are the model's continuation of *your* prefixed input under greedy (or beam) decoding: "
        "fluent text that can add, drop or invert a fact, and the pipeline attaches no probability, confidence or "
        "quality score to it — `generated_tokens`, `input_tokens` and `stopped_by` are counts and flags, not evidence of "
        "correctness. On the synthetic sample the checks prove only that the input contract, the prefix convention, the "
        "verified snapshot load and the generation path work end to end; the evaluation report is `not-measurable` "
        "because none can be computed without reference outputs, and a real evaluation needs referenced inputs from your "
        "own domain, a ROUGE/BLEU-style scorer, and enough items to state a dispersion. An unknown prefix produces "
        "plausible-looking output rather than an error; inputs above `MAX_INPUT_TOKENS` are refused rather than cut; "
        "outputs that stop at `max_new_tokens` are truncated mid-thought. The pipeline exposes no instruction following, "
        "sampling, batching, or the upstream `task_specific_params` decoding settings. Decoding is deterministic on a "
        "fixed device and dtype, but CPU and CUDA float32 kernels can diverge on a near-tied token.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this notebook, "
        "can acquire and digest-verify the pinned model snapshot, validate the demonstrated inputs against the enforced "
        "ceilings, execute the public pipeline path, and emit the shown machine-readable outputs in the tested runtime — "
        "without the repository being reachable. It does **not** establish benchmark superiority, summarisation or "
        "translation quality on any domain, a usable acceptance threshold, safety for high-consequence decisions, or "
        "production fitness on an unseen domain.\n\n"
        "**Troubleshooting.** `RuntimeError: Core dependencies changed while older modules were loaded` in Section 1: "
        "the pinned install replaced a package the runtime had pre-imported — restart the runtime and rerun from the "
        "top. `FileNotFoundError: snapshot file missing` or a `sha256`/`size` `ValueError` in Section 3: a staged file is "
        "incomplete or altered — delete it from `weights/{MODEL_KEY}/` and rerun Section 3. `ValueError: input is N "
        "tokens; ceiling is MAX_INPUT_TOKENS=512` in Section 6: split or shorten that BYOD line and rerun from "
        "Section 4. `stopped_by` equal to `max_new_tokens`: raise `GEN_MAX_NEW_TOKENS` (ceiling `MAX_NEW_TOKENS`) and "
        "rerun Section 6. Output that copies the input: the line has no trained prefix, or the passage is too short to "
        "summarise.\n\n"
        "**Next experiments.** Set `NUM_BEAMS = 4` and compare the beam-search outputs with the greedy ones (the "
        "card-pass smoke found the short translation unchanged); translate the same sentence with the French and "
        "Romanian prefixes; hand the `summarize: ` input a passage you wrote a one-sentence reference summary for and "
        "score the output with a ROUGE implementation of your choice — the first step towards the real evaluation the "
        "report asks for; run the same batch on a CUDA runtime and diff the outputs against the CPU run. None of these "
        "turns the sample result into evidence of production fitness.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/t5-base-text2text-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/t5-base-text2text-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/t5-base-text2text-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/google-research/text-to-text-transfer-transformer\n"
        "- Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (Raffel et al., JMLR 2020): https://arxiv.org/abs/1910.10683"
    ),
}
