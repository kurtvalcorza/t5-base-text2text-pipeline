# Release verification

`tutorials/t5_base_text2text_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the
exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation,
code-cell compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but
are **not** runtime evidence under DIMER Notebook Specification 2.0 (REL8). This file is the durable release-gate
record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.0`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `samples.py`, `metrics.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed 7-entry snapshot manifest and the
  inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; the pinned-install cell with its
  restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same
  identity string in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions (the pinned corpus
  commit is the one allowed second hash);
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `T5BaseText2TextPipeline.from_pretrained(weights_dir=...)`, `fetch_corpus` from the pinned cache path,
  `read_corpus` + `build_sample_dataset(seed=SPLIT_SEED)` / `load_byod_dataset`, `validate_dataset` per split,
  `check_split_disjoint`, `known_prefix(PREFIX)`, `write_dataset_csv`, `validate_inputs` with the `num_beams` refusal
  probe, `pipe.generate` with the sanity checks on trained prefixes and the new-prefix probe, `lead_baseline`,
  `pipe.evaluate` on the frozen model under the new prefix and under `summarize: ` and on the validation and test
  splits after adaptation with the ROUGE-L assertion, `pipe.adapt` with its explicit hyperparameters, `prefix=PREFIX`
  and `eval_generation=GEN`, `pipe.evaluate` on the unseen abstracts plus the single-item `evaluation_report` and
  the trained-prefix re-check, `pipe.save_artifact`, `T5BaseText2TextPipeline.from_artifact` and the reload-parity
  assertion, and the provenance fields `weight_format`, `weight_sha256` and the `corpus` block), the six expected
  `outputs/` paths, the learner-facing statements (no score emitted, greedy decoding, adaptation to a new prefix, a
  prefix the checkpoint has never seen, the Lead-1 baseline, rouge-score-style not rouge-score-identical, no
  dispersion estimate, inputs above the token ceiling rejected not truncated, named exclusions, the Apache-2.0 corpus
  licence) and the gated-off BYOD default; forbidden patterns (credential-in-URL, any `git clone` / `github.com` /
  repository import on the primary path, a mutable `revision='main'`, direct `from transformers import` /
  `T5ForConditionalGeneration` / `T5TokenizerFast` / `model.generate(` / `from huggingface_hub import` /
  `urllib.request` / `safetensors` / `torch.optim` / `.backward(` / `pipe._model` use **outside the carried module
  cells**, `trust_remote_code=True`, `pickle.load`, `torch.load(` without `weights_only=True`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the 19 required headings in order, and the
  immutable provenance section.

CI also installs the pinned CPU-only torch wheel plus `transformers`, `tokenizers`, `sentencepiece`,
`huggingface-hub`, `safetensors` and `numpy`, runs `ruff check src tests tools`, `tools/build_notebook.py --check`,
and the offline unit suite (`tests/test_pipeline.py`, `tests/test_adaptation.py`, `tests/test_role_helpers.py`,
`tests/test_import_boundary.py`, `tests/test_notebook_parity.py`; injected runner, token counter and corpus fetcher,
temporary manifests, no weights — `tests/test_model_backed.py` is skipped without the snapshot). These are
source/provenance and unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh CPU or GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; promotion evidence |
| Local harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, pre-staged pins | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU or CUDA runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/t5-base/` or the corpus cache `weights/scitldr/` (the standalone path writes the manifest
   itself, stages the missing files from the Hub, and fetches the three pinned SciTLDR-A files from the project
   repository, so neither directory may be seeded);
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `SPLIT_SEED = 42`, `PREFIX = 'paper title: '`, `GEN_MAX_NEW_TOKENS = 24`, `NUM_BEAMS = 4`,
   `EPOCHS = 2`, `LEARNING_RATE = 5e-4`, `BATCH_SIZE = 8`, `TRAINABLE_DECODER_LAYERS = 4`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `transformers==4.57.6`, `tokenizers==0.22.2`, `sentencepiece==0.2.2`,
   `huggingface-hub==0.36.2`, `safetensors==0.8.0`, `numpy==2.5.3` (an interpreter restart after the install is
   expected where the runtime's preinstalled torch or numpy differ from the pins);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the three carried module cells execute (defining `T5BaseText2TextPipeline`, `verify_snapshot`,
     `stage_missing_files`, `validate_inputs`, `evaluation_report`, `known_prefix`, `fetch_corpus`, `read_corpus`,
     `build_sample_dataset`, `filter_records`, `validate_dataset`, `check_split_disjoint`, `split_dataset`,
     `load_byod_dataset`, `write_dataset_csv`, `rouge_n`, `rouge_l`, `text_metrics`, `lead_baseline` and the
     ceilings) with no import of the repository package;
   - the inline manifest asserted against the module's constants, then `stage_missing_files(WEIGHTS_DIR,
     allow_download=True)` reporting `['model.safetensors']` (and any other absent entry) fetched from
     `google-t5/t5-base` at the immutable revision, and `verify_snapshot` returning its dict (7 files);
     `from_pretrained(weights_dir=WEIGHTS_DIR)` loading from the verified directory with `source` `local-snapshot`;
   - Section 4: `fetch_corpus` fetching the three pinned files (3,155,015 / 1,124,865 / 1,204,107 bytes) from
     `raw.githubusercontent.com` into `weights/scitldr/`, 1,992 + 619 + 618 raw papers read, `prefix_is_trained`
     `False`, and the seeded draw of 300 / 50 / 100 records with `check_split_disjoint` reporting no shared abstract
     and the three dataset digests `eadcee41…` / `6139c992…` / `ea8cb17c…`; `outputs/…_train.csv` written; the four
     dataset refusal probes each raising `ValueError`;
   - Section 5: the ceilings (`MAX_TEXT_CHARS` 20000, `MAX_INPUT_TOKENS` 512, `MAX_NEW_TOKENS` 512, `MAX_NUM_BEAMS`
     8, `DEFAULT_MAX_NEW_TOKENS` 64, `MAX_PREFIX_CHARS` 64), `DECISION_RULE` and `TASK_PREFIXES` surfaced;
     `validate_inputs` writing `outputs/…_input_manifest.json` (verdict `accepted`, one recorded rejection finding
     from the `num_beams` ceiling probe); `pipe.generate` on the German-translation and `summarize: ` inputs with
     every sanity check `True` (the card-pass smoke gave `Das Haus ist wunderbar.`), then the new-prefix probe on
     one test abstract returning a one-word answer (`True` in the recorded run);
   - Section 6: the Lead-1 baseline (ROUGE-L ≈ 14.08 on the sample), the frozen model under `paper title: `
     (ROUGE-L ≈ 0.71 at ≈ 1 word) and under `summarize: ` (ROUGE-L ≈ 18.99 at ≈ 16.7 words) against
     ≈ 7.4-word titles on CPU float32, with the cell's assertion that the Lead-1 ROUGE-1 is positive;
   - Section 7: `pipe.adapt` printing epoch 0 as the frozen model, 37,757,952 trainable of 222,903,552 parameters,
     300 training abstracts, and a two-epoch history with validation ROUGE-L rising from near zero into the
     thirties (≈ 0.73 → 33.11 → 35.32 in the recorded run; `best_epoch` 2);
   - Section 8: `pipe.evaluate` on the validation and test splits with the four-way comparison and
     `outputs/…_evaluation_report.json` written (the cell asserts the adapted test ROUGE-L exceeds the frozen one
     under the same prefix — on the sample ≈ 29.97 versus ≈ 0.71, at ≈ 6.5 words);
   - Section 9: four unseen dev abstracts titled with `pipe.evaluate` returning `measured-small-sample`, the
     single-item `evaluation_report` verdict `sample-sanity`, the German translation generated again after
     adaptation (printed beside the frozen output; a change is a finding to record, not a failure),
     `outputs/…_generations.csv` written; `pipe.save_artifact` writing `outputs/…_adapter/{adapter.safetensors,
     manifest.json}` (52 tensors, about 151 MB, `prefix` recorded) and `T5BaseText2TextPipeline.from_artifact`
     reloading it with 6/6 identical outputs (the cell asserts it); `outputs/…_result.json` written with
     `NOTEBOOK_SOURCE`, the model identity and licence, the snapshot block (`weight_format`, `weight_sha256`), the
     prefix, the `corpus` block, the inference-contract items, the comparison, the artifact digest, the reload parity,
     the runtime versions and device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device), the model
   identifier and immutable revision, whether the model cache, the weights directory and the corpus cache were clean,
   outcome, produced outputs, the observed metrics (as observations, not a benchmark) and any warning or applicable
   `SHOULD` deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `t5_base_text2text_colab.ipynb` (`E2E`, `GUIDED`, spec 2.5) | `ba60b61` / `ac38dca0` | 2026-10-25 | Colab CLI sequential execution, fresh Colab Tesla T4 VM (`colab exec -f` on a fresh session; fresh VM, empty Hugging Face cache, no repository checkout) — not a browser `Run all`; no execution counts, order from `exec.log` | **PASSED in one pass, no restart, 0 errors** — 13/13 code cells, 296.3 s wall; default settings only; test ROUGE-L frozen new prefix 0.71 → adapted 28.21, epoch 2 kept; see Recorded executions; evidence in `docs/execution-evidence/2026-10-25-ba60b61/` |
| `t5_base_text2text_colab.ipynb` (`E2E`) | `3220429` / `d2f10303` | 2026-09-19 | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-t5-base-text2text` v2; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` after, Python 3.12.13, `cuda:0`) | **PASSED** — 11/11 code cells ok (1 restart after install cell); 17 files, 899 MB staged from the Hub into a clean cache; comparison {rouge1: {lead1: 16.73, frozen_summarize: 21.68, frozen_new_prefix: 0.71, adapted: 29.78}, rouge2: {lead1: 4.83, frozen_summarize: 8.53, frozen_new_prefix: 0.67, adapted: 14.84}, rougeL: {lead1: 14.08, frozen_summarize: 18.99, frozen_new_prefix: 0.71, adapted: 28.21}, mean_output_words: {lead1: 21.41, frozen_summarize: 16.7, frozen_new_prefix: 1.03, adapted: 6.93}, delta_vs_frozen: {rouge1: 29.08, rouge2: 14.17, rougeL: 27.51}}; reload parity {identical_outputs: 6, of: 6}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-t5-base-text2text/v2/evidence/` in the workspace |
| `t5_base_text2text_colab.ipynb` (`E2E`) | `e09a306` / `486beac4` | 2026-09-19 | Local pre-flight harness (Windows, CPython 3.12.10, CPU, `google.colab` shim, pins pre-installed) | PASS — pre-flight only, **not** promotion evidence |
| `t5_base_text2text_colab.ipynb` (`TASK-INFERENCE`, superseded) | `198685c` / `fbc9be0e4069` | 2026-09-14 | Kaggle CPU (`kurtvalcorza/dimer-nb2-t5-base-text2text` v1) | PASSED — 8/8 code cells, 285.0 s; evidence for the earlier inference-only notebook, not for the `E2E` blob |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/t5_base_text2text_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/t5_base_text2text_colab.ipynb`). Wall times are the sum of per-cell times
reported by the executor and include the model download where it occurred; they are measurements for the stated
runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-10-25 | `ba60b61` / `ac38dca0` (PR #9 head) | Colab CLI sequential execution, fresh Colab Tesla T4 VM (`colab exec -f` on a fresh session; fresh VM, empty Hugging Face cache, no repository checkout) — not a browser `Run all`; no execution counts, order from `exec.log` | Default path, every form field at its default, no overrides; notebook bytes fetched at the full commit SHA `ba60b6153239732152352f9e7acdc714418a474c`, git blob SHA-1 `ac38dca05ca77b15f58f8a44e4643512d33f542f` checked, input unchanged; captured executed-notebook SHA-256 `b89a388fb6741614cf4ea0a325613a882f196f936062d2a5b1484e2fb4ddc15c` equals `run_summary.json` `executed_sha256` | 296.3 s | **PASSED in one pass, no restart, 0 errors** — 13/13 code cells (`exec.log`); isolated environment built in 48 s (CPython 3.12.12 from the 48-package lock, kernel Python 3.13.16, `cuda:0`); raw papers 1,992 + 619 + 618, splits 300 / 50 / 100; trained-prefix probes (`Das Haus ist wunderbar.`); frozen model under `paper title: ` answered `True`; test ROUGE-1/2/L Lead-1 16.73 / 4.83 / 14.08 (21.4 words), `summarize: ` 21.68 / 8.53 / 18.99 (16.7 words), frozen new prefix 0.71 / 0.67 / 0.71 (1.03 words, against references of 7.4); validation ROUGE-L 0.73 → 30.88 → 35.39, `best_epoch` 2 (37,757,952 of 222,903,552 parameters, started from the pinned base, 61.7 s); adapted ROUGE-1/2/L 29.78 / 14.84 / 28.21 (6.93 words), `delta_vs_frozen` ROUGE-L 27.51, verdict `improved`; four new abstracts ROUGE-L 32.05 `measured-small-sample`; German translation unchanged before/after; `reload_parity` `{'identical_outputs': 6, 'of': 6}`; Section 10 experiment skipped at its default. Not exercised: BYOD (REL12), the experiment and a browser `Run all` (REL2). Evidence in `docs/execution-evidence/2026-10-25-ba60b61/` |
| 2026-10-09 | `93a578f` / `09d88ee9a276` | Colab CLI 0.7.4 sequential execution, fresh Colab Tesla T4 (session `suite-t5-93a578f-8dff`) — not a browser Run all | Default sample path (`USE_BYOD = False`, `RUN_EXPERIMENT = False`) | 331.3 s | **PASSED** — 13/13 code cells in order, 0 error outputs, `restarted: false` (no output asks for a restart; the three carried-module cells are silent by design). Isolated env: 48 locked packages, CPython 3.12.12 (kernel 3.13.15), setup 63 s; torch 2.14.0+cu130, transformers 4.57.6, `cuda:0`; `google-t5/t5-base` @ `a9723ea7f1b3` staged clean (6 files, 893,837,231 bytes, `local-snapshot`). Frozen `paper title: ` answers `True`; validation ROUGE-L 0.73 / 30.88 / 35.39 over epochs 0-2, epoch 2 kept, `started_from` `pinned base`; test ROUGE-L Lead-1 14.08, `summarize:` 18.99, frozen new prefix 0.71, adapted 28.21 (ROUGE-1 29.78, ROUGE-2 14.84), verdicts `improved` / Lead-1 shares words; translation unchanged (`Das Haus ist wunderbar.`); adapter 52 tensors, reload parity 6/6; six exports. Evidence: `docs/execution-evidence/2026-10-09-93a578f/` (executed notebook SHA-256 `3d4048443ec8…`). An earlier run of `2d130f7` passed identically but recorded a misleading `started_from` (fixed in `93a578f`). Not exercised: REL12 BYOD, Section 10. |
| 2026-09-19 | `3220429` / `d2f10303` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-t5-base-text2text` v2; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` after, Python 3.12.13, `cuda:0`) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout (blob SHA-1 verified against GitHub before execution) | 477.4 s | **PASSED** — 11/11 code cells ok (1 restart after install cell); 17 files, 899 MB staged from the Hub into a clean cache; comparison {rouge1: {lead1: 16.73, frozen_summarize: 21.68, frozen_new_prefix: 0.71, adapted: 29.78}, rouge2: {lead1: 4.83, frozen_summarize: 8.53, frozen_new_prefix: 0.67, adapted: 14.84}, rougeL: {lead1: 14.08, frozen_summarize: 18.99, frozen_new_prefix: 0.71, adapted: 28.21}, mean_output_words: {lead1: 21.41, frozen_summarize: 16.7, frozen_new_prefix: 1.03, adapted: 6.93}, delta_vs_frozen: {rouge1: 29.08, rouge2: 14.17, rougeL: 27.51}}; reload parity {identical_outputs: 6, of: 6}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-t5-base-text2text/v2/evidence/` in the workspace |
| 2026-09-19 | `e09a306` / `486beac4` | Local pre-flight harness (Windows, CPython 3.12.10, CPU float32, `torch 2.14.0+cu130` with `CUDA_VISIBLE_DEVICES=-1`, `transformers 4.57.6`) | Default sample path (install skipped, pins pre-installed → three carried modules → inline manifest assert → `stage_missing_files` fetched 0 of 7 entries because the snapshot was pre-staged → `verify_snapshot` 7 files → `from_pretrained` on CPU → `fetch_corpus` served from the pre-staged cache after its digest checks → 1,992 + 619 + 618 papers read, `prefix_is_trained` `False`, 300 / 50 / 100 drawn with `check_split_disjoint` clean and digests `eadcee41…` / `6139c992…` / `ea8cb17c…` → four dataset refusals → input manifest + `num_beams` refusal probe → German translation `Das Haus ist wunderbar.` and a `summarize: ` output with every sanity check `True`, new-prefix probe `True` → Lead-1 → frozen under both prefixes → `adapt` → validation + test evaluation → four unseen abstracts + German re-check → adapter export → reload parity) | 467.7 s | **PASSED** — 11/11 code cells; Lead-1 ROUGE-1/2/L 16.73 / 4.83 / 14.08; frozen under `paper title: ` 0.71 / 0.67 / 0.71 at 1.0 words (26.0 s); frozen under `summarize: ` 21.68 / 8.53 / 18.99 at 16.7 words (63.9 s); titles average 7.4 words; `adapt` 37,757,952 of 222,903,552 params, 300 abstracts, 2 epochs, 283.8 s, validation ROUGE-L 0.73 → 33.11 → 35.32 (`best_epoch` 2, train loss 2.882 → 2.223); **adapted test 31.61 / 16.40 / 29.97 at 6.5 words (Δ +30.91 / +15.73 / +29.27 vs the frozen new prefix)**; four unseen abstracts 39.15 / 33.78 / 39.15 `measured-small-sample`, single-item report `sample-sanity`; German translation unchanged after adaptation (`Das Haus ist wunderbar.`); adapter 151,037,976 B / 52 tensors, SHA-256 `4066061e…`; reload parity 6/6; six exports written. Pre-flight; hosted clean-runtime run still required |
| 2026-09-14 | `198685c` / `fbc9be0e4069` (`TASK-INFERENCE`, superseded) | Kaggle CPU (`kurtvalcorza/dimer-nb2-t5-base-text2text` v1) | Default sample path of the inference-only notebook: two synthetic prefixed inputs, `stage_missing_files` fetching `model.safetensors` from the Hub, `verify_snapshot`, `generate`, `not-measurable` report | 285.0 s | **PASSED** — 8/8 code cells, 14 files, 894 MB staged; does not cover the `E2E` blob |

### Hosted Colab T4 runs of blob `4161fa32` (2026-10-29)

These runs executed the PR head `65e616bd9c23fe738155966be658900697b3409b`, notebook `tutorials/t5_base_text2text_colab.ipynb` blob `4161fa3213094ccba6b45884d9fcc065811308f3`, fetched at the 40-character SHA and Git-blob verified, with the Colab CLI (`colab exec -f`) on a fresh Colab Tesla T4 session (not a browser `Run all`). Each `run_summary.json` records `verdict` `PASS`, `exec_exit_code` 0, `input_unchanged` true and `stop_verified` true (the VM was released), and the SHA-256 of each executed notebook equals the summary's `executed_sha256`.

**Default path (2026-10-29 22:15:03 UTC start).** Default settings, no overrides; session `suite-t5-65e616b-8dff`; wall 303.4 s (`wall_s`); `code_cells` 13, `executed_cells` 10, 0 errors. Retained evidence: [`docs/execution-evidence/2026-10-29-65e616b/`](execution-evidence/2026-10-29-65e616b/).

Printed by the default run (verbatim lines from the executed outputs):

```text
{'verified_files': 6, 'revision': 'a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1'}
{'lead1_baseline': {'rouge1': 16.73, 'rouge2': 4.83, 'rougeL': 14.08, 'mean_output_words': 21.4, 'n': 100}}
{'frozen_model_summarize_test': {'rouge1': 21.68, 'rouge2': 8.53, 'rougeL': 18.99, 'mean_output_words': 16.7, 'known_prefix': 'summarize: '}, 'seconds': 51.2}
{'rouge1': {'lead1': 16.73, 'frozen_summarize': 21.68, 'frozen_new_prefix': 0.71, 'adapted': 29.78}}
{'rouge2': {'lead1': 4.83, 'frozen_summarize': 8.53, 'frozen_new_prefix': 0.67, 'adapted': 14.84}}
{'rougeL': {'lead1': 14.08, 'frozen_summarize': 18.99, 'frozen_new_prefix': 0.71, 'adapted': 28.21}}
{'delta_vs_frozen': {'rouge1': 29.08, 'rouge2': 14.17, 'rougeL': 27.51}}
{'reload_parity': {'identical_outputs': 6, 'of': 6}, 'reloaded_prefix': 'paper title: ', 'reloaded_best_epoch': 2}
```

**DAT20 refusal record (2026-10-29, blob `4161fa32`).** Overrides, set by `--set` on a copy of the blob: `USE_BYOD` `False` → `True` and `BYOD_PATH` `''` → `'/content/sample_data/california_housing_test.csv'` (the run copy differs from the blob in exactly those 2 source lines). The suite cut the run copy right after the refusing cell `t5_base_text2text-16`, dropping 20 cells (6 of them code cells), which reproduces `Run all` stopping at the first error; the cut copy is `run_copy.ipynb`, SHA-256 `4468c57ce999c0e55f197f54b2ff09f1a40558fb4409e238da6133a63ca696c9` (Git blob `4e0d21e7`), equal to the summary's `run_copy_sha256`. The cut copy's earlier code cells completed without an error output; the refusing cell's only error output has `ename` `IsolatedCellRefusal` and no traceback, and its output holds exactly one `Refused:` line:

```text
Refused: california_housing_test.csv: CSV is missing columns ['id', 'source', 'target'] (it needs id, source, target); fix the file and run this cell again (the earlier results of this session were removed).
```

This run says nothing about the 20 cut cells, which did not run in it. Session `suite-t5-65e616b-8dff`, wall 122.1 s. Retained evidence: [`docs/execution-evidence/2026-10-29-65e616b-refusal/`](execution-evidence/2026-10-29-65e616b-refusal/).

**Local-path redaction.** Local paths were replaced byte-wise by `<local-run-dir>` (and the Colab CLI install path by `<colab-cli-tool-dir>/`) before the first commit; each executed notebook and run copy was checked against its `run_summary.json` before any edit.

| File (under `docs/execution-evidence/2026-10-29-65e616b/`) | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `t5_base_text2text_colab_65e616b_colab-cli-t4_default.ipynb` | `446c010b619a251c9dcb82d3c594b4c0670da6a97d0e286f542b5da0418ab771` | `446c010b619a251c9dcb82d3c594b4c0670da6a97d0e286f542b5da0418ab771` (unchanged, no local path) |
| `exec.log` | `b1b5caab9e059ddea77e98550e60c63176358f7ceabdbcec6b394f8061e8ea8c` | `67775edf0df0a51139e33ff1217d194b58f5dd8ed627b92be3cf15425835d540` (2 local-path replacements) |
| `run_summary.json` | `4c0426809a950b918bf5b6f878ad9424e79005b2bd4d5fa42b18f8cee16273e3` | `4c0426809a950b918bf5b6f878ad9424e79005b2bd4d5fa42b18f8cee16273e3` (unchanged, no local path) |

| File (under `docs/execution-evidence/2026-10-29-65e616b-refusal/`) | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `t5_base_text2text_colab_65e616b_colab-cli-t4_refusal.ipynb` | `d4f47336074b3fd371cd90527cea8b4520a08c1bdc5017e3e5a32953d8aedde0` | `d4f47336074b3fd371cd90527cea8b4520a08c1bdc5017e3e5a32953d8aedde0` (unchanged, no local path) |
| `exec.log` | `ddbc35dc3011b0cdd11e759095810409f2a96ef6fac58534209d0181e045e857` | `0e12e92b163048f548a4c1eee371e5cd2ae81ccda64e488c35a12ff57d87d8f4` (2 local-path replacements) |
| `run_summary.json` | `605280f32d379c7bb96a695f6c4436f5c7de5590d882c4da1df4c9d00da0dd95` | `605280f32d379c7bb96a695f6c4436f5c7de5590d882c4da1df4c9d00da0dd95` (unchanged, no local path) |
| `run_copy.ipynb` | `4468c57ce999c0e55f197f54b2ff09f1a40558fb4409e238da6133a63ca696c9` | `4468c57ce999c0e55f197f54b2ff09f1a40558fb4409e238da6133a63ca696c9` (unchanged, no local path) |

Not exercised by these runs: an accepted BYOD input (REL12), a browser `Run all` of the interactive journeys (REL2), and every other open gate named in this file; these runs close none of them.

## Current status

**Candidate (2026-10-26, DAT20 refusal display).** The current blob `4161fa32` (code commit `960ec12`) shows a refused BYOD input as one `Refused: <file>: <reason>` line with no traceback, removes the earlier run's results from the session and this notebook's exports from `outputs/`, and stops Run all at that cell, while genuine failures keep their traceback (NOTEBOOK_SPEC DAT19/DAT20); its default path (2026-10-29; 13 code cells, exit 0, 0 errors) and its scripted DAT20 refusal (2026-10-29; `USE_BYOD=True;BYOD_PATH=/content/sample_data/california_housing_test.csv`: exactly one `Refused:` line at cell `t5_base_text2text-16`, no traceback, earlier cells clean, VM released) both passed on a hosted Colab Tesla T4 (Colab CLI `colab exec -f`, not a browser `Run all`), and the hosted runs of earlier blobs, including the run of `ac38dca05ca7`, are history for it (REL14). The status below is history.

**Candidate (2026-10-25).** The current notebook `tutorials/t5_base_text2text_colab.ipynb` is blob `ac38dca05ca77b15f58f8a44e4643512d33f542f` (commit `ba60b61`), generated at `a03787d` (T89-1..T89-4); it changes markdown, the use notice, `metadata.dimer` and the `NOTEBOOK_SOURCE` record only. That exact blob completed one pass with no restart and 0 errors on a fresh Colab Tesla T4 on 2026-10-25 (Colab CLI sequential execution, 13/13 code cells, 296.3 s wall; validation ROUGE-L 0.73 → 35.39 at the kept epoch 2; test ROUGE-L Lead-1 14.08 / `summarize:` 18.99 / frozen new prefix 0.71 / adapted 28.21, verdict `improved`; reload parity 6/6; evidence in `docs/execution-evidence/2026-10-25-ba60b61/`) — the same figures as the previous blob's run. The paths this blob's runs cover: the default path only. The remaining gates are REL12 (a hosted BYOD run) and REL2 (a browser `Run all`). The runs of earlier blobs below are history (REL14).

Evidence files for the 2026-10-25 default run of blob `ac38dca0`, copied from the run directory and redacted byte-wise (local run paths replaced by `<local-run-dir>`, the Colab CLI install path by `<colab-cli-tool-dir>`; no JSON re-serialised; the executed notebook's captured hash equals `run_summary.json` `executed_sha256`):

| File (under `docs/execution-evidence/2026-10-25-ba60b61/`) | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `t5_base_text2text_colab_ba60b61_colab-cli-t4_default.ipynb` | `b89a388fb6741614cf4ea0a325613a882f196f936062d2a5b1484e2fb4ddc15c` | `b89a388fb6741614cf4ea0a325613a882f196f936062d2a5b1484e2fb4ddc15c` (unchanged, no local path) |
| `exec.log` | `daccfdd2336f0d07a23dbae2f69f7165dd4926a3149ff57dc290d014f3ef0460` | `96f368fba06ad8713a2df84ad85d929a98254d0ca9d8179b2952f8fd580803b6` (2 local-path replacements) |
| `run_summary.json` | `1df28b2886b159f8d6f3ff9c2f90285242c2b331e52aedaafdb5fe9015b70664` | `1df28b2886b159f8d6f3ff9c2f90285242c2b331e52aedaafdb5fe9015b70664` (unchanged, no local path) |

**Prose check (2026-10-25).** Every number the notebook prose quotes from a recorded run was compared with what this run printed; none differs (timings and CPU-only figures that name another blob or runtime are labelled history and were not compared).


**Candidate.** The 2026-09-19 Kaggle T4 run of blob `d2f10303` (commit `3220429`) needed a manual restart after the install cell (11/11 ok (1 restart after install cell)), so it is not a one-pass `Run all` and not promotion evidence (review finding T5B-M1); it is kept above as history. The review-fix blob `09d88ee9a276` (commit `93a578f`; isolated `uv` environment, restore-from-base adaptation, recorded verdicts, guided layer, T5B-M1..M4 / T5B-m1..m2 fixes) completed one pass with no restart and 0 errors on a fresh Colab Tesla T4 on 2026-10-09 (Colab CLI 0.7.4 sequential execution, 13/13 code cells, 331.3 s; validation ROUGE-L 0.73 frozen -> 35.39 at the kept epoch 2, started from the pinned base; test ROUGE-L Lead-1 14.08 / `summarize:` 18.99 / frozen new prefix 0.71 / adapted 28.21, verdict improved; reload parity 6/6). REL12 BYOD exercise (path-based and upload) and the optional Section 10 experiment are pending on a hosted runtime, so the status stays Candidate (REL14). Any later change to the carried modules or to the notebook produces a new blob that needs its own recorded run; promotion is not performed by the builder.
