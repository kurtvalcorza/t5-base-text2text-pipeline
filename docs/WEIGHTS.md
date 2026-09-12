# Weight provenance and DIMER hosting

- Upstream: `google-t5/t5-base`
- Immutable revision: `a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1`
- Weight format: SafeTensors (`model.safetensors`, 891,646,390 bytes, float32)
- Upstream weight license: Apache-2.0 (`license: apache-2.0` in the pinned upstream README front matter)
- Local layout: `weights/t5-base/` holds the 6 files listed in `dimer-base-manifest.json` (`config.json`, `generation_config.json`, `model.safetensors`, `spiece.model`, `tokenizer.json`, upstream `README.md` — no `tokenizer_config.json` exists upstream at this revision, so `T5TokenizerFast` is built from `tokenizer.json`; `totalBytes` 893,837,231) with byte size and SHA-256 for each. `verify_snapshot()` in `src/t5_base_text2text_pipeline/pipeline.py` checks all of them before any load; `stage_missing_files(allow_download=True)` fetches only absent entries at the pinned revision into that directory. `.safetensors` files are git-ignored; the Git repository does not vendor the checkpoint.
- DIMER hosting: Apache-2.0 permits use, modification, redistribution and commercial use subject to preservation of the license and notices. DIMER may mirror the pinned checkpoint in its model store under those terms.
- Loader trust boundary: Transformers `T5ForConditionalGeneration` + `T5TokenizerFast` with `trust_remote_code=False`; the loader reads only the verified local directory (`local_files_only=True`) and falls back to the Hub at the pinned revision only when `allow_download=True` is passed explicitly.
