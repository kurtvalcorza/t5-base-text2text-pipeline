"""Regression tests for the 2026-10-05 notebook review findings (T5B-M1..M4, T5B-m1, T5B-m2).

Every test needs only CI's dependencies and no model: the notebook's own cell sources are executed with stand-ins
where a model would be needed. Stand-in evidence is plumbing evidence, not model evidence.
"""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from t5_base_text2text_pipeline import T5BaseText2TextPipeline
from t5_base_text2text_pipeline import samples as sm

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "t5_base_text2text_colab.ipynb"
LOCK = ROOT / "tutorials" / "requirements-colab.lock.txt"
PIPELINE = ROOT / "src" / "t5_base_text2text_pipeline" / "pipeline.py"
STEM = "t5_base_text2text"


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_cells(notebook: dict) -> list[dict]:
    return [c for c in notebook["cells"] if c["cell_type"] == "code"]


def _cell(notebook: dict, marker: str) -> str:
    found = [c["source"] for c in _code_cells(notebook) if marker in c["source"]]
    assert len(found) == 1, f"expected one code cell containing {marker!r}, found {len(found)}"
    return found[0]


def _markdown(notebook: dict) -> str:
    return "\n".join(c["source"] for c in notebook["cells"] if c["cell_type"] == "markdown")


# --- T5B-M1: no in-kernel install, no restart, idempotent Section 1 ------------------------------------------


def test_t5b_m1_nothing_is_pip_installed_into_the_kernel_and_no_restart_is_requested(notebook):
    code = "\n".join(c["source"] for c in _code_cells(notebook))
    assert "pip install" not in code and "'-m', 'pip'" not in code
    assert "Restart the runtime" not in json.dumps(notebook)
    kernel = [c for c in _code_cells(notebook) if "# dimer: kernel cell" in c["source"]]
    assert len(kernel) == 1, "exactly one cell may run in the kernel"
    source = kernel[0]["source"]
    for needed in ("'--require-hashes', '--only-binary', ':all:'", "'--managed-python'", "UV_SHA256", "LOCK_SHA256", 'MPLBACKEND="Agg"', '"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"'):
        assert needed in source


def test_t5b_m1_carried_lock_is_the_committed_lock_and_pins_every_runtime_pin(notebook):
    source = _cell(notebook, "# dimer: kernel cell")
    lock_text = LOCK.read_text(encoding="utf-8")
    digest = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    assert digest == hashlib.sha256(lock_text.encode("utf-8")).hexdigest()
    assert f"LOCK_TEXT = r'''{lock_text}'''" in source
    spec = importlib.util.spec_from_file_location("_review_build_notebook", ROOT / "tools" / "build_notebook.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    build.check_lock(build._pins(ROOT), lock_text)


def test_t5b_m1_section_1_is_idempotent_and_keeps_the_live_worker(notebook, tmp_path, monkeypatch, capsys):
    """The real Section 1 cell, run twice with a stand-in interpreter: the matching environment is reused (no
    download) and the live worker — with every variable later cells created — is kept."""
    source = _cell(notebook, "# dimer: kernel cell")
    lock_sha = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    env = tmp_path / "env"
    (env / "bin").mkdir(parents=True)
    try:
        (env / "bin" / "python").symlink_to(sys.executable)
    except OSError as exc:  # Windows without the symlink privilege (WinError 1314); the cell targets Linux runtimes
        pytest.skip(f"cannot create a symlink here: {exc}")
    (env / ".dimer-lock-sha256").write_text(lock_sha + "\n", encoding="utf-8")
    monkeypatch.setenv("DIMER_ISOLATED_ENV", str(env))
    monkeypatch.delenv("DIMER_NOTEBOOK_CI_PREINSTALLED", raising=False)
    shell = types.SimpleNamespace(input_transformers_cleanup=[])
    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: shell
    ipython_display = types.ModuleType("IPython.display")
    ipython_display.display = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", ipython_display)

    def no_download(*args, **kwargs):
        raise AssertionError("a matching environment must be reused, not downloaded again")

    monkeypatch.setattr("urllib.request.urlopen", no_download)
    namespace: dict = {"__name__": "__main__"}
    exec(compile(source, "<section 1>", "exec"), namespace)
    runtime = namespace["_DIMER_ISOLATED_RUNTIME"]
    try:
        assert "'reused': True" in capsys.readouterr().out
        runtime.run("learner_value = 41 + 1\n")
        exec(compile(source, "<section 1>", "exec"), namespace)  # the learner re-runs Section 1 on its own
        assert namespace["_DIMER_ISOLATED_RUNTIME"] is runtime and runtime.alive()
        assert [t.__name__ for t in shell.input_transformers_cleanup] == ["_route_to_isolated_runtime"]
        runtime.run("print('value', learner_value)\n")
        assert "value 42" in capsys.readouterr().out
        assert namespace["_route_to_isolated_runtime"](["x = 1\n"]) == ["_DIMER_ISOLATED_RUNTIME.run('x = 1\\n')\n"]
        assert namespace["_route_to_isolated_runtime"]([source]) == [source]
    finally:
        runtime.close()


# --- T5B-M2: every adaptation starts from the pinned base -------------------------------------------------------


def test_t5b_m2_adapt_and_load_artifact_restore_the_base_first():
    text = PIPELINE.read_text(encoding="utf-8")
    adapt = text[text.index("    def adapt(") : text.index("    # ---- artifacts")]
    order = [adapt.index(m) for m in ("touched = wanted | set(self._base_state)", "self._remember_base(model, names)", "restored = self.restore_base()", '"note": "frozen model"', "initial_state = {", "for epoch in range(1, epochs + 1):")]
    assert order == sorted(order)
    assert "restore.update(previous_state)" in adapt  # a failed call leaves the weights as before
    load = text[text.index("    def load_artifact(") : text.index("    def from_artifact(")]
    assert load.index("self.restore_base()") < load.index("model.load_state_dict(merged, strict=True)")


class _Tensor(np.ndarray):
    def detach(self):
        return self

    def clone(self):
        return self.copy().view(_Tensor)


class _FakeModel:
    """Stand-in model (numpy arrays, not torch) with the state-dict surface restore_base uses."""

    def __init__(self):
        self.params = {"decoder.block.10.w": np.zeros(3).view(_Tensor), "decoder.block.11.w": np.ones(3).view(_Tensor)}

    def state_dict(self):
        return dict(self.params)

    def load_state_dict(self, state, strict=True):
        for key, value in state.items():
            self.params[key][...] = value

    def eval(self):
        return self


def test_t5b_m2_restore_base_undoes_every_earlier_change_stand_in():
    model = _FakeModel()
    pipe = T5BaseText2TextPipeline(lambda *a: ("", 0, "eos"), len, _model=model, _tokenizer=object())
    names = sorted(model.params)
    pipe._remember_base(model, names)
    model.params[names[0]] += 5  # an earlier adaptation
    pipe.adapter = {"best_epoch": 3}
    assert pipe.restore_base() == [names[0]] and pipe.adapter is None  # only the changed tensor is reported
    assert pipe.restore_base() == []  # nothing differs from the base any more
    assert (model.params[names[0]] == 0).all() and (model.params[names[1]] == 1).all()
    pipe._remember_base(model, names)  # a second call keeps the first (base) values
    model.params[names[1]] += 2
    pipe.restore_base()
    assert (model.params[names[1]] == 1).all()


def test_t5b_m2_byod_rerun_restores_the_base_and_the_experiment_has_its_own_pipeline(notebook):
    section_4 = _cell(notebook, "USE_BYOD = False")
    assert section_4.index("pipe.restore_base()") < section_4.index("if USE_BYOD:")
    experiment = _cell(notebook, "RUN_EXPERIMENT = False")
    assert "experiment_pipe = T5BaseText2TextPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, device=pipe.device)" in experiment
    assert f"Path('outputs/{STEM}_experiment')" in experiment
    assert "raise RuntimeError(f'the experiment changed a default export: {unchanged}')" in experiment
    assert not re.search(r"(?<!experiment_)pipe\.adapt\(", experiment)
    assert "**Predict → Change one thing → Run → Observe → Explain**" in _markdown(notebook)
    assert "they do not affect the default path" not in _markdown(notebook)


# --- T5B-M3: quality outcomes are reported verdicts -------------------------------------------------------------


def test_t5b_m3_no_quality_assert_remains(notebook):
    code = "\n".join(c["source"] for c in _code_cells(notebook) if not c["metadata"].get("dimer", {}).get("embedded_module"))
    assert re.findall(r"(?m)^\s*assert .*$", code) == ["assert parity['identical_outputs'] == parity['of']"]


def _rouge(r):
    return {"rouge1": r, "rouge2": r / 2, "rougeL": r, "mean_output_words": 5.0, "mean_reference_words": 7.0, "n": 4, "verdict": "measured-small-sample", "generation": {}, "known_prefix": None, "hit_token_ceiling": 0, "definitions": {}}


def test_t5b_m3_negative_results_are_recorded_and_do_not_stop_the_notebook(notebook, tmp_path, monkeypatch):
    """Sections 6 and 8 executed with stand-ins: a Lead-1 baseline sharing no word with the references (a translation
    task, say) and an adapted model worse than the frozen one complete and record their findings (no model)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    scores = iter([_rouge(30.0), _rouge(20.0), _rouge(10.0), _rouge(12.0)])
    pipe = types.SimpleNamespace(evaluate=lambda *a, **k: next(scores), generate=lambda *a, **k: {"text": "x"})
    record = {"id": "r0", "source": "a source", "targets": ["a target"]}
    ns = {
        "pipe": pipe, "test_records": [record, record], "val_records": [record], "PREFIX": "paper title: ", "GEN": {}, "time": __import__("time"), "json": json,
        "lead_baseline": lambda records, n_sentences=1: _rouge(0.0),
        "MODEL_ID": "stand-in", "MODEL_REVISION": "0" * 40, "MODEL_KEY": "stand-in", "data_source": "stand-in",
        "dataset_manifests": {"test": {"digest": "d"}}, "disjoint": {}, "adapt_result": {"history": []}, "adapt_seconds": 0.0,
    }
    exec(_cell(notebook, "baseline_lead1 = lead_baseline("), ns)
    assert ns["lead1_overlap"].startswith("Lead-1 shares no unigram")
    exec(_cell(notebook, "adapted_test = pipe.evaluate("), ns)
    report = json.loads((tmp_path / "outputs" / f"{STEM}_evaluation_report.json").read_text(encoding="utf-8"))
    assert report["comparison"]["verdicts"]["adapted_vs_frozen_rougeL"] == "worse"


# --- T5B-M4: guided layer and infrastructure labelling ----------------------------------------------------------


def test_t5b_m4_guided_layer_is_present(notebook):
    markdown = _markdown(notebook)
    for heading in ("**Who this notebook is for.**", "**Input → Model → Output.**", "**How to use this notebook.**", "**Roadmap:**", "## Troubleshooting", "## Glossary", "## Conclusion (your notes)", "## 10. Change one thing", "**Learner:**"):
        assert heading in markdown, heading
    assert markdown.count("**Predict") >= 7
    assert markdown.count("<details><summary>Check your reasoning</summary>") >= 7
    assert markdown.count("**What to notice:**") >= 6


def test_t5b_m4_infrastructure_cells_are_labelled_and_collapsed(notebook):
    infra = [c for c in _code_cells(notebook) if c["metadata"].get("cellView") == "form"]
    assert len([c for c in infra if c["metadata"].get("dimer", {}).get("embedded_module")]) == 3
    titled = [c["source"].splitlines()[0] for c in infra if not c["metadata"].get("dimer")]
    assert len(titled) == 3 and all(t.startswith("# @title Infrastructure:") for t in titled), titled


def test_t5b_m4_no_template_placeholders_leak(notebook):
    learner = "\n".join(c["source"] for c in notebook["cells"] if not c.get("metadata", {}).get("dimer", {}).get("embedded_module"))
    for leftover in ("{{", "{MODEL_ID}", "{stem}", "@P:"):
        assert leftover not in learner, leftover
    assert "}}" not in _markdown(notebook)


# --- T5B-m1 / T5B-m2: BYOD contract and the opening cells ------------------------------------------------------


def _csv(path: Path, n: int, *, bom: bool = False, source_prefix: str = "") -> Path:
    rows = "id,source,target\n" + "".join(f"r{i},{source_prefix}An abstract about topic {i} and its methods.,Title {i}\n" for i in range(n))
    path.write_bytes(("\ufeff" if bom else "").encode("utf-8") + rows.encode("utf-8"))
    return path


def test_t5b_m1_stated_minimum_is_what_the_split_accepts(tmp_path):
    assert sm.min_byod_records() == {"total": 12, "train": 8, "validation": 2, "test": 2}
    split = sm.split_dataset(sm.load_byod_dataset(_csv(tmp_path / "ok.csv", 12)), seed=42)
    assert {k: len(v) for k, v in split.items()} == {"test": 2, "validation": 2, "train": 8}
    with pytest.raises(ValueError, match=r"training split with 7 record\(s\).*supply at least 12 distinct sources"):
        sm.split_dataset(sm.load_byod_dataset(_csv(tmp_path / "small.csv", 11)), seed=42)


def test_t5b_m1_bom_header_only_and_prefixed_sources(tmp_path):
    assert len(sm.load_byod_dataset(_csv(tmp_path / "bom.csv", 12, bom=True))) == 12
    (tmp_path / "header.csv").write_text("id,source,target\n", encoding="utf-8")
    with pytest.raises(ValueError, match="header but no data rows"):
        sm.load_byod_dataset(tmp_path / "header.csv")
    with pytest.raises(ValueError, match=r"CSV line 2 \(id 'r0'\): the source already starts with the task prefix 'summarize:'"):
        sm.load_byod_dataset(_csv(tmp_path / "pre.csv", 12, source_prefix="summarize: "))
    with pytest.raises(ValueError, match="task prefix 'paper title:'"):
        sm.load_byod_dataset(_csv(tmp_path / "mine.csv", 12, source_prefix="paper title: "), prefix="paper title: ")
    (tmp_path / "bad.jsonl").write_text('{"id": "a", "source": "x", "target": "y"}\n{"id": "b", "source": ""}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"JSONL line 2: "):
        sm.load_byod_dataset(tmp_path / "bad.jsonl")


def _section_4(notebook: dict, path: str) -> str:
    source = _cell(notebook, "USE_BYOD = False")
    source = source.replace("USE_BYOD = False  # @param", "USE_BYOD = True  # @param", 1)
    return source.replace("BYOD_PATH = ''  # @param", f"BYOD_PATH = {path!r}  # @param", 1)


def _section_4_namespace(restored: list) -> dict:
    from t5_base_text2text_pipeline import pipeline as pl

    ns = {k: getattr(pl, k) for k in dir(pl) if not k.startswith("__")}
    ns.update({k: getattr(sm, k) for k in dir(sm) if not k.startswith("__")})
    pipe = types.SimpleNamespace(adapter={"best_epoch": 2}, restore_base=lambda: restored.append(True) or ["a"])
    ns.update({"os": __import__("os"), "Path": Path, "pipe": pipe, "__name__": "__main__"})
    return ns


def test_t5b_m1_byod_path_runs_section_4_outside_colab_from_the_base(notebook, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _csv(tmp_path / "mine.csv", 12, bom=True)
    restored: list = []
    ns = _section_4_namespace(restored)
    exec(_section_4(notebook, "mine.csv"), ns)
    out = capsys.readouterr().out
    assert restored == [True], "a BYOD re-run must put the model back to the pinned base first"
    assert ns["raw_papers"] == {"byod": 12, "duplicate_sources_dropped": 0, "effective_minimum": 12}
    assert ns["disjoint"] == {"test": 2, "validation": 2, "train": 8} and "only 2 held-out test records" in out


def test_t5b_m1_upload_outside_colab_cancelled_and_bad_path_are_explained(notebook, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "google", None)
    with pytest.raises(RuntimeError, match="upload dialog exists only in Google Colab"):
        exec(_section_4(notebook, ""), _section_4_namespace([]))
    with pytest.raises(FileNotFoundError, match="BYOD_PATH 'nowhere.csv' is not a file"):
        exec(_section_4(notebook, "nowhere.csv"), _section_4_namespace([]))
    for uploaded, message in (({}, "received 0"), ({"a.csv": b"", "b.csv": b""}, "received 2")):
        google, colab, files = (types.ModuleType(n) for n in ("google", "google.colab", "google.colab.files"))
        files.upload = lambda uploaded=uploaded: uploaded
        colab.files, google.colab = files, colab
        for name, module in (("google", google), ("google.colab", colab), ("google.colab.files", files)):
            monkeypatch.setitem(sys.modules, name, module)
        with pytest.raises(ValueError, match=message):
            exec(_section_4(notebook, ""), _section_4_namespace([]))


def test_t5b_m2_no_escaped_braces_and_both_hosts_are_named(notebook):
    markdown = _markdown(notebook)
    assert "{{" not in markdown and "}}" not in markdown
    assert "`[A-Za-z0-9_.:-]{1,64}`" in markdown and "`{id, source, targets}`" in markdown
    access = next(line for line in markdown.splitlines() if line.startswith("- **External access:**"))
    assert "huggingface.co" in access or "Hugging Face Hub" in access
    assert "raw.githubusercontent.com" in access and "No GitHub access" not in markdown
