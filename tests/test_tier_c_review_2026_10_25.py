"""Tier C review findings T89-1..T89-4 (2026-10-25)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "tutorials" / "t5_base_text2text_colab.ipynb"


def _nb():
    return json.loads(NB.read_text(encoding="utf-8"))


def _markdown():
    md = "\n".join("".join(c["source"]) for c in _nb()["cells"] if c["cell_type"] == "markdown")
    return " ".join(md.split())


def test_t89_1_dat9_and_independence_caveat():
    md = _markdown()
    assert "**Overlap with the training data:**" in md and "inferred, not verified" in md
    assert "evaluate on an independent" not in md and "on an independent split" not in md


def test_t89_2_numbers_name_their_run():
    md = _markdown()
    assert "the Kaggle T4 run of 19 September 2026" not in md and "about 33 after the first epoch" not in md
    assert "30.88 after the first epoch and 35.39" in md and "`09d88ee9`" in md


def test_t89_3_tuning_basis_disclosed():
    md = _markdown()
    assert "build record" not in md and "does not record which split" in md


def test_t89_4_spec_items():
    nb = _nb()
    dimer = nb["metadata"]["dimer"]
    assert dimer["notebook_spec"] == "2.5" and dimer["notebook_kind"] == "TUTORIAL"
    first_code = next(i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code")
    head = "\n".join("".join(c["source"]) for c in nb["cells"][:first_code])
    assert "Notice.** For research, training and evaluation only." in head
    md = _markdown()
    assert "[1] C. Raffel" in md and "`DIMER_ISOLATED_ENV`" in md
