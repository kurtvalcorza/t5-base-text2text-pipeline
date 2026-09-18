"""Model-backed checks that run only where the pinned snapshot is staged (local pre-flight): a
referenced evaluation under a new prefix, a one-epoch adaptation of the last decoder block on a dozen
abstracts, and the artifact round trip. Skipped when the weights are absent."""

from __future__ import annotations

import json

import pytest

from t5_base_text2text_pipeline import DEFAULT_WEIGHTS_DIR, WEIGHT_FILE, T5BaseText2TextPipeline

pytest.importorskip("transformers")
if not (DEFAULT_WEIGHTS_DIR / WEIGHT_FILE).is_file():
    pytest.skip("snapshot not staged", allow_module_level=True)

PAPERS = [
    (
        "We study graph neural networks for molecule property prediction. Our model beats three baselines.",
        "Graph Networks for Molecule Property Prediction",
    ),
    (
        "Transformers are hard to train on small data. "
        "We propose a curriculum that orders examples by length.",
        "A Length Curriculum for Small-Data Transformers",
    ),
    (
        "Reinforcement learning agents forget old tasks. We add a replay buffer with prioritised sampling.",
        "Prioritised Replay Against Forgetting",
    ),
    (
        "Speech recognition degrades with accents. We fine-tune on accented data with adapters.",
        "Adapters for Accented Speech Recognition",
    ),
    (
        "Image captioning models hallucinate objects. We penalise captions naming absent objects.",
        "Grounded Penalties for Caption Hallucination",
    ),
    (
        "Sparse attention scales to long documents. "
        "We route tokens to experts by locality-sensitive hashing.",
        "Hash-Routed Sparse Attention",
    ),
    (
        "Tabular data resists deep learning. We pretrain a transformer on synthetic tables.",
        "Synthetic-Table Pretraining for Tabular Transformers",
    ),
    (
        "Machine translation for low-resource languages lacks data. We back-translate monolingual text.",
        "Back-Translation for Low-Resource Translation",
    ),
    (
        "Protein structure prediction is costly. We distil a large model into a small one.",
        "Distilling Protein Structure Predictors",
    ),
    (
        "Robots grasp unfamiliar objects poorly. We learn grasps from simulated point clouds.",
        "Simulated Point Clouds for Robot Grasping",
    ),
    (
        "Recommender systems amplify popularity bias. We reweight the loss by item frequency.",
        "Frequency Reweighting for Long-Tail Recommendation",
    ),
    (
        "Code models struggle with long files. We add retrieval over the repository.",
        "Repository Retrieval for Code Completion",
    ),
]
RECORDS = [{"id": f"p{i:02d}", "source": s, "targets": [t]} for i, (s, t) in enumerate(PAPERS)]
GEN = {"max_new_tokens": 16, "num_beams": 2}
PREFIX = "paper title: "


@pytest.fixture(scope="module")
def pipe():
    return T5BaseText2TextPipeline.from_pretrained(device="cpu")


def test_evaluate_scores_references_under_a_prefix(pipe):
    metrics = pipe.evaluate(RECORDS[:3], prefix=PREFIX, **GEN)
    assert metrics["n"] == 3 and 0.0 <= metrics["rouge1"] <= 100.0 and metrics["adapted"] is False
    assert metrics["known_prefix"] is None


def test_one_epoch_adaptation_and_artifact_round_trip(pipe, tmp_path):
    result = pipe.adapt(
        RECORDS[:8],
        RECORDS[8:11],
        prefix=PREFIX,
        epochs=1,
        trainable_decoder_layers=1,
        batch_size=4,
        eval_generation=GEN,
    )
    assert result["n_trainable"] == 9_439_488 and result["history"][0]["note"] == "frozen model"
    assert result["prefix"] == PREFIX
    assert all(name.startswith("decoder.block.11.") for name in result["trainable_names"])
    artifact = pipe.save_artifact(tmp_path / "adapter", {"note": "test"})
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert (
        len(manifest["tensors"]) == len(result["trainable_names"]) and manifest["adapter"]["prefix"] == PREFIX
    )
    reloaded = T5BaseText2TextPipeline.from_artifact(artifact, device="cpu")
    a = [pipe.generate(PREFIX + r["source"], **GEN)["text"] for r in RECORDS[:2]]
    b = [reloaded.generate(PREFIX + r["source"], **GEN)["text"] for r in RECORDS[:2]]
    assert a == b
    assert reloaded.adapter["best_epoch"] == result["best_epoch"]
