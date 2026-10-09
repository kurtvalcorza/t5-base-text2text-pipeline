"""Text-to-text dataset contract for teaching a new task prefix: the pinned SciTLDR sample (abstract →
paper title), validation, seeded splitting, BYOD loaders and CSV export.

The default dataset is **real** and defines a task the pinned checkpoint was never trained on: SciTLDR
(Cachola et al., EMNLP Findings 2020; Apache-2.0) ships, for every computer-science paper, the abstract and
the paper's title. The three `SciTLDR-A` JSON-Lines files are fetched one by one from the project repository
at a pinned commit and refused on any byte-size or SHA-256 mismatch; the tutorial pairs each abstract with its
title under a **new prefix** (`paper title: `) that the caller chooses. The frozen model answers an unknown
prefix with whatever its trained prefixes make of it (the build record saw `False` and `True`), and the
fine-tuning question is whether a bounded adaptation of the last decoder blocks teaches the prefix on
held-out papers.

A record is ``{id, source, targets}``: the abstract (sentences joined by a space) without any prefix, and one
or more reference outputs — here the paper title. The prefix is supplied at evaluation and training time.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import re
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .pipeline import MAX_TEXT_CHARS, MODEL_ID, TASK_PREFIXES

CORPUS_NAME = "SciTLDR-A (abstract -> title)"
CORPUS_RELEASE = "allenai/scitldr @ 5ccad9c00a60ad75c9e04abf7f27d0f53f983b20"
CORPUS_BASE_URL = "https://raw.githubusercontent.com/allenai/scitldr/5ccad9c00a60ad75c9e04abf7f27d0f53f983b20/SciTLDR-Data/SciTLDR-A/"
CORPUS_FILES = {
    "train": ("train.jsonl", 3_155_015, "b222771d387be585cfdf5ae957b36757138415a352e0a3e3b23f73f87c3b1119"),
    "dev": ("dev.jsonl", 1_124_865, "3191fa98ccc09521332b7a1cd63b1930be4e8df125a235ccd31e40329709525e"),
    "test": ("test.jsonl", 1_204_107, "fb42dd6cd4f4a1928ae8a01a189456fbfe994a07e938bd49f68653933f6503c9"),
}
CORPUS_LICENSE = "Apache-2.0 (Cachola et al. 2020; allenai/scitldr)"
CORPUS_PAPERS = {"train": 1_992, "dev": 619, "test": 618}
DEFAULT_PREFIX = "paper title: "  # the new task prefix the tutorial teaches; not one of TASK_PREFIXES
DEFAULT_CACHE_DIR = Path("weights") / "scitldr"
MAX_SAMPLE_SOURCE_CHARS = 1_600  # abstracts above this are left out of the sample (the ceiling is 512 tokens)
MIN_SAMPLE_SOURCE_CHARS = 200
SAMPLE_SEED = 42
SAMPLE_SPLIT = {"train": 300, "validation": 50, "test": 100}
MIN_RECORDS = 8
MAX_RECORDS = 20_000
MAX_TARGET_CHARS = 400
_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_corpus(*, cache_dir: str | Path | None = None, fetcher: Any = None) -> dict[str, bytes]:
    """Return the three pinned SciTLDR-A files (bytes) from the cache or the project repository, verified."""
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    for split, (name, size, digest) in CORPUS_FILES.items():
        local = cache / name
        data = local.read_bytes() if local.is_file() else b""
        if len(data) != size or _sha256_bytes(data) != digest:
            url = CORPUS_BASE_URL + name
            if fetcher is not None:
                data = fetcher(url)
            else:
                with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 (pinned https URL)
                    data = response.read()
            if len(data) != size or _sha256_bytes(data) != digest:
                raise ValueError(
                    f"{name}: fetched {len(data)} bytes with sha256 {_sha256_bytes(data)[:16]}…, "
                    f"pinned {size} / {digest[:16]}…"
                )
            local.write_bytes(data)
        out[split] = data
    return out


def read_corpus(files: Mapping[str, bytes]) -> dict[str, list[dict[str, Any]]]:
    """Parse the JSON-Lines members into abstract -> title records keeping each SciTLDR `paper_id`."""
    out = {}
    for split in CORPUS_FILES:
        if split not in files:
            raise ValueError(f"corpus is missing the {split} file")
        records = []
        for line in files[split].decode("utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            title = str(row.get("title", "")).strip()
            records.append(
                {
                    "id": f"{split}-{row['paper_id']}",
                    "source": " ".join(str(s).strip() for s in row["source"]),
                    "targets": [title] if title else [],
                    "paper_id": str(row["paper_id"]),
                }
            )
        if len(records) != CORPUS_PAPERS[split]:
            raise ValueError(f"{split}: {len(records)} papers, expected {CORPUS_PAPERS[split]}")
        out[split] = records
    return out


def filter_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep records whose source is within the sample length window and that have a non-empty title; drop
    repeated sources and repeated titles case-insensitively."""
    seen: set[str] = set()
    seen_titles: set[str] = set()
    kept = []
    for record in records:
        source = str(record["source"])
        if not MIN_SAMPLE_SOURCE_CHARS <= len(source) <= MAX_SAMPLE_SOURCE_CHARS or not record.get("targets"):
            continue
        key, title_key = source.lower(), str(record["targets"][0]).lower()
        if key in seen or title_key in seen_titles:
            continue
        seen.add(key)
        seen_titles.add(title_key)
        kept.append(dict(record))
    return kept


def build_sample_dataset(
    corpus: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    seed: int = SAMPLE_SEED,
    sizes: Mapping[str, int] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Seeded draws from the three SciTLDR members: training from `train`, validation from `dev`, test from
    `test` — the release's own paper-disjoint partition, re-checked on abstracts by `check_split_disjoint`."""
    sizes = dict(sizes or SAMPLE_SPLIT)
    source_of = {"train": "train", "validation": "dev", "test": "test"}
    rng = random.Random(seed)
    out: dict[str, list[dict[str, Any]]] = {}
    for name, size in sizes.items():
        pool = filter_records(corpus[source_of[name]])
        if size > len(pool):
            raise ValueError(f"requested {size} {name} records but only {len(pool)} fit")
        rng.shuffle(pool)
        out[name] = [
            {
                "id": f"{name}-{i:04d}",
                "source": r["source"],
                "targets": list(r["targets"]),
                "paper_id": r["paper_id"],
            }
            for i, r in enumerate(pool[:size])
        ]
    return out


def fetch_sample_dataset(
    *,
    cache_dir: str | Path | None = None,
    fetcher: Any = None,
    seed: int = SAMPLE_SEED,
    sizes: Mapping[str, int] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """The tutorial splits from the pinned corpus."""
    return build_sample_dataset(
        read_corpus(fetch_corpus(cache_dir=cache_dir, fetcher=fetcher)), seed=seed, sizes=sizes
    )


def _check_record(record: Any, index: int) -> dict[str, Any]:
    label = f"records[{index}]"
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} must be a mapping with id/source/targets")
    if "targets" not in record and "target" in record:
        record = {**record, "targets": [record["target"]]}
    for key in ("id", "source", "targets"):
        if key not in record:
            raise ValueError(f"{label} is missing {key!r}")
    rid, source, targets = record["id"], record["source"], record["targets"]
    if not isinstance(rid, str) or not _ID_RE.match(rid):
        raise ValueError(f"{label}: id must match {_ID_RE.pattern}")
    if not isinstance(source, str):
        raise ValueError(f"{label}: source must be a string")
    if not source.strip():
        raise ValueError(f"{label}: source is empty")
    if len(source) > MAX_TEXT_CHARS:
        raise ValueError(
            f"{label}: source has {len(source)} chars; ceiling is MAX_TEXT_CHARS={MAX_TEXT_CHARS}"
        )
    if isinstance(targets, str) or not isinstance(targets, Sequence) or not targets:
        raise ValueError(f"{label}: targets must be a non-empty list of reference outputs")
    checked_targets = []
    for j, target in enumerate(targets):
        if not isinstance(target, str) or not target.strip():
            raise ValueError(f"{label}: targets[{j}] must be a non-empty string")
        if len(target) > MAX_TARGET_CHARS:
            raise ValueError(
                f"{label}: targets[{j}] has {len(target)} chars; "
                f"ceiling is MAX_TARGET_CHARS={MAX_TARGET_CHARS}"
            )
        checked_targets.append(target.strip())
    item = {"id": rid, "source": source.strip(), "targets": checked_targets}
    if "paper_id" in record:
        item["paper_id"] = str(record["paper_id"])
    return item


def validate_dataset(
    records: Sequence[Mapping[str, Any]], *, min_records: int = MIN_RECORDS, max_records: int = MAX_RECORDS
) -> dict[str, Any]:
    """Structural validation of a text-to-text dataset; raises ValueError before any model import."""
    if isinstance(records, Mapping) or not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("records must be a list of {id, source, targets} mappings")
    if not min_records <= len(records) <= max_records:
        raise ValueError(f"{len(records)} records; {min_records}..{max_records} are required")
    checked = []
    ids: set[str] = set()
    sources: set[str] = set()
    for index, record in enumerate(records):
        item = _check_record(record, index)
        if item["id"] in ids:
            raise ValueError(f"duplicate id {item['id']!r}")
        ids.add(item["id"])
        sources.add(item["source"].lower())
        checked.append(item)
    return {
        "records": checked,
        "n_records": len(checked),
        "unique_sources": len(sources),
        "references_per_record": {
            "min": min(len(r["targets"]) for r in checked),
            "max": max(len(r["targets"]) for r in checked),
        },
        "source_chars": {
            "min": min(len(r["source"]) for r in checked),
            "max": max(len(r["source"]) for r in checked),
        },
        "target_words": {
            "min": min(len(r["targets"][0].split()) for r in checked),
            "max": max(len(r["targets"][0].split()) for r in checked),
        },
        "digest": dataset_digest(checked),
        "model_id": MODEL_ID,
    }


def dataset_digest(records: Sequence[Mapping[str, Any]]) -> str:
    payload = [[r["id"], r["source"], list(r["targets"])] for r in records]
    return _sha256_bytes(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def check_split_disjoint(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Assert no lower-cased source document appears in two splits (leakage check)."""
    seen: dict[str, str] = {}
    for name, records in splits.items():
        for record in records:
            key = str(record["source"]).lower()
            if key in seen and seen[key] != name:
                raise ValueError(
                    f"a document ({record['source'][:60]!r}…) appears in both {seen[key]} and {name}"
                )
            seen[key] = name
    return {name: len(records) for name, records in splits.items()}


def _split_sizes(n: int, val_fraction: float, test_fraction: float) -> tuple[int, int, int]:
    """(train, validation, test) that `split_dataset` cuts from `n` distinct sources."""
    n_test = max(1, round(n * test_fraction))
    n_val = round(n * val_fraction)
    return n - n_test - n_val, n_val, n_test


def min_byod_records(*, val_fraction: float = 0.15, test_fraction: float = 0.2) -> dict[str, int]:
    """The smallest dataset `split_dataset` accepts with these fractions: `MIN_RECORDS` training records
    and (when `val_fraction` > 0) at least one validation record, after at least one test record is held
    out. With the default fractions that is 12 distinct sources, split 8 / 2 / 2."""
    for n in range(1, MAX_RECORDS + 1):
        train, val, test = _split_sizes(n, val_fraction, test_fraction)
        if train >= MIN_RECORDS and (val >= 1 or val_fraction == 0):
            return {"total": n, "train": train, "validation": val, "test": test}
    raise ValueError("no dataset size satisfies these fractions")


def split_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """Seeded shuffle of a BYOD dataset into train/validation/test after de-duplicating sources. The
    training split must hold `MIN_RECORDS` records and the validation split (when `val_fraction` > 0) at
    least one; the refusal names the split and the minimum total (`min_byod_records`)."""
    if not (0.0 <= val_fraction < 1.0 and 0.0 < test_fraction < 1.0 and val_fraction + test_fraction < 1.0):
        raise ValueError("fractions must satisfy 0 <= val < 1, 0 < test < 1, val + test < 1")
    checked = validate_dataset(records)["records"]
    seen: set[str] = set()
    unique = []
    for record in checked:
        key = record["source"].lower()
        if key not in seen:
            seen.add(key)
            unique.append(record)
    random.Random(seed).shuffle(unique)
    n_train, n_val, n_test = _split_sizes(len(unique), val_fraction, test_fraction)
    if n_train < MIN_RECORDS or (val_fraction > 0 and n_val < 1):
        need = min_byod_records(val_fraction=val_fraction, test_fraction=test_fraction)
        short = "training split" if n_train < MIN_RECORDS else "validation split"
        raise ValueError(
            f"{len(unique)} distinct sources ({len(checked) - len(unique)} duplicate source(s) removed) "
            "leave the "
            f"{short} with {n_train if n_train < MIN_RECORDS else n_val} record(s); the training split needs "
            f"{MIN_RECORDS} and the validation split 1, so supply at least {need['total']} distinct sources "
            "(split "
            f"{need['train']} / {need['validation']} / {need['test']})"
        )
    return {
        "test": unique[:n_test],
        "validation": unique[n_test : n_test + n_val],
        "train": unique[n_test + n_val :],
    }


def load_byod_dataset(path: str | Path, *, prefix: str | None = None) -> list[dict[str, Any]]:
    """Read records from a CSV (columns id, source, target — one reference per row), a JSON array or JSONL
    of ``{id, source, target}`` or ``{id, source, targets: [...]}`` objects (sources without any prefix).

    Files are read as UTF-8 with or without a byte-order mark (Excel writes one). A source that already starts
    with a trained task prefix (`TASK_PREFIXES`) or with `prefix` is refused — the notebook prepends the
    prefix itself — and every refusal names the line (CSV/JSONL) or the array index (JSON) and the rule."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"dataset not found: {file_path}")
    suffix = file_path.suffix.lower()
    text = file_path.read_text(encoding="utf-8-sig")
    if suffix == ".csv":
        reader = csv.DictReader(io.StringIO(text))
        missing = {"id", "source", "target"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing columns {sorted(missing)} (it needs id, source, target)")
        rows = list(reader)
        if not rows:
            raise ValueError("CSV has a header but no data rows: add one row per input-output pair")
        located = [
            (f"CSV line {i}", {"id": r["id"], "source": r["source"], "targets": [r["target"]]})
            for i, r in enumerate(rows, start=2)
        ]
    elif suffix == ".jsonl":
        located = []
        for i, line in enumerate(text.splitlines(), start=1):
            if line.strip():
                try:
                    located.append((f"JSONL line {i}", _normalise(json.loads(line))))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"JSONL line {i}: not valid JSON ({exc.msg})") from None
    elif suffix == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON dataset is not valid JSON (line {exc.lineno}: {exc.msg})") from None
        if not isinstance(data, list):
            raise ValueError("JSON dataset must be an array of records")
        located = [(f"JSON record [{i}]", _normalise(r)) for i, r in enumerate(data)]
    else:
        raise ValueError("BYOD datasets must be .csv, .json or .jsonl")
    if not located:
        raise ValueError(f"{file_path.name} holds no records")
    known = tuple(p.lower() for p in (*TASK_PREFIXES, *((prefix,) if prefix else ())))
    out = []
    for where, record in located:
        if not isinstance(record, Mapping):
            raise ValueError(f"{where}: each record must be an object with id, source and target(s)")
        source = record.get("source")
        if isinstance(source, str):
            hit = next((p for p in known if source.strip().lower().startswith(p.strip().lower())), None)
            if hit:
                raise ValueError(
                    f"{where} (id {record.get('id')!r}): the source already starts with the task prefix "
                    f"{hit.strip()!r}; "
                    "give sources without any prefix — the notebook prepends PREFIX itself"
                )
        try:
            out.append(_check_record(record, len(out)))
        except ValueError as exc:
            raise ValueError(f"{where}: {str(exc).split(': ', 1)[-1]}") from None
    return out


def _normalise(record: Any) -> Any:
    if isinstance(record, Mapping) and "targets" not in record and "target" in record:
        return {**{k: v for k, v in record.items() if k != "target"}, "targets": [record["target"]]}
    return record


def write_dataset_csv(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """One row per record with its first reference output."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "source", "target"])
        writer.writeheader()
        for record in records:
            writer.writerow({"id": record["id"], "source": record["source"], "target": record["targets"][0]})
    return out
