"""Regression tests for the Step 2 -> Step 4 WildGuard label handoff."""

import importlib.util
import json
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _install_import_stubs():
    """Keep these smoke tests independent of optional model/ML dependencies."""
    if "huggingface_hub" not in sys.modules:
        hub = types.ModuleType("huggingface_hub")
        hub.login = lambda *args, **kwargs: None
        sys.modules["huggingface_hub"] = hub

    if "sklearn" not in sys.modules:
        sklearn = types.ModuleType("sklearn")
        metrics = types.ModuleType("sklearn.metrics")
        metrics.silhouette_score = lambda *args, **kwargs: 0.0
        sys.modules["sklearn"] = sklearn
        sys.modules["sklearn.metrics"] = metrics


def _load_script(module_name: str, relative_path: str):
    _install_import_stubs()
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_step2_writes_all_scored_with_wildguard_labels(tmp_path, monkeypatch):
    """Step 2 should persist raw WildGuard labels for Step 4 to consume."""
    evaluate = _load_script("script02_evaluate_test", "scripts/02_evaluate.py")

    generations_dir = tmp_path / "generations"
    output_dir = tmp_path / "results_aya" / "evaluation"
    generation_records = [
        {
            "prompt_id": "harmful_0000_en_standard_translation",
            "model": "CohereForAI/aya-expanse-8b",
            "language": "en",
            "perturbation": "standard_translation",
            "tier": "tier1",
            "category": "test_harm",
            "is_harmful": True,
            "prompt": "harmful prompt 0",
            "response": "I cannot help with that request, but I can offer safe alternatives.",
        },
        {
            "prompt_id": "harmful_0001_en_standard_translation",
            "model": "CohereForAI/aya-expanse-8b",
            "language": "en",
            "perturbation": "standard_translation",
            "tier": "tier1",
            "category": "test_harm",
            "is_harmful": True,
            "prompt": "harmful prompt 1",
            "response": "Here is harmful content for the synthetic test.",
        },
        {
            "prompt_id": "harmless_0000_en_standard_translation",
            "model": "CohereForAI/aya-expanse-8b",
            "language": "en",
            "perturbation": "standard_translation",
            "tier": "tier1",
            "category": "benign",
            "is_harmful": False,
            "prompt": "benign prompt 0",
            "response": "Here is a normal helpful answer with enough text to be coherent.",
        },
    ]
    _write_jsonl(generations_dir / "aya_en_standard_translation.jsonl", generation_records)

    def fake_score_wildguard(responses, **_kwargs):
        labels = {
            "harmful_0000_en_standard_translation": "refusal",
            "harmful_0001_en_standard_translation": "unsafe",
            "harmless_0000_en_standard_translation": "safe",
        }
        scored = []
        for rec in responses:
            augmented = dict(rec)
            augmented["wildguard_label"] = labels[rec["prompt_id"]]
            augmented["wildguard_score"] = f"fake:{augmented['wildguard_label']}"
            scored.append(augmented)
        return scored

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(evaluate, "ensure_hf_token", lambda _args, _logger: "hf_test")
    monkeypatch.setattr(evaluate, "score_wildguard", fake_score_wildguard)
    monkeypatch.setattr(evaluate, "load_config", lambda: {"experiment": {"coherence": {}}})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "02_evaluate.py",
            "--generations-dir",
            str(generations_dir),
            "--output-dir",
            str(output_dir),
            "--batch-size",
            "2",
        ],
    )

    evaluate.main()

    all_scored = output_dir / "all_scored.jsonl"
    assert all_scored.exists()
    rows = _read_jsonl(all_scored)
    assert len(rows) == len(generation_records)
    assert all("wildguard_label" in row for row in rows)
    assert {
        row["prompt_id"]: row["wildguard_label"]
        for row in rows
    } == {
        "harmful_0000_en_standard_translation": "refusal",
        "harmful_0001_en_standard_translation": "unsafe",
        "harmless_0000_en_standard_translation": "safe",
    }


def test_step4_loads_step2_wildguard_labels_and_blocks_naive_fallback(tmp_path):
    """Step 4 should find Step 2 labels and avoid silent naive splitting."""
    representation = _load_script(
        "script04_representation_analysis_test",
        "scripts/04_representation_analysis.py",
    )

    scored_path = tmp_path / "results_aya" / "evaluation" / "all_scored.jsonl"
    _write_jsonl(
        scored_path,
        [
            {
                "prompt_id": "harmful_0000_en_standard_translation",
                "model": "CohereForAI/aya-expanse-8b",
                "language": "en",
                "perturbation": "standard_translation",
                "is_harmful": True,
                "wildguard_label": "refusal",
            },
            {
                "prompt_id": "harmful_0001_en_standard_translation",
                "model": "CohereForAI/aya-expanse-8b",
                "language": "en",
                "perturbation": "standard_translation",
                "is_harmful": True,
                "wildguard_label": "unsafe",
            },
            {
                "prompt_id": "harmful_0002_fr_standard_translation",
                "model": "CohereForAI/aya-expanse-8b",
                "language": "fr",
                "perturbation": "standard_translation",
                "is_harmful": True,
                "wildguard_label": "unsafe",
            },
            {
                "prompt_id": "harmless_0000_en_standard_translation",
                "model": "CohereForAI/aya-expanse-8b",
                "language": "en",
                "perturbation": "standard_translation",
                "is_harmful": False,
                "wildguard_label": "safe",
            },
        ],
    )

    args = SimpleNamespace(
        model="aya",
        output_dir=str(tmp_path / "results_aya" / "representation"),
        scored_path=None,
        perturbation="standard_translation",
        allow_naive_refusal_split=False,
    )
    labels = representation._load_en_wildguard_labels(
        args,
        logging.getLogger("test_step4_wildguard"),
    )

    assert labels == {
        "harmful_0000_en_standard_translation": "refusal",
        "harmful_0001_en_standard_translation": "unsafe",
    }

    with pytest.raises(RuntimeError, match="refusing to use the naive"):
        representation._fallback_or_raise_naive_split(
            np.zeros((10, 4), dtype=np.float32),
            args,
            logging.getLogger("test_step4_wildguard"),
            "No WildGuard labels found.",
        )
