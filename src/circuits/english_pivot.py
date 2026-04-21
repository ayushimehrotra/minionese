"""
English-Pivot Hypothesis Test (NEW)

Test whether safety circuits route through the model's internal
English representation.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def train_language_probe(
    activations_by_lang: Dict[str, np.ndarray],
    layers: List[int],
) -> dict:
    """
    Train a multi-class classifier: activation -> language.

    Args:
        activations_by_lang: Dict mapping lang_code -> (n_samples, n_layers, hidden) array.
                             If (n_samples, hidden) is given, treated as single layer.
        layers: Layer indices to train probes for.

    Returns:
        Dict with keys:
            "per_layer": {layer_idx: {"accuracy": float, "confusion_matrix": np.ndarray}}
            "classifiers": {layer_idx: fitted classifier}
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    languages = sorted(activations_by_lang.keys())
    le = LabelEncoder()
    le.fit(languages)

    per_layer_results = {}
    classifiers = {}

    for layer_idx in layers:
        X_list = []
        y_list = []
        for lang in languages:
            acts = activations_by_lang[lang]
            if acts.ndim == 3:
                # (n_samples, n_layers, hidden)
                if layer_idx >= acts.shape[1]:
                    continue
                layer_acts = acts[:, layer_idx, :]
            else:
                layer_acts = acts

            X_list.append(layer_acts)
            y_list.extend([lang] * len(layer_acts))

        if not X_list:
            continue

        X = np.vstack(X_list)
        y = le.transform(y_list)

        # Train/test split
        n = len(X)
        perm = np.random.RandomState(42).permutation(n)
        train_idx = perm[: int(0.8 * n)]
        test_idx = perm[int(0.8 * n) :]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        clf = LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1)
        clf.fit(X_train, y[train_idx])
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y[test_idx], y_pred)
        cm = confusion_matrix(y[test_idx], y_pred, labels=list(range(len(languages))))

        per_layer_results[layer_idx] = {
            "accuracy": round(acc, 4),
            "confusion_matrix": cm,
            "label_encoder": le,
        }
        classifiers[layer_idx] = clf

        logger.debug(f"Language probe layer {layer_idx}: accuracy={acc:.3f}")

    return {"per_layer": per_layer_results, "classifiers": classifiers}


def english_pivot_correlation(
    lang_probe_results: dict,
    harm_probe_results: dict,
    layers: List[int],
) -> pd.DataFrame:
    """
    Compute correlation between English-likeness and safety probe accuracy.

    Args:
        lang_probe_results: Output from train_language_probe().
        harm_probe_results: Dict mapping layer_idx -> {"cv_accuracy": float, ...}.
                            (per-layer probe results for English harm detection).
        layers: Layer indices.

    Returns:
        DataFrame with columns: layer, lang_probe_accuracy, harm_probe_accuracy,
                                english_likeness, correlation_note.
    """
    rows = []
    for layer in layers:
        lang_acc = lang_probe_results["per_layer"].get(layer, {}).get("accuracy", float("nan"))

        harm_acc = float("nan")
        if isinstance(harm_probe_results, dict):
            entry = harm_probe_results.get(layer, {})
            if isinstance(entry, dict):
                harm_acc = entry.get("cv_accuracy", float("nan"))

        rows.append({
            "layer": layer,
            "lang_probe_accuracy": lang_acc,
            "harm_probe_accuracy": harm_acc,
            # English-likeness: fraction of test samples predicted as English
            # (placeholder -- computed by probe.predict_proba on LangX activations)
            "english_likeness": float("nan"),
        })

    df = pd.DataFrame(rows)
    return df


def causal_rotation_test(
    model_name: str,
    language: str,
    rotation_matrix: np.ndarray,
    test_prompts: List[str],
    target_layer: int,
    refusal_direction: Optional[np.ndarray] = None,
) -> dict:
    """
    Apply a learned rotation to LangX activations at target_layer
    to rotate them toward the English cluster, then measure safety restoration.

    Args:
        model_name: HuggingFace model name.
        language: Source language code.
        rotation_matrix: (hidden_dim, hidden_dim) learned rotation matrix.
        test_prompts: Test prompt strings.
        target_layer: Layer to apply rotation at.
        refusal_direction: Optional refusal direction for metric computation.

    Returns:
        Dict with keys: metric_before, metric_after, delta.
    """
    import torch
    from nnsight import LanguageModel
    from transformers import AutoTokenizer
    from src.utils.gpu import clear_gpu_memory

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    lm = LanguageModel(model_name, device_map="auto", torch_dtype=torch.bfloat16)
    R = torch.tensor(rotation_matrix, dtype=torch.bfloat16).to(lm.device)

    metrics_before = []
    metrics_after = []

    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True).to(lm.device)

        # Baseline
        with torch.no_grad():
            with lm.trace(inputs["input_ids"]) as tracer:
                try:
                    h_out = lm.model.layers[-1].output[0][:, -1, :].save()
                except AttributeError:
                    h_out = lm.model.model.layers[-1].output[0][:, -1, :].save()

        if refusal_direction is not None:
            r = torch.tensor(refusal_direction, dtype=torch.bfloat16).to(lm.device)
            r = r / (r.norm() + 1e-12)
            metrics_before.append(float((h_out.value @ r).mean().item()))

        # With rotation
        with torch.no_grad():
            with lm.trace(inputs["input_ids"]) as tracer:
                try:
                    layer_obj = lm.model.layers[target_layer]
                except AttributeError:
                    layer_obj = lm.model.model.layers[target_layer]

                h = layer_obj.output[0]
                h_rotated = h @ R.T
                layer_obj.output[0][:] = h_rotated

                try:
                    h_out_r = lm.model.layers[-1].output[0][:, -1, :].save()
                except AttributeError:
                    h_out_r = lm.model.model.layers[-1].output[0][:, -1, :].save()

        if refusal_direction is not None:
            metrics_after.append(float((h_out_r.value @ r).mean().item()))

    del lm
    clear_gpu_memory()

    m_before = float(np.mean(metrics_before)) if metrics_before else 0.0
    m_after = float(np.mean(metrics_after)) if metrics_after else 0.0

    return {
        "metric_before": m_before,
        "metric_after": m_after,
        "delta": m_after - m_before,
    }
