"""
Linear Probe Training

Train logistic regression probes to detect harmfulness at each layer.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def train_probe(
    harmful_activations: np.ndarray,
    benign_activations: np.ndarray,
    C_values: List[float] = None,
    cv_folds: int = 5,
) -> dict:
    """
    Train a logistic regression probe to classify harmful vs. benign.

    Args:
        harmful_activations: (n_harmful, hidden_dim) array.
        benign_activations: (n_benign, hidden_dim) array.
        C_values: Regularization strengths to try. Default: [0.01, 0.1, 1.0, 10.0].
        cv_folds: Number of cross-validation folds.

    Returns:
        Dict with keys: weights, bias, best_C, cv_accuracy, cv_auc,
                        classification_report, scaler_mean, scaler_std.
    """
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.metrics import classification_report, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    if C_values is None:
        C_values = [0.01, 0.1, 1.0, 10.0]

    X = np.vstack([harmful_activations, benign_activations])
    y = np.array([1] * len(harmful_activations) + [0] * len(benign_activations))

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegressionCV(
        Cs=C_values,
        cv=cv_folds,
        max_iter=1000,
        scoring="roc_auc",
        n_jobs=-1,
        l1_ratios=(0.0,),
        use_legacy_attributes=False,
    )
    clf.fit(X_scaled, y)

    y_pred = clf.predict(X_scaled)
    y_prob = clf.predict_proba(X_scaled)[:, 1]

    try:
        auc = roc_auc_score(y, y_prob)
    except Exception:
        auc = 0.0

    accuracy = (y_pred == y).mean()

    return {
        "weights": clf.coef_[0].copy(),
        "bias": float(clf.intercept_[0]),
        "best_C": float(np.asarray(clf.C_).reshape(-1)[0]),
        "cv_accuracy": float(accuracy),
        "cv_auc": float(auc),
        "classification_report": classification_report(y, y_pred, output_dict=True),
        "scaler_mean": scaler.mean_.copy(),
        "scaler_std": scaler.scale_.copy(),
    }


def train_all_probes(
    activations_dir: str,
    dataset: pd.DataFrame,
    languages: List[str],
    layers: List[int],
    harm_categories: List[str],
    output_dir: str,
    model_short: str = "llama",
    perturbation: str = "standard_translation",
    token_position: str = "last_post_instruction",
) -> pd.DataFrame:
    """
    Train probes for all (language, layer, harm_category) combinations.

    Args:
        activations_dir: Directory with cached activation files.
        dataset: Dataset DataFrame with is_harmful, language, category columns.
        languages: Language codes to train for.
        layers: Layer indices.
        harm_categories: Harm category names to train separate probes for.
        output_dir: Directory to save probe results.
        model_short: Short model name for file lookup.
        perturbation: Perturbation type to use.
        token_position: Token position name.

    Returns:
        Summary DataFrame with all probe results.
    """
    from src.activations.cache import load_activations, get_activation_path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for lang in languages:
        # Load residual stream activations for this language
        act_path = get_activation_path(
            model=model_short,
            language=lang,
            perturbation=perturbation,
            position=token_position,
            component="residual",
            cache_dir=activations_dir,
        )
        if not Path(act_path).exists():
            logger.warning(f"No activations found: {act_path}. Skipping {lang}.")
            continue

        try:
            all_activations = load_activations(act_path)  # (n_prompts, n_layers, hidden)
        except Exception as e:
            logger.error(f"Error loading activations {act_path}: {e}")
            continue

        lang_mask = dataset["language"] == lang
        lang_df = dataset[lang_mask].reset_index(drop=True)

        if len(lang_df) == 0:
            logger.warning(f"No dataset rows for language '{lang}'.")
            continue

        for layer_idx in layers:
            if layer_idx >= all_activations.shape[1]:
                continue
            layer_acts = all_activations[:, layer_idx, :].numpy()  # (n_prompts, hidden)

            for category in harm_categories + ["all"]:
                if category == "all":
                    harm_mask = lang_df["is_harmful"].values
                    benign_mask = ~lang_df["is_harmful"].values
                else:
                    harm_mask = lang_df["is_harmful"] & (lang_df.get("category", "") == category)
                    benign_mask = (~lang_df["is_harmful"]).values

                harm_indices = np.where(harm_mask)[0]
                benign_indices = np.where(benign_mask)[0]

                if len(harm_indices) < 5 or len(benign_indices) < 5:
                    continue

                # Map dataset row indices to activation array indices
                # Assumes dataset rows are ordered to match activation rows
                n = min(len(harm_indices), len(benign_indices))
                harm_acts = layer_acts[harm_indices[:n]]
                benign_acts = layer_acts[benign_indices[:n]]

                try:
                    result = train_probe(harm_acts, benign_acts)
                except Exception as e:
                    logger.error(f"Probe training failed ({lang}, layer={layer_idx}, cat={category}): {e}")
                    continue

                probe_data = {
                    "language": lang,
                    "layer": layer_idx,
                    "category": category,
                    "cv_accuracy": result["cv_accuracy"],
                    "cv_auc": result["cv_auc"],
                    "best_C": result["best_C"],
                    "n_harmful": len(harm_indices),
                    "n_benign": len(benign_indices),
                }
                summary_rows.append(probe_data)

                # Save probe weights
                probe_path = output_dir / f"probe_{lang}_layer{layer_idx}_{category}.npz"
                np.savez(
                    str(probe_path),
                    weights=result["weights"],
                    bias=np.array([result["bias"]]),
                    scaler_mean=result["scaler_mean"],
                    scaler_std=result["scaler_std"],
                )

        logger.info(f"Probes trained for language: {lang}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "probe_summary.csv"
    summary_df.to_csv(str(summary_path), index=False)
    logger.info(f"Probe summary saved to {summary_path}.")
    return summary_df


def load_probe(probe_path: str) -> dict:
    """Load a saved probe from a .npz file."""
    data = np.load(probe_path)
    return {
        "weights": data["weights"],
        "bias": float(data["bias"][0]),
        "scaler_mean": data["scaler_mean"],
        "scaler_std": data["scaler_std"],
    }
