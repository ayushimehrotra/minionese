"""
Visualization: LaTeX Table Generation
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_asr_table(
    data: pd.DataFrame,
    caption: str = "Attack Success Rate by Language and Model",
    label: str = "tab:asr",
) -> str:
    """
    Generate LaTeX table for ASR results.

    Args:
        data: DataFrame with columns: language, model, perturbation, asr_wildguard,
              asr_llamaguard, n_samples.
        caption: Table caption.
        label: LaTeX label.

    Returns:
        LaTeX table string.
    """
    if data.empty:
        return "% No data available for ASR table."

    # Pivot: rows=language, columns=model/perturbation
    group_cols = [c for c in ["language", "perturbation", "model"] if c in data.columns]
    if not group_cols:
        return "% Insufficient columns for ASR table."

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\small",
    ]

    # Determine columns
    display_cols = group_cols + [
        c for c in ["asr_wildguard", "asr_llamaguard", "n_samples"] if c in data.columns
    ]
    col_format = "l" * len(group_cols) + "c" * (len(display_cols) - len(group_cols))
    lines.append(f"\\begin{{tabular}}{{{col_format}}}")
    lines.append(r"\toprule")

    # Header
    headers = [c.replace("_", " ").title() for c in display_cols]
    lines.append(" & ".join(headers) + r" \\")
    lines.append(r"\midrule")

    # Data rows
    for _, row in data[display_cols].iterrows():
        cells = []
        for col in display_cols:
            val = row[col]
            if isinstance(val, float):
                cells.append(f"{val:.3f}")
            else:
                cells.append(str(val))
        lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def generate_intervention_table(
    data: pd.DataFrame,
    caption: str = "Intervention Comparison",
    label: str = "tab:interventions",
) -> str:
    """
    Generate LaTeX table for intervention evaluation results.

    Args:
        data: DataFrame with columns: intervention, param_value, asr,
              over_refusal, mmlu_accuracy, langid_consistency.
        caption: Table caption.
        label: LaTeX label.

    Returns:
        LaTeX table string.
    """
    if data.empty:
        return "% No intervention data available."

    display_cols = [
        c for c in [
            "intervention", "param_value", "asr", "over_refusal",
            "mmlu_accuracy", "langid_consistency"
        ]
        if c in data.columns
    ]

    col_fmt = "l" * 2 + "c" * (len(display_cols) - 2)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\small",
        f"\\begin{{tabular}}{{{col_fmt}}}",
        r"\toprule",
    ]

    headers = {
        "intervention": "Method",
        "param_value": "Param",
        "asr": "ASR $\\downarrow$",
        "over_refusal": "OR $\\downarrow$",
        "mmlu_accuracy": "MMLU $\\uparrow$",
        "langid_consistency": "LangID $\\uparrow$",
    }
    header_row = " & ".join(headers.get(c, c) for c in display_cols) + r" \\"
    lines.append(header_row)
    lines.append(r"\midrule")

    prev_interv = None
    for _, row in data[display_cols].iterrows():
        cells = []
        for col in display_cols:
            val = row.get(col, "")
            if pd.isna(val):
                cells.append("--")
            elif isinstance(val, float):
                cells.append(f"{val:.3f}")
            else:
                cells.append(str(val))
        # Add midrule between intervention groups
        if "intervention" in data.columns:
            current_interv = row.get("intervention", "")
            if prev_interv is not None and current_interv != prev_interv:
                lines.append(r"\midrule")
            prev_interv = current_interv

        lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def generate_feature_table(
    data: pd.DataFrame,
    caption: str = "Top SAE Failure Features",
    label: str = "tab:sae_features",
    max_label_len: int = 50,
) -> str:
    """
    Generate LaTeX table for SAE feature analysis.

    Args:
        data: DataFrame with columns: rank, feature_idx, delta_score, abs_delta, label.
        caption: Table caption.
        label: LaTeX label.
        max_label_len: Truncate labels to this length.

    Returns:
        LaTeX table string.
    """
    if data.empty:
        return "% No SAE feature data available."

    display_cols = [c for c in ["rank", "feature_idx", "delta_score", "abs_delta", "label"] if c in data.columns]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\small",
        r"\begin{tabular}{clccp{4cm}}",
        r"\toprule",
        r"Rank & Feature Idx & $\Delta$ & $|\Delta|$ & Label \\",
        r"\midrule",
    ]

    for _, row in data.head(10).iterrows():
        rank = row.get("rank", "")
        fidx = row.get("feature_idx", "")
        delta = row.get("delta_score", 0.0)
        abs_d = row.get("abs_delta", 0.0)
        label_text = str(row.get("label", ""))
        if len(label_text) > max_label_len:
            label_text = label_text[:max_label_len] + "..."
        # Escape LaTeX special chars
        label_text = label_text.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")

        lines.append(
            f"{rank} & {fidx} & {delta:.3f} & {abs_d:.3f} & {label_text} \\\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)
