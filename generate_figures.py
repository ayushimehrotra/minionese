"""
Generate ASR and coherence heatmaps from all_scored_en.jsonl.
Run after back-translation + WildGuard re-scoring is complete.
"""
import json
import sys
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
sys.path.insert(0, ".")

from src.evaluation.asr import compute_asr, asr_by_tier, asr_delta_from_english
from src.evaluation.coherence import compute_coherence_table
from src.visualization.heatmaps import plot_asr_heatmap, plot_coherence_heatmap

SCORED_PATH = "results/evaluation/all_scored_en.jsonl"
TIER_MAP = {
    "en": "tier1", "de": "tier1", "fr": "tier1", "zh": "tier1", "es": "tier1",
    "ar": "tier2", "ru": "tier2", "ko": "tier2", "ja": "tier2",
    "tr": "tier3", "id": "tier3", "hi": "tier3", "sw": "tier3",
    "yo": "tier4", "zu": "tier4", "gd": "tier4", "gn": "tier4", "jw": "tier4",
    "minionese": "minionese",
}

# ── Load ──────────────────────────────────────────────────────────────────────
logger.info(f"Loading {SCORED_PATH}...")
records = []
with open(SCORED_PATH) as f:
    for line in f:
        rec = json.loads(line)
        if "is_harmful" not in rec:
            rec["is_harmful"] = rec.get("prompt_id", "").startswith("harmful_")
        rec.setdefault("tier", TIER_MAP.get(rec.get("language", ""), "unknown"))
        records.append(rec)

logger.info(f"Loaded {len(records)} records.")

# ── Sanity checks ─────────────────────────────────────────────────────────────
df = pd.DataFrame(records)
n_harmful = df["is_harmful"].sum()
n_coherent = df.get("is_coherent", pd.Series([True]*len(df))).sum() if "is_coherent" in df.columns else len(df)
n_with_en_label = df["wildguard_label_en"].notna().sum() if "wildguard_label_en" in df.columns else 0

logger.info(f"  harmful prompts     : {n_harmful}")
logger.info(f"  coherent responses  : {n_coherent}")
logger.info(f"  wildguard_label_en  : {n_with_en_label}")

if "wildguard_label_en" in df.columns:
    dist = df[df["is_harmful"]]["wildguard_label_en"].value_counts()
    logger.info(f"  WG-EN label dist (harmful): {dist.to_dict()}")

# ── Compute ASR ────────────────────────────────────────────────────────────────
logger.info("Computing ASR...")
asr_df = compute_asr(records, group_by=["language", "perturbation", "model"])
asr_df["tier"] = asr_df["language"].map(TIER_MAP)

asr_df.to_csv("results/evaluation/asr_detailed.csv", index=False)
logger.info(f"ASR table: {len(asr_df)} rows  →  results/evaluation/asr_detailed.csv")

tier_df = asr_by_tier(asr_df)
tier_df.to_csv("results/evaluation/asr_by_tier.csv", index=False)

delta_df = asr_delta_from_english(asr_df)
delta_df.to_csv("results/evaluation/asr_delta_from_english.csv", index=False)

# Quick spot-check: English ASR should be low (model refuses in English)
en_asr = asr_df[asr_df["language"] == "en"]["asr_wildguard"].mean()
hi_asr = asr_df[asr_df["language"] == "hi"]["asr_wildguard"].mean() if "hi" in asr_df["language"].values else None
logger.info(f"  Spot-check — English ASR (expect ~low): {en_asr:.3f}")
if hi_asr is not None:
    logger.info(f"  Spot-check — Hindi  ASR (expect higher): {hi_asr:.3f}")

# ── Compute coherence table ────────────────────────────────────────────────────
logger.info("Computing coherence table...")
coherence_df = compute_coherence_table(records)
coherence_df["tier"] = coherence_df["language"].map(TIER_MAP)
coherence_df.to_csv("results/evaluation/coherence_table.csv", index=False)
logger.info(f"Coherence table: {len(coherence_df)} rows  →  results/evaluation/coherence_table.csv")

# ── Plot ASR heatmap ───────────────────────────────────────────────────────────
logger.info("Plotting ASR heatmap...")
# One heatmap per model
models = sorted(asr_df["model"].unique()) if "model" in asr_df.columns else ["all"]
for model in models:
    model_slug = model.replace("/", "_")
    subset = asr_df[asr_df["model"] == model] if "model" in asr_df.columns else asr_df
    plot_asr_heatmap(
        subset,
        output_path=f"results/figures/asr_heatmap_{model_slug}",
        title=f"Attack Success Rate — {model.split('/')[-1]}",
    )

# ── Plot coherence heatmap ─────────────────────────────────────────────────────
logger.info("Plotting coherence heatmap...")
plot_coherence_heatmap(
    coherence_df,
    output_path="results/figures/coherence_heatmap",
    title="Generation Coherence Rate",
)

logger.info("All figures saved to results/figures/")
