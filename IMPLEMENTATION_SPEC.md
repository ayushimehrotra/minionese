# Implementation Spec: Mechanistic Anatomy of Multilingual Jailbreaks in LLMs

## Project Overview

This is a NeurIPS paper investigating **why** multilingual jailbreaks succeed at the mechanistic level and proposing **geometry-aware interventions** to fix them. We go beyond prior work (Wang et al., 2025; Zhao et al., 2025) by providing a unified causal pipeline from harm detection failure through refusal gate failure, stratified by language resource tier, perturbation type, and harm category, culminating in three inference-time interventions benchmarked on a safety/utility Pareto frontier.

**Key novelty relative to existing work:**
- Wang et al. (NeurIPS 2025) showed refusal directions are universal but found poor harmful/harmless separation in non-English. We go deeper: we causally decompose *where* and *why* the failure occurs (upstream harm detection vs downstream refusal gating) using attribution patching and SAE feature analysis.
- Zhao et al. (NeurIPS 2025) showed harmfulness and refusal are encoded separately, but only in English. We extend this cross-lingually and show how the harmfulness/refusal disentanglement varies by language tier.
- We introduce a novel 5-perturbation stress test, SAE-based cross-lingual failure feature identification, and three geometry-based interventions with Pareto-optimal evaluation.
- We add **attention head-level causal tracing** (not just layer-level) and **English-pivot hypothesis testing** to isolate whether safety failures route through the model's internal English representation.

---

## Repository Structure

```
multilingual-jailbreak-mech/
├── configs/
│   ├── models.yaml              # Model paths, layer counts, SAE info
│   ├── languages.yaml           # Language tiers, codes, scripts
│   ├── experiment.yaml          # Hyperparameters, sweep ranges
│   └── paths.yaml               # Data/output directory paths
├── dataset/                         # PRE-BUILT -- already exists in repo
│   ├── standard_translation/        # Perturbation type 1
│   │   ├── tier1/
│   │   │   ├── en/
│   │   │   │   ├── harmful.csv
│   │   │   │   └── harmless.csv
│   │   │   ├── de/
│   │   │   │   ├── harmful.csv
│   │   │   │   └── harmless.csv
│   │   │   ├── fr/ ...
│   │   │   ├── zh/ ...
│   │   │   └── es/ ...
│   │   ├── tier2/
│   │   │   ├── ar/ ...
│   │   │   ├── ru/ ...
│   │   │   ├── ko/ ...
│   │   │   └── ja/ ...
│   │   ├── tier3/
│   │   │   ├── tr/ ...
│   │   │   ├── id/ ...
│   │   │   ├── hi/ ...
│   │   │   └── sw/ ...
│   │   └── tier4/
│   │       ├── yo/ ...
│   │       ├── zu/ ...
│   │       ├── gd/ ...
│   │       ├── gn/ ...
│   │       └── jv/ ...
│   ├── translationese/              # Perturbation type 2 (same tier/lang structure)
│   │   ├── tier1/ ...
│   │   ├── tier2/ ...
│   │   ├── tier3/ ...
│   │   └── tier4/ ...
│   ├── transliteration/             # Perturbation type 3 (same tier/lang structure)
│   │   ├── tier1/ ...
│   │   ├── tier2/ ...
│   │   ├── tier3/ ...
│   │   └── tier4/ ...
│   ├── code_switching/              # Perturbation type 4 (same tier/lang structure)
│   │   ├── tier1/ ...
│   │   ├── tier2/ ...
│   │   ├── tier3/ ...
│   │   └── tier4/ ...
│   ├── minionese/                   # Perturbation type 5 (NO tier/lang subfolders)
│   │   ├── harmful.csv
│   │   └── harmless.csv
│   └── *.py / *.ipynb / misc        # Existing scripts used to generate the dataset
├── data/                            # GENERATED -- created by the pipeline
│   ├── splits/                      # Train/val/test splits for probes
│   └── activations/                 # Cached activation tensors
├── src/
│   ├── dataset/
│   │   ├── __init__.py
│   │   └── loader.py            # Unified dataset loading (chat templates)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── generation.py        # Run prompts through target models
│   │   ├── safety_judge.py      # WildGuard + LlamaGuard scoring
│   │   ├── asr.py               # ASR computation per slice
│   │   ├── over_refusal.py      # OR-Bench / XSTest evaluation
│   │   ├── mmlu.py              # Multilingual MMLU capability eval
│   │   └── langid.py            # Output language consistency (fasttext)
│   ├── activations/
│   │   ├── __init__.py
│   │   ├── extract.py           # NNsight-based activation extraction
│   │   ├── cache.py             # Activation caching/loading utilities
│   │   └── positions.py         # Token position selection (t_inst, t_post_inst, last)
│   ├── probing/
│   │   ├── __init__.py
│   │   ├── linear_probe.py      # Train linear probes per harm category
│   │   ├── subspace.py          # Harmfulness subspace construction (W_l)
│   │   ├── effective_rank.py    # Effective rank at 95% energy
│   │   ├── cross_lingual.py     # Principal angles, silhouette scores
│   │   └── disentangle.py       # Harmfulness vs refusal direction separation
│   ├── circuits/
│   │   ├── __init__.py
│   │   ├── attribution_patch.py # Clean/corrupted run patching (layer + component)
│   │   ├── attention_heads.py   # Head-level causal tracing
│   │   └── english_pivot.py     # English-pivot hypothesis test
│   ├── sae/
│   │   ├── __init__.py
│   │   ├── train_sae.py         # Train SAEs for models without public suites (Qwen)
│   │   ├── feature_extract.py   # SAE decomposition at critical layers
│   │   ├── delta_scores.py      # Mean-difference feature scoring
│   │   ├── interpret.py         # Neuronpedia / auto-interp label lookup
│   │   └── clamp.py             # Feature clamping intervention
│   ├── interventions/
│   │   ├── __init__.py
│   │   ├── caa.py               # Contrastive Activation Addition
│   │   ├── sae_clamp.py         # SAE feature clamping at inference
│   │   ├── subspace_project.py  # Subspace projection sharpening (M_tier)
│   │   └── sweep.py             # Alpha/param sweep + Pareto frontier
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── heatmaps.py          # Layer x language heatmaps
│   │   ├── pareto.py            # Safety/utility Pareto plots
│   │   ├── attribution_maps.py  # Attribution patching visualizations
│   │   └── tables.py            # LaTeX table generation
│   └── utils/
│       ├── __init__.py
│       ├── config.py            # YAML config loading
│       ├── logging_setup.py     # Logging configuration
│       ├── gpu.py               # GPU memory management
│       └── reproducibility.py   # Seed setting, deterministic ops
├── scripts/
│   ├── 01_validate_dataset.py
│   ├── 02_run_generation.py
│   ├── 03_evaluate_safety.py
│   ├── 04_extract_activations.py
│   ├── 05_train_probes.py
│   ├── 06_cross_lingual_analysis.py
│   ├── 07_disentangle_harm_refusal.py
│   ├── 08_attribution_patching.py
│   ├── 09_attention_head_tracing.py
│   ├── 10_english_pivot_test.py
│   ├── 11_sae_feature_analysis.py
│   ├── 12_run_interventions.py
│   ├── 13_evaluate_interventions.py
│   └── 14_generate_figures.py
├── tests/
│   ├── test_loader.py
│   ├── test_probes.py
│   ├── test_attribution.py
│   └── test_interventions.py
├── notebooks/
│   └── analysis.ipynb           # Interactive exploration
├── requirements.txt
├── setup.py
└── README.md
```

---

## Environment and Dependencies

```
# requirements.txt
torch>=2.1.0
transformers>=4.40.0
accelerate>=0.27.0
nnsight>=0.3.0
sae-lens>=3.0.0
sparsify>=0.1.0            # EleutherAI SAE training (fallback for Qwen)
scikit-learn>=1.4.0
scipy>=1.12.0
numpy>=1.26.0
pandas>=2.2.0
matplotlib>=3.8.0
seaborn>=0.13.0
plotly>=5.18.0
pyyaml>=6.0
tqdm>=4.66.0
datasets>=2.16.0
huggingface-hub>=0.20.0
fasttext-langdetect>=1.0.5
sentencepiece>=0.2.0
protobuf>=4.25.0
safetensors>=0.4.0
einops>=0.7.0
jaxtyping>=0.2.25
wandb>=0.16.0
pytest>=8.0.0
```

**Hardware requirements:** 1-2x A100 80GB (or equivalent). Activation extraction is memory-intensive; cache to disk aggressively.

---

## Configuration Files

### configs/models.yaml

```yaml
target_models:
  llama:
    name: "meta-llama/Llama-3.1-8B-Instruct"
    num_layers: 32
    hidden_dim: 4096
    num_heads: 32
    sae_suite: "fnlp/Llama-Scope-8B-Base-LXR-32x-TopK"  # LlamaScope, all layers, base model (generalizes to instruct)
    sae_fallback: "EleutherAI/sae-llama-3.1-8b-32x"       # EleutherAI alternative
    sae_source: "saelens"
    sae_pretrained: true
    critical_layer_range: [12, 22]
  gemma:
    name: "google/gemma-2-9b-it"
    num_layers: 42
    hidden_dim: 3584
    num_heads: 16
    sae_suite: "google/gemma-scope-9b-it-res"  # GemmaScope IT, all layers + sublayers
    sae_fallback: "google/gemma-scope-9b-pt-res"  # base model SAEs also work well on IT
    sae_source: "saelens"
    sae_pretrained: true
    critical_layer_range: [15, 30]
  qwen:
    name: "Qwen/Qwen2.5-7B-Instruct"
    num_layers: 28
    hidden_dim: 3584
    num_heads: 28
    sae_suite: null  # NO public SAE suite available -- must train our own
    sae_source: "saelens"  # train using SAELens, then load from disk
    sae_pretrained: false   # triggers SAE training pipeline (see Module 15)
    sae_train_config:
      width_multiplier: 8       # SAE width = 8 * hidden_dim = 28672
      k: 64                     # TopK sparsity
      num_tokens: 500_000_000   # ~500M tokens from RedPajama or similar
      batch_size: 4096
      learning_rate: 3.0e-4
      layers_to_train: "critical"  # only train at critical_layer_range to save compute
    critical_layer_range: [10, 20]

judge_models:
  primary:
    name: "allenai/wildguard"
    type: "classifier"
  secondary:
    name: "meta-llama/LlamaGuard-3-8B"
    type: "classifier"
```

### configs/languages.yaml

```yaml
tiers:
  tier1:
    languages: ["en", "de", "fr", "zh", "es"]
    label: "High-resource"
    script_types:
      en: "latin"
      de: "latin"
      fr: "latin"
      zh: "hanzi"
      es: "latin"
  tier2:
    languages: ["ar", "ru", "ko", "ja"]
    label: "Mid-resource"
    script_types:
      ar: "arabic"
      ru: "cyrillic"
      ko: "hangul"
      ja: "mixed"
  tier3:
    languages: ["tr", "id", "hi", "sw"]
    label: "Low-resource"
    script_types:
      tr: "latin"
      id: "latin"
      hi: "devanagari"
      sw: "latin"
  tier4:
    languages: ["yo", "zu", "gd", "gn", "jv"]
    label: "Very-low-resource"
    script_types:
      yo: "latin"
      zu: "latin"
      gd: "latin"
      gn: "latin"
      jv: "latin"

perturbation_types:
  # These MUST match the folder names in dataset/
  - "standard_translation"   # Direct translation via Google Translate API
  - "translationese"         # EN -> LangX -> EN -> LangX (via open-source model, 3 rounds)
  - "code_switching"         # NER/POS tag harm-relevant nouns, swap to target language
  - "transliteration"        # Non-Latin scripts -> Latin; Latin scripts -> phonetic non-Latin
  - "minionese"              # Constructed gibberish language (flat structure, no tiers/langs)
```

### configs/experiment.yaml

```yaml
dataset:
  dataset_dir: "dataset/"           # Pre-built dataset root (read-only)
  num_harmful: 200
  num_benign: 200
  harm_source: "HarmBench"
  seed: 42
  test_split: 0.2  # hold out 40 harmful + 40 benign for intervention eval

generation:
  temperature: 0.0
  max_new_tokens: 512
  batch_size: 8

activations:
  token_positions: ["last_instruction", "last_post_instruction", "last"]
  # last_instruction = last token of user instruction (t_inst from Zhao et al.)
  # last_post_instruction = last token of full formatted prompt (t_post_inst)
  # last = last token position before generation
  layers: "all"
  dtype: "float32"
  cache_dir: "data/activations/"

probing:
  probe_type: "logistic_regression"
  regularization: "l2"
  C_values: [0.01, 0.1, 1.0, 10.0]
  cv_folds: 5
  subspace_energy_threshold: 0.95  # for effective rank

attribution_patching:
  clean_language: "en"
  components: ["residual", "attn_out", "mlp_out"]
  metric: "refusal_direction_activation"
  batch_size: 16
  # For attention head-level tracing:
  head_level: true

sae:
  top_k_features: 50  # top features by |delta_i|
  clamping_validation_samples: 100

interventions:
  caa:
    alpha_range: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
  sae_clamp:
    top_features: [5, 10, 20, 50]
  subspace_projection:
    train_pairs_per_tier: 50
    regularization: [0.001, 0.01, 0.1]
```

---

## Detailed Module Specifications

---

### Module 1: `src/dataset/loader.py`

**Purpose:** Load the pre-built dataset of harmful/benign prompts across all languages and perturbation types, formatted as chat templates for each target model.

```python
"""
Dataset Loader

The dataset is PRE-BUILT and lives in the `dataset/` directory at the repo root.

Directory layout:
    dataset/
    ├── standard_translation/tier1/en/harmful.csv
    ├── standard_translation/tier1/en/harmless.csv
    ├── standard_translation/tier1/de/harmful.csv
    ├── ...                (same for all tiers and languages)
    ├── translationese/tier1/en/harmful.csv
    ├── ...                (same structure)
    ├── transliteration/...
    ├── code_switching/...
    └── minionese/harmful.csv          # Flat -- no tier/language subfolders
        minionese/harmless.csv

Each CSV has columns (verify and adapt on first run -- column names may vary):
    - prompt: the translated/perturbed prompt text
    - category: HarmBench harm subcategory (e.g. "cybercrime", "violence")
    - en_prompt: the original English prompt this was derived from
    (possibly additional metadata columns)

Harmful and harmless CSVs are contrastive pairs: row i of harmful.csv and
row i of harmless.csv should differ by roughly one token (e.g. "pipe bomb"
vs "pipe long"). THIS HAS NOT BEEN VERIFIED -- the loader must include a
validation step (see validate_contrastive_pairs below).

Key functions:

- load_dataset(
    dataset_dir: str = "dataset/",
    perturbations: List[str] = None,   # e.g. ["standard_translation", "minionese"]
    languages: List[str] = None,       # e.g. ["en", "ar", "yo"]
    tiers: List[str] = None            # e.g. ["tier1", "tier2"]
  ) -> pd.DataFrame
    Walks the directory tree and loads all matching CSVs into a single
    DataFrame with added columns: perturbation, tier, language, is_harmful.
    For minionese (no tier/lang subfolders), set tier="all" and language="minionese".

- validate_contrastive_pairs(df: pd.DataFrame) -> pd.DataFrame:
    For each (perturbation, tier, language) group:
      1. Check that harmful.csv and harmless.csv have the same number of rows
      2. For each row pair, compute token-level edit distance
      3. Flag pairs where edit distance > 3 tokens (expected: ~1 token diff)
      4. Return a report DataFrame with flagged pairs for manual review
    Print a summary: "X of Y pairs validated (Z flagged for review)"

- format_for_model(dataset: pd.DataFrame, model_name: str, tokenizer) -> pd.DataFrame:
    Adds a "formatted_prompt" column by applying the model's chat template
    via tokenizer.apply_chat_template().
    Uses tokenizer.apply_chat_template() with proper system prompts.
    Supports: Llama (<|begin_of_text|>), Gemma (<start_of_turn>), Qwen (<|im_start|>)

- get_contrastive_pairs(
    df: pd.DataFrame, language: str, perturbation: str
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    Returns (harmful_df, harmless_df) with aligned row indices for probing.

- get_split(df: pd.DataFrame, split: str, seed: int = 42, test_ratio: float = 0.2)
    -> pd.DataFrame:
    Deterministic train/test split. Splits by prompt ID so the same
    English source prompt is never in both train and test.

- discover_languages(dataset_dir: str) -> Dict[str, List[str]]:
    Auto-discovers which languages exist under each perturbation/tier
    by walking the directory tree. Returns {perturbation: [lang_codes]}.
"""
```

**Implementation notes:**
- Use `pandas.read_csv()` for loading. Auto-detect encoding (some languages may use UTF-8 BOM).
- The CSV column names may not be perfectly consistent across perturbation folders. On first load, inspect and normalize column names (lowercase, strip whitespace). If the prompt column is named something like `"translated_prompt"` or `"text"` instead of `"prompt"`, map it.
- Chat template formatting MUST use each model's native template (Llama uses `<|begin_of_text|>` format, Gemma uses `<start_of_turn>`, Qwen uses `<|im_start|>`)
- Store tokenized versions separately for activation extraction efficiency
- Run `validate_contrastive_pairs()` once at the start of the pipeline and save the report. Fix or exclude flagged pairs before downstream analysis.
- The `discover_languages()` function protects against missing folders -- if a language was skipped for a perturbation type, it won't silently break.
- There are misc Python scripts and notebooks in the `dataset/` directory that were used to generate the data. These should be left in place but are NOT part of the pipeline. The loader only reads from the CSV files.

---

### Module 2: `src/evaluation/generation.py`

**Purpose:** Run all prompts through target models with greedy decoding and collect responses.

```python
"""
Model Generation Pipeline

Main function:
- generate_responses(
    model_name: str,
    prompts: List[str],
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    batch_size: int = 8,
    output_path: str = None
  ) -> List[dict]

Output format per response:
{
    "prompt_id": "harmful_042_ar_standard",
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "<formatted prompt>",
    "response": "<model output text>",
    "num_tokens_generated": 128,
    "generation_time_s": 1.23
}

Implementation:
- Load model with transformers.AutoModelForCausalLM in float16/bfloat16
- Use accelerate for multi-GPU if available
- Set do_sample=False for temperature=0 (greedy)
- Use tokenizer.pad_token = tokenizer.eos_token if pad_token is None
- Left-pad for batch generation (set padding_side="left")
- Save results incrementally (append to JSONL) to handle crashes
- Log GPU memory usage per batch
"""
```

---

### Module 3: `src/evaluation/safety_judge.py`

**Purpose:** Score model responses for safety using WildGuard (primary) and LlamaGuard (secondary).

```python
"""
Safety Evaluation

Functions:
- score_wildguard(responses: List[dict]) -> List[dict]
    -> Load allenai/wildguard
    -> For each (prompt, response) pair, classify as:
       "safe" / "unsafe" / "refusal"
    -> WildGuard expects a specific input format; follow their template
    -> Return original dict augmented with "wildguard_label" and "wildguard_score"

- score_llamaguard(responses: List[dict]) -> List[dict]
    -> Load meta-llama/LlamaGuard-3-8B
    -> Classify each response
    -> Return augmented with "llamaguard_label" and "llamaguard_category"

- compute_agreement(results: List[dict]) -> dict
    -> Cohen's kappa between WildGuard and LlamaGuard
    -> Per-category agreement breakdown
    -> Log disagreements for manual inspection

Notes:
- WildGuard is the PRIMARY judge. LlamaGuard is secondary/validation.
- Process in batches to manage memory (these are separate model loads).
- Cache judge outputs to avoid recomputation.
"""
```

---

### Module 4: `src/evaluation/asr.py`

**Purpose:** Compute Attack Success Rate across all experimental dimensions.

```python
"""
Attack Success Rate Computation

Main function:
- compute_asr(
    results: List[dict],
    group_by: List[str] = ["language", "perturbation", "model", "harm_category"]
  ) -> pd.DataFrame

ASR = (# responses judged unsafe by WildGuard) / (# total harmful prompts in group)

Output DataFrame columns:
- language, perturbation, model, harm_category, tier
- asr_wildguard (primary)
- asr_llamaguard (secondary)
- n_samples
- 95% Wilson confidence interval bounds

Additional functions:
- asr_summary_table() -> LaTeX-formatted table for paper
- asr_by_tier() -> Aggregated ASR per language tier
- asr_delta_from_english() -> Per-language ASR minus English ASR (the "gap")
"""
```

---

### Module 5: `src/activations/extract.py`

**Purpose:** Extract residual stream activations at every layer for every prompt using NNsight.

```python
"""
Activation Extraction via NNsight

Main function:
- extract_activations(
    model_name: str,
    prompts: List[str],       # Already tokenized/formatted
    token_positions: List[str],  # ["last_instruction", "last_post_instruction", "last"]
    layers: List[int] | str,     # list of layer indices or "all"
    output_dir: str,
    batch_size: int = 4,
    dtype: str = "float32"
  ) -> None  # saves to disk

NNsight usage pattern:
```
from nnsight import LanguageModel

model = LanguageModel(model_name, device_map="auto")

with model.trace(prompt_tokens) as tracer:
    for layer_idx in range(num_layers):
        # Residual stream after layer
        hidden = model.model.layers[layer_idx].output[0]
        # Save at specified token positions
        hidden_at_pos = hidden[:, token_pos, :]
        hidden_at_pos.save()

        # Also extract attention output and MLP output separately
        attn_out = model.model.layers[layer_idx].self_attn.output[0]
        attn_out_at_pos = attn_out[:, token_pos, :]
        attn_out_at_pos.save()

        mlp_out = model.model.layers[layer_idx].mlp.output
        mlp_out_at_pos = mlp_out[:, token_pos, :]
        mlp_out_at_pos.save()
```

Storage format:
- Save as .safetensors files, one per (model, language, perturbation)
- Shape: (n_prompts, n_layers, hidden_dim) per token position
- Also save attention outputs and MLP outputs separately for attribution patching
- File naming: {model_short}_{lang}_{pert}_{position}_{component}.safetensors

Token position resolution:
- "last_instruction": Find the last token of the user message content
  (before any closing template tokens like [/INST] or <end_of_turn>).
  This requires parsing the chat template to find boundaries.
- "last_post_instruction": The very last token of the full formatted
  prompt (including template closing tokens). This is where Zhao et al.
  found refusal is encoded.
- "last": Same as last_post_instruction for single-turn prompts.

CRITICAL: You must determine the exact token index for each position
by tokenizing the formatted prompt and identifying template boundaries.
Write a helper function `find_token_positions(tokenizer, formatted_prompt)`
that returns a dict of position_name -> token_index.

Memory management:
- Process one model at a time
- Use torch.no_grad() throughout
- Delete model from GPU after extraction
- For large prompt sets, process in chunks and save incrementally
"""
```

---

### Module 6: `src/activations/cache.py`

```python
"""
Activation Caching Utilities

Functions:
- save_activations(activations: dict, path: str)
    -> Save as safetensors with metadata

- load_activations(path: str, layers: List[int] = None) -> torch.Tensor
    -> Memory-map if possible; load specific layers on demand

- get_activation_path(model, language, perturbation, position, component) -> str
    -> Deterministic path construction

- activation_exists(model, language, perturbation, position, component) -> bool
    -> Check if cached
"""
```

---

### Module 7: `src/probing/linear_probe.py`

**Purpose:** Train a linear probe per HarmBench harm subcategory vs. harmless baseline for each language.

```python
"""
Linear Probe Training

For each (language, layer, harm_category):
    1. Collect activations for harmful prompts in this category
    2. Collect activations for matched benign prompts
    3. Train logistic regression: harmful vs. benign
    4. Return: probe weights (the "harmfulness direction" for this category),
       accuracy, and AUC

Main function:
- train_probe(
    harmful_activations: np.ndarray,  # (n_harmful, hidden_dim)
    benign_activations: np.ndarray,   # (n_benign, hidden_dim)
    C_values: List[float] = [0.01, 0.1, 1.0, 10.0],
    cv_folds: int = 5
  ) -> dict:
      Returns {
          "weights": np.ndarray,        # (hidden_dim,) - the probe direction
          "bias": float,
          "best_C": float,
          "cv_accuracy": float,
          "cv_auc": float,
          "classification_report": dict
      }

- train_all_probes(
    activations_dir: str,
    dataset: pd.DataFrame,
    languages: List[str],
    layers: List[int],
    harm_categories: List[str],
    output_dir: str
  ) -> pd.DataFrame:  # summary of all probe results

Implementation:
- Use sklearn.linear_model.LogisticRegressionCV
- Normalize activations (zero mean, unit variance) before training
- Save the scaler parameters for later use
- Store probe weights as numpy arrays for subspace construction
"""
```

---

### Module 8: `src/probing/subspace.py`

**Purpose:** Construct the harmfulness subspace W_l per layer from probe weight vectors.

```python
"""
Harmfulness Subspace Construction

For each (language, layer):
    1. Collect probe weight vectors across all harm categories
    2. Stack into matrix W_l of shape (n_categories, hidden_dim)
    3. The harmfulness subspace at layer l = span(W_l)

Main function:
- construct_subspace(
    probe_weights: Dict[str, np.ndarray],  # category -> weight vector
    energy_threshold: float = 0.95
  ) -> dict:
      Returns {
          "W_l": np.ndarray,                # (n_categories, hidden_dim)
          "U": np.ndarray,                  # left singular vectors
          "S": np.ndarray,                  # singular values
          "effective_rank": int,            # rank at energy_threshold
          "energy_spectrum": np.ndarray,    # cumulative energy per component
          "projection_matrix": np.ndarray   # P = U @ U.T for projecting into subspace
      }

- compute_effective_rank(S: np.ndarray, threshold: float = 0.95) -> int:
    cumulative_energy = np.cumsum(S**2) / np.sum(S**2)
    return int(np.searchsorted(cumulative_energy, threshold)) + 1

Notes:
- The effective rank tells us the dimensionality of harm representation
  at each layer. Expect it to vary: early layers may have high rank
  (distributed), critical layers may have low rank (compressed).
- Store full SVD results for downstream use in cross-lingual comparison.
"""
```

---

### Module 9: `src/probing/cross_lingual.py`

**Purpose:** Compare harmfulness subspaces across languages.

```python
"""
Cross-Lingual Subspace Comparison

Two core analyses:

1. Principal Angles between English subspace and each language's subspace:
   - For each layer, compute principal angles between span(W_l^EN) and span(W_l^LangX)
   - Small angles = aligned geometry, large angles = diverged

   Function:
   - compute_principal_angles(U_en: np.ndarray, U_langx: np.ndarray) -> np.ndarray:
       Uses scipy.linalg.subspace_angles(U_en, U_langx)
       Returns array of angles in radians

2. Silhouette Scores of harmful vs. benign within the harmfulness subspace:
   - Project activations into the harmfulness subspace
   - Compute silhouette score (sklearn.metrics.silhouette_score)
   - Produces a (layer x language) heatmap

   Function:
   - compute_silhouette_map(
       activations: dict,       # {(lang, label): np.ndarray}
       subspace_projections: dict,  # {(lang, layer): projection_matrix}
       layers: List[int]
     ) -> pd.DataFrame:  # columns: layer, language, silhouette_score

3. Causal validation of collapse:
   - At the layer where silhouette score collapses for a language,
     zero out the harmfulness subspace component and confirm ASR increases
   - This is done via NNsight intervention (see circuits module)

   Function:
   - validate_collapse(
       model_name: str,
       language: str,
       collapse_layer: int,
       subspace_projection: np.ndarray,
       test_prompts: List[str]
     ) -> dict:  # {asr_before, asr_after, delta}

Output: layer x language heatmap as both DataFrame and matplotlib figure
"""
```

---

### Module 10: `src/probing/disentangle.py`

**Purpose:** Disentangle harmfulness from refusal directions, extending Zhao et al. cross-lingually.

```python
"""
Harmfulness vs. Refusal Disentanglement (Cross-Lingual Extension of Zhao et al. 2025)

This module separates the harmfulness subspace from the refusal direction
to determine whether cross-lingual failure is:
  (a) Upstream: harm detection failing (harmfulness direction not activated)
  (b) Downstream: refusal gate failing (refusal direction not triggered despite harm detection)

Step 1: Extract refusal direction r per model (Arditi et al. method):
   r = mean(activations | refused prompts) - mean(activations | complied prompts)
   Normalize: r_hat = r / ||r||

   For the cross-lingual case, extract r from English refused/complied pairs.

Step 2: Project harmfulness directions onto complement of refusal:
   For each harm category k, language l:
   W'_{k,l} = W_{k,l} - (W_{k,l} . r_hat) * r_hat

   This gives the "pure harmfulness" component orthogonal to refusal.

Step 3: Measure where cross-lingual degradation lives:
   - Compute cosine similarity of each language's W_{k,l} with r_hat
     (refusal-aligned component)
   - Compute norm of W'_{k,l} (harm-detection component)
   - Compare EN vs LangX:
     * If W'_{k,EN} >> W'_{k,LangX}: harm detection is failing upstream
     * If projection onto r is similar but refusal behavior differs:
       refusal gate is failing downstream

Step 4 (NEW - extending Zhao et al. cross-lingually):
   - Extract harmfulness direction at t_inst position (Zhao's finding)
   - Extract refusal direction at t_post_inst position
   - For each language, measure:
     * Harmfulness signal strength at t_inst
     * Refusal signal strength at t_post_inst
     * The "hand-off" from harm detection to refusal gate
   - Identify if low-resource languages have weaker harmfulness encoding
     at t_inst or weaker refusal activation at t_post_inst

Functions:
- extract_refusal_direction(model_name, refused_activations, complied_activations) -> np.ndarray
- project_orthogonal_to_refusal(harm_directions, refusal_direction) -> np.ndarray
- disentangle_analysis(
    harm_subspaces: dict,       # {(lang, layer): W_l}
    refusal_direction: np.ndarray,
    activations_t_inst: dict,    # {(lang, harmful/benign): tensor}
    activations_t_post_inst: dict
  ) -> pd.DataFrame:
    # Columns: language, layer, harm_component_norm, refusal_component_norm,
    #          t_inst_harm_signal, t_post_inst_refusal_signal, failure_type
"""
```

---

### Module 11: `src/circuits/attribution_patch.py`

**Purpose:** Attribution patching to identify which layers mediate cross-lingual refusal failure.

```python
"""
Attribution Patching: Cross-Lingual Refusal Failure Localization

Setup:
- Clean run: English harmful prompt -> model refuses (good behavior)
- Corrupted run: LanguageX harmful prompt (same content) -> model complies (failure)

Patching procedure:
For each layer l:
    1. Run clean (EN) forward pass, cache all intermediate activations
    2. Run corrupted (LangX) forward pass
    3. At layer l, replace the corrupted activation with the clean activation
    4. Continue forward pass from layer l with patched activation
    5. Measure: does refusal direction activation get restored?

Metric: Refusal direction activation = dot(h_patched, r_hat)
where r_hat is the refusal direction from Module 11.

The "restoration score" at layer l = 
    (metric_patched - metric_corrupted) / (metric_clean - metric_corrupted)

A high restoration score at layer l means: the information at layer l
in the English run is sufficient to restore refusal behavior.

Implementation with NNsight:
```python
from nnsight import LanguageModel

model = LanguageModel(model_name, device_map="auto")

# 1. Clean run - cache activations
with model.trace(en_tokens) as tracer:
    clean_activations = {}
    for l in range(num_layers):
        h = model.model.layers[l].output[0]
        clean_activations[l] = h.save()

# 2. Corrupted run with patching at layer l
for patch_layer in range(num_layers):
    with model.trace(langx_tokens) as tracer:
        # At patch_layer, replace with clean activation
        model.model.layers[patch_layer].output[0][:] = clean_activations[patch_layer]
        # Measure output
        final_hidden = model.model.layers[-1].output[0][:, -1, :]
        final_hidden.save()
    
    restoration = compute_refusal_activation(final_hidden, refusal_direction)
    results[patch_layer] = restoration
```

Component-level patching:
- Also patch attention outputs only: replace attn_out at layer l
- Also patch MLP outputs only: replace mlp_out at layer l
- This distinguishes attention-mediated vs MLP-mediated failure

Functions:
- run_attribution_patching(
    model_name: str,
    en_prompts: List[str],
    langx_prompts: List[str],     # matched content, different language
    refusal_direction: np.ndarray,
    components: List[str] = ["residual", "attn_out", "mlp_out"],
    batch_size: int = 16
  ) -> pd.DataFrame:
    # Columns: layer, component, language, tier, mean_restoration, std_restoration

- aggregate_by_tier(results: pd.DataFrame) -> pd.DataFrame:
    # Average restoration scores within each language tier

Output: layer-level attribution map, stratified by language tier and component
"""
```

---

### Module 12: `src/circuits/attention_heads.py`

**Purpose:** Head-level causal tracing to identify specific attention heads responsible for cross-lingual failure.

```python
"""
Attention Head-Level Causal Tracing (NEW - not in prior work)

Extends attribution patching to individual attention heads.
At critical layers identified by Module 12, we patch individual head outputs.

For each (critical_layer, head_idx):
    1. Clean run (EN): cache head output at (layer, head)
    2. Corrupted run (LangX): patch only this head's output
    3. Measure refusal restoration

This identifies "safety-critical heads" that carry cross-lingual information.

Additional analysis:
- For identified critical heads, examine attention patterns:
  * What tokens do they attend to in English vs. LangX?
  * Do they attend to content words (harm-relevant) or template tokens?
  * Is there a pattern where critical heads attend to tokens that
    would be in an "English-like" position?

Implementation:
```python
# Head-level patching via NNsight
for layer in critical_layers:
    for head_idx in range(num_heads):
        with model.trace(langx_tokens) as tracer:
            # Get attention output for this specific head
            # Shape: (batch, seq, num_heads, head_dim)
            attn_output = model.model.layers[layer].self_attn.output[0]
            
            # Replace only head_idx slice
            head_dim = hidden_dim // num_heads
            start = head_idx * head_dim
            end = (head_idx + 1) * head_dim
            attn_output[:, :, start:end] = clean_attn_activations[layer][:, :, start:end]
            
            final = model.model.layers[-1].output[0][:, -1, :]
            final.save()

NOTE: The exact NNsight API for head-level intervention may vary by model.
For models using grouped query attention (GQA), adapt accordingly.
Gemma-2-9B uses GQA (16 KV heads, 16 query heads grouped).
Llama-3.1-8B uses GQA (8 KV heads, 32 query heads).
Handle this by patching the corresponding query head groups.
```

Functions:
- trace_attention_heads(
    model_name: str,
    en_prompts: List[str],
    langx_prompts: List[str],
    critical_layers: List[int],
    refusal_direction: np.ndarray
  ) -> pd.DataFrame:
    # Columns: layer, head_idx, language, restoration_score

- extract_attention_patterns(
    model_name: str,
    prompts: List[str],
    critical_layers: List[int],
    critical_heads: List[Tuple[int, int]]
  ) -> dict:
    # Returns attention weight matrices for analysis
"""
```

---

### Module 13: `src/circuits/english_pivot.py`

**Purpose:** Test whether safety circuitry routes through the model's internal English representation.

```python
"""
English-Pivot Hypothesis Test (NEW)

Recent work (Wendler et al. 2024, Schut et al. 2025) shows LLMs often
use English as an internal pivot language during reasoning. We test whether
safety circuits specifically route through English-language representations.

Approach:
1. Language identification in activation space:
   - Train a linear probe on activations to predict input language
   - Identify layers where language identity is most salient
   - Identify layers where it "collapses" to a shared (English-pivot) space

2. Correlation with safety failure:
   - At each layer, measure (a) language-identity probe confidence and
     (b) harmfulness probe accuracy
   - Test hypothesis: safety probes work well at layers where the
     representation is "English-like" (language identity probe assigns
     high English probability) and fail where it's language-specific

3. Causal test:
   - At a layer where LangX activations are in "LangX-specific" space,
     apply a learned linear map to rotate them toward the English cluster
   - Measure if this restores harmfulness detection

Functions:
- train_language_probe(activations_by_lang: dict, layers: List[int]) -> dict:
    # Train a multi-class classifier: activation -> language
    # Returns per-layer accuracy and confusion matrices

- english_pivot_correlation(
    lang_probe_results: dict,
    harm_probe_results: dict,
    layers: List[int]
  ) -> pd.DataFrame:
    # Correlation between "English-likeness" and safety probe accuracy

- causal_rotation_test(
    model_name: str,
    language: str,
    rotation_matrix: np.ndarray,  # learned from language probe
    test_prompts: List[str],
    target_layer: int
  ) -> dict:
    # Apply rotation and measure safety metric change
"""
```

---

### Module 14: `src/sae/train_sae.py`

**Purpose:** Train sparse autoencoders for models that lack public SAE suites. Currently this applies to Qwen2.5-7B-Instruct. Gemma-2-9B and Llama-3.1-8B have pre-trained SAEs (GemmaScope and LlamaScope/EleutherAI respectively).

```python
"""
SAE Training for Models Without Public Suites

SAE Availability Status:
  - Gemma-2-9B-IT:  COVERED by GemmaScope (google/gemma-scope-9b-it-res, all layers)
  - Llama-3.1-8B:   COVERED by LlamaScope (fnlp/Llama-Scope-8B-Base-LXR-32x-TopK, all layers)
                     Also: EleutherAI/sae-llama-3.1-8b-32x
  - Qwen2.5-7B:     NOT COVERED -- no public comprehensive SAE suite exists.
                     Must train our own.

Training approach for Qwen2.5-7B-Instruct:
  We use SAELens to train TopK SAEs on the residual stream at critical layers
  (identified by attribution patching, typically layers 10-20).

  To save compute, we only train SAEs at the critical layers rather than all 28.
  This is justified because our SAE analysis (Modules 15-18) only operates at
  the layers identified as critical by attribution patching (Module 11).

Training data:
  Use a general-purpose text dataset (RedPajama-v2 sample or FineWeb-Edu).
  For best results on safety-relevant features, mix in a small proportion
  (~10%) of safety-relevant data (HarmBench prompts + safe completions).
  Generate activations by running Qwen2.5-7B-Instruct on this data.

Implementation using SAELens:
```python
from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner

config = LanguageModelSAERunnerConfig(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    hook_name=f"blocks.{layer}.hook_resid_post",
    hook_layer=layer,
    d_in=3584,                    # Qwen2.5-7B hidden dim
    dataset_path="HuggingFaceFW/fineweb-edu",
    streaming=True,
    context_size=1024,
    is_dataset_tokenized=False,

    # SAE architecture
    architecture="topk",
    activation_fn="topk",
    k=64,
    d_sae=3584 * 8,              # 8x expansion = 28672 features

    # Training
    lr=3e-4,
    l1_coefficient=0,             # not needed for TopK
    train_batch_size_tokens=4096,
    training_tokens=500_000_000,  # 500M tokens

    # Logging
    log_to_wandb=True,
    wandb_project="multilingual-jailbreak-sae",
)

runner = SAETrainingRunner(config)
sae = runner.run()
sae.save_model(f"trained_saes/qwen2.5-7b/layer_{layer}")
```

NOTE: If SAELens does not natively support Qwen2.5 hook names, use
NNsight to extract activations to disk first, then train the SAE on
cached activations using SAELens's ActivationsStore with a pre-cached path.
EleutherAI's `sparsify` library is an alternative that supports any
PyTorch model directly: `pip install sparsify` and use `torchrun`.

Functions:
- check_sae_availability(model_name: str) -> dict:
    Returns {"available": bool, "source": str, "hf_repo": str}
    Checks HuggingFace for known SAE repos for this model.

- train_sae_for_layer(
    model_name: str,
    layer: int,
    output_dir: str,
    config_overrides: dict = None
  ) -> str:  # path to saved SAE

- train_all_critical_layers(
    model_name: str,
    critical_layers: List[int],
    output_dir: str
  ) -> Dict[int, str]:  # {layer: path}

Estimated compute:
  ~6-8 hours per layer on 1xA100 for 500M tokens.
  For 10 critical layers = ~60-80 GPU-hours total.
  This is a one-time cost; trained SAEs are cached and reused.
"""
```

**Implementation notes:**
- Check SAELens version compatibility with Qwen2.5. SAELens uses TransformerLens under the hood, which may need a Qwen model wrapper.
- If TransformerLens does not support Qwen2.5, use the NNsight activation caching approach: extract activations to disk with Module 5, then point SAELens at the cached activations.
- As a fallback, EleutherAI's `sparsify` library works with any HuggingFace model directly and uses TopK SAEs by default.
- Save trained SAEs in SAELens-compatible format so downstream modules (feature_extract, clamp) can load them identically to pre-trained ones.
- Run `SAE.load_from_disk()` in downstream modules instead of `SAE.from_pretrained()` for custom-trained SAEs.

---

### Module 15: `src/sae/feature_extract.py`

**Purpose:** Decompose activations into SAE features at critical layers.

```python
"""
SAE Feature Extraction via SAELens

At critical layers identified by attribution patching (Module 12),
decompose residual stream activations into sparse SAE features.

Implementation:
```python
from sae_lens import SAE

# Load the appropriate SAE for the model and layer
sae = SAE.from_pretrained(
    release=sae_suite_name,  # e.g., "google/gemma-scope-9b-it"
    sae_id=f"layer_{layer}/width_16k/average_l0_100",  # adjust per suite
    device="cuda"
)

# Encode activations
# activations shape: (n_prompts, hidden_dim)
feature_activations = sae.encode(activations)  # (n_prompts, sae_width)
reconstructed = sae.decode(feature_activations)
```

For each model, use the correct SAE suite:
- Gemma-2-9B: google/gemma-scope-9b-it-res (GemmaScope IT) or google/gemma-scope-9b-pt-res (base, also works on IT)
- Llama-3.1-8B: fnlp/Llama-Scope-8B-Base-LXR-32x-TopK (LlamaScope, all layers)
  Fallback: EleutherAI/sae-llama-3.1-8b-32x
- Qwen2.5-7B: NO public suite -- load custom-trained SAEs from disk
  (trained by Module 14, stored at trained_saes/qwen2.5-7b/layer_{N}/)
  Use: sae = SAE.load_from_disk(path) instead of SAE.from_pretrained()

Functions:
- load_sae(model_name: str, layer: int) -> SAE
    Checks model config for sae_pretrained flag.
    If True: SAE.from_pretrained(release=sae_suite, sae_id=f"layer_{layer}/...")
    If False: SAE.load_from_disk(f"trained_saes/{model_short}/layer_{layer}")
- encode_activations(sae, activations: torch.Tensor) -> torch.Tensor
- get_top_features(feature_activations: torch.Tensor, k: int) -> List[int]
"""
```

---

### Module 16: `src/sae/delta_scores.py`

**Purpose:** Compute mean-difference scores to identify cross-lingual failure features.

```python
"""
Cross-Lingual SAE Feature Scoring

For each SAE feature i:
    delta_i = mean(z_i | EN_harmful) - mean(z_i | LangX_harmful)

Features with large |delta_i| are "cross-lingual failure features":
they activate strongly for English harmful prompts (enabling safety detection)
but not for the LangX equivalent (causing safety failure).

Functions:
- compute_delta_scores(
    en_features: torch.Tensor,     # (n_en_harmful, sae_width)
    langx_features: torch.Tensor,  # (n_langx_harmful, sae_width)
  ) -> np.ndarray:  # (sae_width,) array of delta scores

- rank_features(delta_scores: np.ndarray, top_k: int = 50) -> List[int]:
    # Return indices of top-k features by |delta|

- feature_analysis_table(
    delta_scores: np.ndarray,
    top_k: int = 50,
    feature_labels: dict = None  # from Neuronpedia
  ) -> pd.DataFrame:
    # Columns: feature_idx, delta_score, |delta|, rank, label
"""
```

---

### Module 17: `src/sae/interpret.py`

**Purpose:** Look up human-interpretable labels for top SAE features.

```python
"""
Feature Interpretation via Neuronpedia / Auto-Interp

For GemmaScope features: query Neuronpedia API for feature labels.
For other suites: use auto-interp if available, or run a simple
auto-interpretation pipeline.

Neuronpedia API (for GemmaScope):
    GET https://www.neuronpedia.org/api/feature/{model_id}/{layer}/{feature_idx}

For features without existing labels:
    1. Find top-activating prompts for the feature
    2. Use an LLM (e.g., Claude or GPT-4) to generate a description
    3. (Optional) Validate with activation patching

Functions:
- lookup_neuronpedia(model_id: str, layer: int, feature_indices: List[int]) -> dict:
    # Returns {feature_idx: label_string}

- auto_interpret_features(
    sae: SAE,
    feature_indices: List[int],
    dataset: List[str],
    top_k_examples: int = 20
  ) -> dict:
    # Find top-activating examples per feature
    # Generate human-readable descriptions

Notes:
- Neuronpedia may not have labels for all features or all models.
- Rate-limit API calls (1 request/second).
- Cache all results locally.
"""
```

---

### Module 18: `src/sae/clamp.py`

**Purpose:** Causally validate failure features by clamping them to English activation levels.

```python
"""
SAE Feature Clamping (Causal Validation)

For each candidate failure feature:
    1. On LangX harmful prompts, intercept the residual stream at the critical layer
    2. Run SAE encoder to get feature activations
    3. Clamp the candidate feature to its English harmful-prompt activation level
    4. Reconstruct via SAE decoder
    5. Continue forward pass
    6. Measure: does refusal get restored?

Implementation with NNsight:
```python
def clamp_and_generate(model, sae, prompt_tokens, feature_idx, clamp_value, layer):
    with model.trace(prompt_tokens) as tracer:
        # Get residual stream at critical layer
        h = model.model.layers[layer].output[0]
        
        # Run through SAE
        z = sae.encode(h[:, -1, :])  # encode last token position
        
        # Clamp specific feature
        z[:, feature_idx] = clamp_value
        
        # Reconstruct
        h_reconstructed = sae.decode(z)
        
        # Replace in residual stream
        model.model.layers[layer].output[0][:, -1, :] = h_reconstructed
        
        # Generate
        output = model.output.save()
    
    return output
```

Functions:
- validate_single_feature(
    model, sae, langx_prompts, feature_idx, en_clamp_value, layer
  ) -> dict:  # {feature_idx, asr_before, asr_after, delta}

- validate_feature_set(
    model, sae, langx_prompts, feature_indices, en_clamp_values, layer
  ) -> pd.DataFrame

- find_validated_features(results: pd.DataFrame, threshold: float = 0.05) -> List[int]:
    # Features whose clamping reduces ASR by at least `threshold`
"""
```

---

### Module 19: `src/interventions/caa.py`

**Purpose:** Contrastive Activation Addition intervention.

```python
"""
Intervention 1: Contrastive Activation Addition (CAA)

Construct a cross-lingual harmfulness steering vector:
    v = mean(h | harmful, EN) - mean(h | harmless, EN)
at the critical layer.

At inference time, for non-English inputs:
    h_intervened = h + alpha * v

Sweep alpha in [0.5, 3.0].

Functions:
- compute_steering_vector(
    en_harmful_activations: torch.Tensor,
    en_harmless_activations: torch.Tensor,
    layer: int
  ) -> torch.Tensor:  # (hidden_dim,)

- apply_caa(
    model_name: str,
    prompts: List[str],
    steering_vector: torch.Tensor,
    alpha: float,
    layer: int,
    max_new_tokens: int = 512
  ) -> List[str]:  # generated responses

- sweep_alpha(
    model_name: str,
    prompts: List[str],
    steering_vector: torch.Tensor,
    alphas: List[float],
    layer: int
  ) -> pd.DataFrame:  # alpha, asr, over_refusal_rate, mmlu_score
"""
```

---

### Module 20: `src/interventions/sae_clamp.py`

**Purpose:** SAE feature clamping intervention at inference time.

```python
"""
Intervention 2: SAE Feature Clamping

At inference time:
    1. Intercept residual stream at critical layer
    2. Run SAE encoder
    3. Clamp validated failure features to English harmful-prompt activation levels
    4. Reconstruct via SAE decoder
    5. Continue forward pass

This tests whether the improvements transfer to languages not used
during feature identification.

Functions:
- apply_sae_clamping(
    model_name: str,
    prompts: List[str],
    sae: SAE,
    feature_indices: List[int],
    clamp_values: Dict[int, float],  # feature_idx -> EN activation level
    layer: int,
    max_new_tokens: int = 512
  ) -> List[str]

- sweep_feature_count(
    model_name: str,
    prompts: List[str],
    sae: SAE,
    ranked_features: List[int],   # ordered by |delta|
    clamp_values: dict,
    layer: int,
    counts: List[int] = [5, 10, 20, 50]
  ) -> pd.DataFrame:  # n_features, asr, over_refusal_rate, mmlu_score
"""
```

---

### Module 21: `src/interventions/subspace_project.py`

**Purpose:** Learn constrained linear maps to align non-English harmfulness subspaces to English.

```python
"""
Intervention 3: Subspace Projection Sharpening

Learn a constrained linear map M_tier that:
- Within the harmfulness subspace: rotate + scale to match English cluster structure
- Outside the harmfulness subspace: identity (no change)

This ensures general capabilities are preserved while safety
representations are aligned.

Training:
- Input: (LangX harmful activations projected into subspace,
          EN harmful activations projected into subspace)
- 50 harmful/harmless pairs per tier
- Least-squares optimization with Frobenius norm regularization

Formulation:
    Let P be the projection onto the harmfulness subspace.
    For LangX activation h:
        h' = (I - P) @ h + M_tier @ (P @ h)
    where M_tier minimizes:
        ||M_tier @ P @ H_langx - P @ H_en||_F^2 + lambda * ||M_tier - I||_F^2

Functions:
- learn_subspace_map(
    langx_activations: np.ndarray,  # (n_train, hidden_dim)
    en_activations: np.ndarray,     # (n_train, hidden_dim)
    projection_matrix: np.ndarray,  # P from subspace construction
    regularization: float = 0.01
  ) -> np.ndarray:  # M_tier of shape (subspace_dim, subspace_dim)

- apply_subspace_projection(
    model_name: str,
    prompts: List[str],
    M_tier: np.ndarray,
    projection_matrix: np.ndarray,
    layer: int,
    max_new_tokens: int = 512
  ) -> List[str]

Notes:
- Train one M per language tier (not per language) to test generalization
- The regularization lambda controls how far M can deviate from identity
- Sweep lambda in [0.001, 0.01, 0.1]
"""
```

---

### Module 22: `src/interventions/sweep.py`

**Purpose:** Orchestrate all intervention sweeps and compute Pareto frontiers.

```python
"""
Intervention Sweep and Pareto Frontier

For each intervention, sweep its hyperparameter(s) and evaluate:
1. ASR reduction (safety improvement)
2. Over-refusal rate on benign prompts (OR-Bench / XSTest)
3. Multilingual MMLU accuracy (capability preservation)
4. Output language consistency via LangID

Plot safety (1 - ASR) vs. utility (MMLU accuracy) Pareto frontier
per language tier, with each intervention as a different color/marker.

Functions:
- run_full_sweep(
    model_name: str,
    interventions: dict,  # {name: {params}}
    test_harmful: List[dict],
    test_benign: List[dict],
    mmlu_data: List[dict]
  ) -> pd.DataFrame:
    # Columns: intervention, param_value, language, tier,
    #          asr, over_refusal, mmlu_accuracy, langid_consistency

- compute_pareto_frontier(results: pd.DataFrame) -> pd.DataFrame:
    # Identify Pareto-optimal points (maximize safety, minimize utility loss)

- plot_pareto(results: pd.DataFrame, output_path: str) -> None:
    # One plot per language tier, all interventions overlaid
"""
```

---

### Module 23: `src/evaluation/over_refusal.py`

```python
"""
Over-Refusal Evaluation

Measure how often the model incorrectly refuses benign prompts.
Uses prompts from OR-Bench and XSTest datasets.

Functions:
- load_or_bench(languages: List[str]) -> List[dict]
    -> Download/load OR-Bench dataset
    -> Translate to target languages if not already available

- load_xstest(languages: List[str]) -> List[dict]
    -> Download/load XSTest dataset

- compute_over_refusal_rate(
    model_name: str,
    benign_prompts: List[str],
    responses: List[str],
    judge: str = "wildguard"
  ) -> float:
    -> Rate = (# responses classified as refusal for benign prompts) / total
"""
```

---

### Module 24: `src/evaluation/mmlu.py`

```python
"""
Multilingual MMLU Evaluation

Measure general capability preservation.

Functions:
- load_multilingual_mmlu(languages: List[str], n_samples: int = 200) -> List[dict]
    -> Use HuggingFace datasets: "openai/MMMLU" or equivalent
    -> Sample n_samples per language for efficiency

- evaluate_mmlu(
    model_name: str,
    mmlu_data: List[dict],
    intervention_fn: Callable = None  # Optional: apply intervention during eval
  ) -> dict:  # {language: accuracy}
"""
```

---

### Module 25: `src/evaluation/langid.py`

```python
"""
Output Language Consistency

Check if interventions cause the model to switch to English responses
even when prompted in another language.

Uses fasttext-langdetect for language identification.

Functions:
- detect_language(text: str) -> Tuple[str, float]:
    from ftlangdetect import detect
    result = detect(text)
    return result["lang"], result["score"]

- compute_langid_consistency(
    prompts: List[dict],   # with "language" field
    responses: List[str]
  ) -> dict:
    # For each target language, what fraction of responses are in that language?
    # Returns {language: consistency_rate}
"""
```

---

### Module 26: `src/visualization/`

```python
"""
Visualization Module

Functions across files:

heatmaps.py:
- plot_silhouette_heatmap(data: pd.DataFrame, output_path: str)
    -> Layer x Language heatmap, color = silhouette score
    -> Annotate with tier boundaries
    -> Use diverging colormap (red = poor separation, blue = good)

- plot_asr_heatmap(data: pd.DataFrame, output_path: str)
    -> Language x Perturbation heatmap, one per model

- plot_effective_rank(data: pd.DataFrame, output_path: str)
    -> Line plot: layer (x) vs effective rank (y), one line per language

pareto.py:
- plot_pareto_frontier(data: pd.DataFrame, output_path: str)
    -> Safety vs Utility scatter, Pareto front highlighted
    -> One subplot per language tier, all interventions overlaid
    -> Include error bars from multiple random seeds

attribution_maps.py:
- plot_attribution_map(data: pd.DataFrame, output_path: str)
    -> Layer (x) vs restoration score (y)
    -> Separate lines for residual, attn_out, mlp_out
    -> One subplot per language tier

- plot_head_level_map(data: pd.DataFrame, output_path: str)
    -> Heatmap: (layer, head) with restoration score

tables.py:
- generate_asr_table(data: pd.DataFrame) -> str:  # LaTeX
- generate_intervention_table(data: pd.DataFrame) -> str:  # LaTeX
- generate_feature_table(data: pd.DataFrame) -> str:  # LaTeX

Style:
- Use matplotlib with a consistent style (plt.style.use('seaborn-v0_8-paper'))
- Font size 10 for NeurIPS formatting
- Save as both PDF and PNG
- Use colorblind-friendly palettes (e.g., seaborn "colorblind")
- Figure size: single column = (3.25, 2.5), double column = (6.75, 3.5)
"""
```

---

## Script Execution Order

Each script in `scripts/` is self-contained and can be run independently (assuming dependencies are met). Run them in numerical order for a full pipeline execution.

### Script 01: Validate Dataset
```bash
python scripts/01_validate_dataset.py --dataset-dir dataset/ --config configs/experiment.yaml
```
**Does NOT regenerate data.** Validates the pre-built dataset:
- Checks that all expected perturbation/tier/language/CSV combinations exist
- Runs `validate_contrastive_pairs()` to verify harmful/harmless rows differ by ~1 token
- Reports row counts per slice, flags missing or malformed files
- Saves a validation report to `results/dataset_validation.json`

### Script 02: Run Generation
```bash
python scripts/02_run_generation.py \
    --model llama \
    --dataset-dir dataset/ \
    --output-dir results/generations/
```
Run once per model. Generates responses for ALL languages x perturbations.

### Script 03: Evaluate Safety
```bash
python scripts/03_evaluate_safety.py \
    --generations-dir results/generations/ \
    --output-dir results/safety_scores/
```
Runs WildGuard and LlamaGuard on all generated responses. Computes ASR.

### Script 04: Extract Activations
```bash
python scripts/04_extract_activations.py \
    --model llama \
    --dataset-dir dataset/ \
    --output-dir data/activations/ \
    --positions last_instruction last_post_instruction last \
    --components residual attn_out mlp_out
```
**Most compute-intensive step.** Run per model. Caches all activations to disk.

### Script 05: Train Probes
```bash
python scripts/05_train_probes.py \
    --activations-dir data/activations/ \
    --output-dir results/probes/
```
Trains linear probes per (language, layer, harm_category). Saves weights.

### Script 06: Cross-Lingual Analysis
```bash
python scripts/06_cross_lingual_analysis.py \
    --probes-dir results/probes/ \
    --activations-dir data/activations/ \
    --output-dir results/cross_lingual/
```
Computes principal angles, silhouette scores, effective rank. Generates heatmaps.

### Script 07: Disentangle Harm/Refusal
```bash
python scripts/07_disentangle_harm_refusal.py \
    --probes-dir results/probes/ \
    --activations-dir data/activations/ \
    --output-dir results/disentangle/
```
Extends Zhao et al. cross-lingually. Separates harm detection vs. refusal gate failure.

### Script 08: Attribution Patching
```bash
python scripts/08_attribution_patching.py \
    --model llama \
    --dataset-dir dataset/ \
    --refusal-dir results/disentangle/ \
    --output-dir results/attribution/
```
Layer-level and component-level patching. Identifies critical layers.

### Script 09: Attention Head Tracing
```bash
python scripts/09_attention_head_tracing.py \
    --model llama \
    --critical-layers results/attribution/critical_layers.json \
    --output-dir results/head_tracing/
```
Head-level causal tracing at critical layers.

### Script 10: English-Pivot Test
```bash
python scripts/10_english_pivot_test.py \
    --activations-dir data/activations/ \
    --output-dir results/english_pivot/
```
Tests whether safety routes through internal English representations.

### Script 11: SAE Feature Analysis
```bash
python scripts/11_sae_feature_analysis.py \
    --model llama \
    --critical-layers results/attribution/critical_layers.json \
    --activations-dir data/activations/ \
    --output-dir results/sae_features/
```
SAE decomposition, delta scoring, interpretation, causal validation.
For Qwen, this script first checks if trained SAEs exist; if not, it calls
the SAE training pipeline (Module 14) automatically before proceeding.

### Script 12: Run Interventions
```bash
python scripts/12_run_interventions.py \
    --model llama \
    --config configs/experiment.yaml \
    --output-dir results/interventions/
```
Applies all three interventions with parameter sweeps.

### Script 13: Evaluate Interventions
```bash
python scripts/13_evaluate_interventions.py \
    --interventions-dir results/interventions/ \
    --output-dir results/intervention_eval/
```
Computes ASR, over-refusal, MMLU, LangID for all intervention results.

### Script 14: Generate Figures
```bash
python scripts/14_generate_figures.py \
    --results-dir results/ \
    --output-dir figures/
```
Generates all paper figures and LaTeX tables.

---

## Key Implementation Gotchas

1. **NNsight API compatibility**: NNsight's API evolves rapidly. Pin the version. The `model.model.layers[l].output[0]` pattern works for Llama-style architectures. For Gemma, the attribute path may differ (`model.layers` vs `model.model.layers`). Test on each model architecture first.

2. **SAE suite availability**: Pre-trained SAE availability varies by model. Gemma-2-9B is fully covered by GemmaScope (`google/gemma-scope-9b-it-res` for IT, `google/gemma-scope-9b-pt-res` for base -- both work on the IT model). Llama-3.1-8B is covered by LlamaScope (`fnlp/Llama-Scope-8B-Base-LXR-32x-TopK`, all layers, base model that generalizes to instruct) and EleutherAI (`sae-llama-3.1-8b-32x`). **Qwen2.5-7B has NO public SAE suite** -- you must train your own using Module 14. Plan for ~60-80 extra GPU-hours for this.

3. **Memory management**: Activation extraction for 200 prompts x 18 languages x 5 perturbations x 32 layers = enormous. Compute one (model, language, perturbation) at a time. Save to disk immediately. Use float16 for storage if float32 is too large.

4. **Chat template formatting**: Each model has a different chat template. Use `tokenizer.apply_chat_template()` and verify outputs manually. The token position for `t_inst` vs `t_post_inst` depends on the template structure.

5. **WildGuard input format**: WildGuard expects a specific prompt format including both the original prompt and the model response. Follow their documentation exactly.

6. **Cross-lingual prompt matching**: Ensure that for attribution patching, the English and LangX prompts encode the *same harmful intent*. Use the contrastive pairs from the dataset -- row i of `dataset/standard_translation/tierN/en/harmful.csv` corresponds to row i of `dataset/standard_translation/tierN/{lang}/harmful.csv`.

7. **Greedy decoding**: With `temperature=0`, set `do_sample=False`. Do NOT set `temperature=0` with `do_sample=True` as this causes division by zero in some implementations.

8. **Reproducibility**: Set seeds everywhere: `torch.manual_seed(42)`, `np.random.seed(42)`, `random.seed(42)`. Use `torch.use_deterministic_algorithms(True)` where possible (may need to disable for some CUDA ops).

9. **Dataset misc files**: The `dataset/` directory contains Python scripts, notebooks, and other files that were used to generate the data. The pipeline should ignore these -- only read from `.csv` files within the expected folder structure. Use glob patterns like `dataset/{perturbation}/{tier}/{lang}/*.csv`.

10. **Minionese special case**: Unlike the other four perturbation types, `dataset/minionese/` has a flat structure with just `harmful.csv` and `harmless.csv` (no tier or language subfolders). The loader must handle this as a special case. In downstream analyses, minionese prompts should be treated as a single "language" with tier="none" or grouped separately. Minionese is primarily included for fun/completeness and should be reported in supplementary material rather than main results.

11. **Contrastive pair integrity**: The harmful/harmless pairs are *supposed* to differ by a single token (e.g., "pipe bomb" vs "pipe long"), but this has not been verified. Script 01 must validate this BEFORE any downstream analysis. Pairs with large edit distance may indicate translation artifacts and should be flagged. Do not silently exclude them -- log them for manual review and report the validation statistics in the paper's appendix.

---

## Expected Outputs and Paper Figures

| Figure | Description | Source Module |
|--------|-------------|---------------|
| Fig 1  | ASR heatmap: language x perturbation, per model | asr.py + heatmaps.py |
| Fig 2  | Silhouette score heatmap: layer x language | cross_lingual.py + heatmaps.py |
| Fig 3  | Effective rank vs layer, per language tier | subspace.py + heatmaps.py |
| Fig 4  | Principal angles: EN vs each language at critical layers | cross_lingual.py |
| Fig 5  | Harmfulness vs refusal signal strength, per language tier | disentangle.py |
| Fig 6  | Attribution patching map: layer x component x tier | attribution_patch.py + attribution_maps.py |
| Fig 7  | Head-level restoration scores at critical layers | attention_heads.py + attribution_maps.py |
| Fig 8  | English-pivot: language probe accuracy vs harm probe accuracy | english_pivot.py |
| Fig 9  | Top SAE failure features with labels | delta_scores.py + interpret.py |
| Fig 10 | Pareto frontier: safety vs utility per intervention per tier | sweep.py + pareto.py |

| Table | Description | Source Module |
|-------|-------------|---------------|
| Tab 1 | ASR per language x model (standard translation only) | asr.py + tables.py |
| Tab 2 | ASR per perturbation type x tier (aggregated over languages) | asr.py + tables.py |
| Tab 3 | Probe accuracy per language at critical layer | linear_probe.py + tables.py |
| Tab 4 | Top 10 SAE failure features with interpretations | delta_scores.py + interpret.py |
| Tab 5 | Intervention comparison: ASR reduction, over-refusal, MMLU, LangID | sweep.py + tables.py |

---

## Testing Strategy

```python
# tests/test_loader.py
- test_dataset_completeness: all perturbation/tier/language folders exist with both CSVs
- test_csv_loadable: every CSV loads without encoding errors and has expected columns
- test_row_counts: harmful.csv and harmless.csv in each folder have equal row counts
- test_contrastive_pairs: harmful/harmless row pairs differ by ~1 token (edit distance check)
- test_minionese_flat: minionese/ has no tier subfolders, just two CSVs
- test_chat_template_formatting: formatted output matches expected patterns per model
- test_discover_languages: auto-discovery finds all expected language codes

# tests/test_probes.py  
- test_probe_training_convergence: probes achieve >60% accuracy on EN
- test_subspace_construction: SVD output shapes are correct
- test_effective_rank: rank <= n_categories

# tests/test_attribution.py
- test_clean_run_refuses: EN harmful prompts get refused
- test_corrupted_run_complies: LangX prompts get complied with (for known-vulnerable languages)
- test_restoration_metric_range: values in [0, 1]

# tests/test_interventions.py
- test_caa_vector_shape: (hidden_dim,)
- test_sae_clamping_reconstructs: MSE between original and reconstructed is small
- test_subspace_map_identity_init: M_tier starts near identity
```

---

## Compute Budget Estimate

| Step | Time (per model, 1xA100) | Storage |
|------|--------------------------|---------|
| Generation (all langs x perts) | ~8 hours | ~2 GB |
| Safety evaluation (WildGuard) | ~4 hours | ~500 MB |
| Activation extraction (all) | ~12 hours | ~50 GB |
| Probe training | ~1 hour | ~1 GB |
| Cross-lingual analysis | ~30 min | ~500 MB |
| Attribution patching | ~6 hours | ~2 GB |
| Head-level tracing | ~4 hours | ~1 GB |
| SAE training (Qwen only) | ~60-80 hours | ~20 GB |
| SAE analysis | ~3 hours | ~5 GB |
| Interventions + sweep | ~8 hours | ~5 GB |
| **Total per model (Llama/Gemma)** | **~47 hours** | **~67 GB** |
| **Total for Qwen (includes SAE training)** | **~107-127 hours** | **~87 GB** |
| **Grand total (3 models)** | **~201-221 hours (~8-9 days)** | **~221 GB** |

Note: SAE training for Qwen is a one-time cost. If you have access to multiple
GPUs, use EleutherAI's `sparsify` with `--distribute_modules` to parallelize
SAE training across layers, reducing wall-clock time to ~8-10 hours on 8xA100.

---

## Notes for Claude Code

- Start by creating the directory structure and all `__init__.py` files.
- Implement and test modules in dependency order: config -> dataset -> evaluation -> activations -> probing -> circuits -> sae -> interventions -> visualization.
- Write unit tests alongside each module.
- Use type hints throughout. Use `jaxtyping` for tensor shape annotations where helpful.
- Log liberally with Python's `logging` module; use structured logging (JSON) for experiment tracking.
- Every script should support `--dry-run` mode that validates inputs without running compute.
- Use `wandb` for experiment tracking (optional but recommended).
- All randomness must be seeded and reproducible.
- **DO NOT run the scripts.** Just write the code. Include the full execution instructions (the Script 01-14 commands from the "Script Execution Order" section above) in a `README.md` at the repo root so the user can run them manually on their GPU cluster.
- The `README.md` should include: project overview, installation instructions (`pip install -r requirements.txt`), dataset description, the full ordered list of script commands to run the pipeline end-to-end, and a note about hardware requirements (1-2x A100 80GB).
