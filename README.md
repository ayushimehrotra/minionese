# Mechanistic Anatomy of Multilingual Jailbreaks in LLMs

This repository contains the code for a NeurIPS paper investigating **why** multilingual jailbreaks succeed at the mechanistic level and proposing **geometry-aware interventions** to fix them.

## Project Overview

We go beyond prior work (Wang et al., 2025; Zhao et al., 2025) by providing a unified causal pipeline from harm detection failure through refusal gate failure, stratified by language resource tier, perturbation type, and harm category, culminating in three inference-time interventions benchmarked on a safety/utility Pareto frontier.

**Key novelties:**
- Causal decomposition of where and why cross-lingual safety failure occurs (upstream harm detection vs downstream refusal gating) using attribution patching and SAE feature analysis
- Cross-lingual extension of Zhao et al.'s harmfulness/refusal disentanglement across language tiers
- Novel 5-perturbation stress test (standard translation, translationese, code switching, transliteration, minionese)
- SAE-based cross-lingual failure feature identification
- Three geometry-based interventions with Pareto-optimal evaluation
- Attention head-level causal tracing and English-pivot hypothesis testing

## Hardware Requirements

- **1-2x A100 80GB** (or equivalent) GPUs
- ~221 GB total storage for all activations and results
- ~8-9 days total GPU time for all three models (Llama, Gemma, Qwen)

## Installation

```bash
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

## Dataset

The dataset is **pre-built** and lives in the `dataset/` directory. It contains harmful/harmless prompt pairs across:

- **4 language tiers**: High-resource (en, de, fr, zh, es), Mid-resource (ar, ru, ko, ja), Low-resource (tr, id, hi, sw), Very-low-resource (yo, zu, gd, gn, jv)
- **5 perturbation types**: standard_translation, translationese, code_switching, transliteration, minionese

Structure:
```
dataset/
├── standard_translation/tier1/en/harmful.csv
├── standard_translation/tier1/en/harmless.csv
├── ...
└── minionese/harmful.csv   # flat structure, no tier/language subfolders
    minionese/harmless.csv
```

## Pipeline Execution

Run scripts in numerical order. Each script is self-contained and supports `--dry-run` for validation.

### Step 1: Validate Dataset

```bash
python scripts/01_validate_dataset.py --dataset-dir dataset/ --config configs/experiment.yaml
```

Validates pre-built dataset structure and contrastive pair integrity. Saves report to `results/dataset_validation.json`.

### Step 2: Run Generation

```bash
python scripts/02_run_generation.py \
    --model llama \
    --dataset-dir dataset/ \
    --output-dir results/generations/
```

Run once per model (`--model llama|gemma|qwen`). Generates responses for all languages x perturbations. Supports resume (skips already-completed prompts).

### Step 3: Evaluate Safety

```bash
python scripts/03_evaluate_safety.py \
    --generations-dir results/generations/ \
    --output-dir results/safety_scores/
```

Runs WildGuard (primary) and LlamaGuard (secondary) on all generated responses. Computes ASR with Wilson confidence intervals.

### Step 4: Extract Activations

```bash
python scripts/04_extract_activations.py \
    --model llama \
    --dataset-dir dataset/ \
    --output-dir data/activations/ \
    --positions last_instruction last_post_instruction last \
    --components residual attn_out mlp_out
```

**Most compute-intensive step** (~12 hours per model on 1xA100). Caches activations to disk as .safetensors files. Supports `--skip-existing` for resume.

### Step 5: Train Probes

```bash
python scripts/05_train_probes.py \
    --activations-dir data/activations/ \
    --output-dir results/probes/
```

Trains linear probes per (language, layer, harm_category). Saves probe weights as .npz files.

### Step 6: Cross-Lingual Analysis

```bash
python scripts/06_cross_lingual_analysis.py \
    --probes-dir results/probes/ \
    --activations-dir data/activations/ \
    --output-dir results/cross_lingual/
```

Computes principal angles between English and other language subspaces, silhouette scores, and effective rank. Generates Figures 2, 3, 4.

### Step 7: Disentangle Harm/Refusal

```bash
python scripts/07_disentangle_harm_refusal.py \
    --probes-dir results/probes/ \
    --activations-dir data/activations/ \
    --output-dir results/disentangle/
```

Cross-lingual extension of Zhao et al. (2025). Separates harm detection from refusal gate failure. Saves refusal direction vector.

### Step 8: Attribution Patching

```bash
python scripts/08_attribution_patching.py \
    --model llama \
    --dataset-dir dataset/ \
    --refusal-dir results/disentangle/ \
    --output-dir results/attribution/
```

Layer-level and component-level patching (~6 hours). Identifies critical layers. Saves `critical_layers.json`. Generates Figure 6.

### Step 9: Attention Head Tracing

```bash
python scripts/09_attention_head_tracing.py \
    --model llama \
    --critical-layers results/attribution/critical_layers.json \
    --output-dir results/head_tracing/
```

Head-level causal tracing at critical layers (~4 hours). Generates Figure 7.

### Step 10: English-Pivot Test

```bash
python scripts/10_english_pivot_test.py \
    --activations-dir data/activations/ \
    --output-dir results/english_pivot/
```

Tests whether safety circuits route through internal English representations. Generates Figure 8.

### Step 11: SAE Feature Analysis

```bash
python scripts/11_sae_feature_analysis.py \
    --model llama \
    --critical-layers results/attribution/critical_layers.json \
    --activations-dir data/activations/ \
    --output-dir results/sae_features/
```

SAE decomposition, delta scoring, Neuronpedia interpretation, and causal validation. Generates Figures 9 and Table 4.

### Step 12: Run Interventions

```bash
python scripts/12_run_interventions.py \
    --model llama \
    --config configs/experiment.yaml \
    --output-dir results/interventions/
```

Applies all three interventions (CAA, SAE clamping, subspace projection) with parameter sweeps (~8 hours).

### Step 13: Evaluate Interventions

```bash
python scripts/13_evaluate_interventions.py \
    --interventions-dir results/interventions/ \
    --output-dir results/intervention_eval/
```

Computes Pareto frontiers and generates Table 5. Generates Figure 10.

### Step 14: Generate Figures

```bash
python scripts/14_generate_figures.py \
    --results-dir results/ \
    --output-dir figures/
```

Generates all paper figures (PDF + PNG) and LaTeX tables.

## Running Tests

```bash
pytest tests/ -v
```

Tests that don't require GPU/model loading run independently. Tests requiring models are integration tests that should be run on GPU hardware.

## Configuration

- `configs/models.yaml`: Model paths, layer counts, SAE availability
- `configs/languages.yaml`: Language tiers, codes, scripts
- `configs/experiment.yaml`: Hyperparameters, sweep ranges
- `configs/paths.yaml`: Data/output directory paths

## Compute Budget

| Step | Time (per model, 1xA100) | Storage |
|------|--------------------------|---------|
| Generation | ~8 hours | ~2 GB |
| Safety eval | ~4 hours | ~500 MB |
| Activation extraction | ~12 hours | ~50 GB |
| Probe training | ~1 hour | ~1 GB |
| Cross-lingual analysis | ~30 min | ~500 MB |
| Attribution patching | ~6 hours | ~2 GB |
| Head-level tracing | ~4 hours | ~1 GB |
| SAE analysis | ~3 hours | ~5 GB |
| Interventions + sweep | ~8 hours | ~5 GB |
| **Total per model** | **~47 hours** | **~67 GB** |
| **Grand total (3 models)** | **~141 hours** | **~201 GB** |

## SAE Availability

| Model | SAE Suite | Notes |
|-------|-----------|-------|
| Llama-3.1-8B-Instruct | LlamaScope (fnlp/Llama-Scope-8B-Base-LXR-32x-TopK) | Pre-trained, all layers |
| Gemma-2-9B-IT | GemmaScope (google/gemma-scope-9b-it-res) | Pre-trained, all layers |
| Qwen2.5-7B-Instruct | andyrdt/saes-qwen2.5-7b-instruct | Pre-trained, all layers |
