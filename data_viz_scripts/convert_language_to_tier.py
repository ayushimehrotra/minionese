#!/usr/bin/env python3
"""
convert_language_to_tier.py

Convert CSVs that use a `language` column into using a `tier` column
according to a fixed mapping. Useful when merging `asr_by_tier` files
that mix "tier" and "language" layouts.

Usage examples
--------------
# Process a single file and write output to `out/` with suffix
python convert_language_to_tier.py --in files/llama/evaluations/asr_by_tier.csv --out-dir out/

# Process all CSVs in a directory (non-destructive)
python convert_language_to_tier.py --dir qwen/evaluations/ --out-dir out/

# Overwrite files in-place (use with caution)
python convert_language_to_tier.py --dir qwen/evaluations/ --inplace
"""

import argparse
from pathlib import Path
import sys
import pandas as pd

# Mapping from language code or lowercase name -> tier
LANG_TO_TIER = {
    # Tier 1
    "en": "tier1", "english": "tier1",
    "es": "tier1", "spanish": "tier1",
    "zh": "tier1", "chinese": "tier1",
    "de": "tier1", "german": "tier1",
    "fr": "tier1", "french": "tier1",

    # Tier 2
    "ar": "tier2", "arabic": "tier2",
    "ru": "tier2", "russian": "tier2",
    "ko": "tier2", "korean": "tier2",
    "ja": "tier2", "japanese": "tier2",

    # Tier 3
    "tr": "tier3", "turkish": "tier3",
    "id": "tier3", "indonesian": "tier3",
    "hi": "tier3", "hindi": "tier3",
    "sw": "tier3", "swahili": "tier3",

    # Tier 4
    "yo": "tier4", "yoruba": "tier4",
    "zu": "tier4", "zulu": "tier4",
    "gd": "tier4", "scottish gaelic": "tier4", "scottish gaelic": "tier4",
    "gn": "tier4", "guaran\u00ed": "tier4", "guaraní": "tier4", "guarani": "tier4",
    "jw": "tier4", "javanese": "tier4",
}


def convert_file(path: Path, out_path: Path, drop_language: bool = True):
    df = pd.read_csv(path)
    # If the file already has a `tier` column (no `language`), keep it
    has_tier = "tier" in df.columns
    has_language = "language" in df.columns
    if (not has_tier) and (not has_language):
        print(f"Skipping {path} — no 'language' or 'tier' column found")
        return

    # If file has `language`, map it to `tier`.
    if has_language:
        lang_series = df["language"].astype(str).str.strip().str.lower()
        mapped = lang_series.map(LANG_TO_TIER)
        unmapped = sorted(set(lang_series[mapped.isna()].tolist()))
        if unmapped:
            print(f"Warning: {path.name} has {len(unmapped)} unmapped language values: {unmapped}")
        df["tier"] = mapped
        if drop_language:
            df = df.drop(columns=["language"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Aggregate ASR values by tier × perturbation × model when possible
    def aggregate_by_tier_perturb(df_in: pd.DataFrame) -> pd.DataFrame:
        # Only aggregate when we have the necessary keys
        if not {"tier", "perturbation", "model"}.issubset(df_in.columns):
            return df_in

        agg_dict = {}
        if "asr_wildguard" in df_in.columns:
            agg_dict["asr_wildguard"] = "mean"
        if "asr_llamaguard" in df_in.columns:
            agg_dict["asr_llamaguard"] = "mean"
        if "n_samples" in df_in.columns:
            agg_dict["n_samples"] = "sum"
        if "ci_lower_95" in df_in.columns:
            agg_dict["ci_lower_95"] = "mean"
        if "ci_upper_95" in df_in.columns:
            agg_dict["ci_upper_95"] = "mean"

        if not agg_dict:
            return df_in

        grouped = (
            df_in
            .groupby(["tier", "perturbation", "model"], sort=False, observed=False)
            .agg(agg_dict)
            .reset_index()
        )

        # reorder columns to common output shape
        out_cols = [c for c in ["tier", "perturbation", "model", "asr_wildguard", "asr_llamaguard", "n_samples", "ci_lower_95", "ci_upper_95"] if c in grouped.columns]
        return grouped[out_cols]

    out_df = aggregate_by_tier_perturb(df)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(out_df)} rows)")


def main():
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--dir", help="Directory containing CSVs to process")
    group.add_argument("--in", dest="infile", help="Single CSV file to process")
    p.add_argument("--out-dir", default="converted_out", help="Where to write converted CSVs")
    p.add_argument("--inplace", action="store_true", help="Overwrite input files (use with care)")
    p.add_argument("--pattern", default="*.csv", help="Filename glob when using --dir")
    p.add_argument("--keep-language", action="store_true", help="Keep the original `language` column instead of dropping it")
    args = p.parse_args()

    out_dir = Path(args.out_dir)

    if args.infile:
        inp = Path(args.infile)
        if not inp.exists():
            print(f"File not found: {inp}")
            sys.exit(1)
        if args.inplace:
            outp = inp
        else:
            outp = out_dir / inp.name
        convert_file(inp, outp, drop_language=not args.keep_language)
        return

    # directory mode
    d = Path(args.dir)
    if not d.exists() or not d.is_dir():
        print(f"Directory not found: {d}")
        sys.exit(1)

    files = sorted(d.glob(args.pattern))
    if not files:
        print(f"No files matching {args.pattern} in {d}")
        return

    for f in files:
        if not f.suffix.lower() == ".csv":
            continue
        if args.inplace:
            outp = f
        else:
            outp = out_dir / f.name
        convert_file(f, outp, drop_language=not args.keep_language)


if __name__ == "__main__":
    main()
