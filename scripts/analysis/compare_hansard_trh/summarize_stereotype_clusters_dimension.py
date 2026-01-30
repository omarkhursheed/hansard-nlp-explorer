#!/usr/bin/env python3
"""
Summarize stereotype clusters using OpenRouter (or compatible OpenAI-style endpoint).

Takes the clustered output from compare_stereotypes.py and, for each cluster_id,
asks an LLM to provide:
  - a short stereotype label, and
  - a 1–2 sentence summary of the bias.

This helps interpret which shared stereotypes appear across Hansard and TRH,
and how frequently they occur in each dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests


DEFAULT_API_URL = "https://openrouter.ai/api/v1/chat/completions"


def get_api_key() -> str:
    """Load the OpenRouter API key, mirroring extract_stereotypes_hansard.py."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("Set OPENROUTER_API_KEY before running this script.")
    return api_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize stereotype clusters via OpenRouter.")
    parser.add_argument(
        "--datasets",
        choices=["hansard", "trh", "both"],
        default="both",
        help="Which dataset clusters to label.",
    )
    parser.add_argument(
        "--hansard-clusters",
        type=Path,
        help="Hansard cluster summary Parquet (from compare_stereotypes.py).",
    )
    parser.add_argument(
        "--hansard-flat",
        type=Path,
        help="Hansard flat Parquet with cluster membership (from compare_stereotypes.py).",
    )
    parser.add_argument(
        "--hansard-output",
        type=Path,
        help="Output Parquet/CSV with Hansard cluster labels.",
    )
    parser.add_argument(
        "--trh-clusters",
        type=Path,
        help="TRH cluster summary Parquet (from compare_stereotypes.py).",
    )
    parser.add_argument(
        "--trh-flat",
        type=Path,
        help="TRH flat Parquet with cluster membership (from compare_stereotypes.py).",
    )
    parser.add_argument(
        "--trh-output",
        type=Path,
        help="Output Parquet/CSV with TRH cluster labels.",
    )
    parser.add_argument(
        "--hansard-prompt-file",
        type=Path,
        help="Prompt file to use when labelling Hansard clusters (required if Hansard is selected).",
    )
    parser.add_argument(
        "--trh-prompt-file",
        type=Path,
        help="Prompt file to use when labelling TRH clusters (required if TRH is selected).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemini-2.5-flash",
        help="OpenRouter model name.",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=DEFAULT_API_URL,
        help="OpenRouter chat completions endpoint.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=300,
        help="Max tokens per response.",
    )
    parser.add_argument(
        "--rate-limit-sleep",
        type=float,
        default=1.0,
        help="Seconds to sleep between API calls.",
    )
    parser.add_argument(
        "--http-referer",
        type=str,
        default="https://example.com",
        help="HTTP-Referer header for OpenRouter.",
    )
    parser.add_argument(
        "--x-title",
        type=str,
        default="Stereotype Cluster Summaries",
        help="X-Title header for OpenRouter.",
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=None,
        help="Optional limit on number of clusters to summarize (for testing).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=2,
        help="Number of example sentences per dataset to include in each cluster prompt.",
    )
    return parser.parse_args()


def load_prompt_sections(path: Path) -> tuple[str, str]:
    """Load SYSTEM and USER prompts from a markdown file."""
    sections: Dict[str, List[str]] = {}
    current: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        header = line.strip()
        if header in {"SYSTEM", "USER"}:
            current = header
            sections[current] = []
            continue
        if current:
            sections[current].append(line.rstrip())

    system_prompt = "\n".join(sections.get("SYSTEM", [])).strip()
    user_prompt = "\n".join(sections.get("USER", [])).strip()
    if not system_prompt or not user_prompt:
        raise ValueError(f"Prompt file {path} must contain SYSTEM and USER sections.")
    return system_prompt, user_prompt


def build_samples_from_flat(df_flat: pd.DataFrame, max_per_dataset: int = 5) -> pd.DataFrame:
    """Pick top stereotype_text per cluster and dataset by count."""
    required = {"cluster_id", "dataset", "stereotype_text"}
    missing = required - set(df_flat.columns)
    if missing:
        raise ValueError(f"Flat file is missing columns: {missing}")

    df_flat = df_flat.copy()
    if "count" not in df_flat.columns:
        df_flat["count"] = 1

    grouped = (
        df_flat.groupby(["cluster_id", "dataset", "stereotype_text"], as_index=False)["count"]
        .sum()
        .rename(columns={"count": "weight"})
    )

    sample_rows = []
    for cluster_id, cluster_grp in grouped.groupby("cluster_id"):
        samples: List[str] = []
        for _, ds_grp in cluster_grp.groupby("dataset"):
            top = ds_grp.sort_values("weight", ascending=False).head(max_per_dataset)
            samples.extend(top["stereotype_text"].tolist())
        sample_rows.append({"cluster_id": cluster_id, "sample_stereotypes": samples})

    return pd.DataFrame(sample_rows)


def _ensure_sentence_list(value) -> List[str]:
    """Best-effort conversion of the sample_stereotypes cell into a list of strings."""
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        text = value.strip()
        # Try to interpret as JSON list first
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except Exception:
            pass
        return [text]
    return [str(value)]


def load_clusters(
    path: Path,
    max_clusters: int | None,
    df_flat: pd.DataFrame,
    samples_per_dataset: int,
) -> pd.DataFrame:
    """
    Load cluster metadata and enrich it with:
      - example stereotype sentences per cluster (from the flat file), and
      - per-dataset counts per cluster, computed from the flat file's `count` column,
      - a coarse bucket / dimension per cluster.
    """
    # We still read the clusters parquet mainly to make sure cluster_ids exist
    df = pd.read_parquet(path)
    required = {"cluster_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Clusters file is missing columns: {missing}")

    # Build example sentences per cluster from the flat membership file:
    # pick top stereotypes by count for each dataset (tunable via --n-samples).
    sample_per_cluster = build_samples_from_flat(df_flat, max_per_dataset=samples_per_dataset)

    # Derive a single dimension per cluster from the flat file.
    # We assume clustering was performed within dimension buckets, so each cluster
    # should have a consistent `dimension`. As a safeguard, we take the mode.
    if "dimension" in df_flat.columns:
        dim_mode = (
            df_flat.groupby("cluster_id")["dimension"]
            .agg(lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0])
            .reset_index()
            .rename(columns={"dimension": "dimension"})
        )
    else:
        # If dimension is unavailable, we fall back to a missing value.
        dim_mode = pd.DataFrame(
            {"cluster_id": df_flat["cluster_id"].unique(), "dimension": None}
        )

    # --- derive per-dataset counts from the flat file --------------------
    df_counts = df_flat.copy()
    if "count" not in df_counts.columns:
        # If no explicit count, treat each row as a single occurrence
        df_counts["count"] = 1

    # We expect a 'dataset' column in the flat file (e.g., 'hansard', 'trh')
    if "dataset" not in df_counts.columns:
        raise ValueError("Flat file is missing required 'dataset' column for counting.")

    # Long -> wide: counts per (cluster_id, dataset)
    counts_long = (
        df_counts
        .groupby(["cluster_id", "dataset"], as_index=False)["count"]
        .sum()
    )

    counts_wide = (
        counts_long
        .pivot(index="cluster_id", columns="dataset", values="count")
        .reset_index()
    )

    # Fill missing combinations with 0 and rename dataset columns to count_<dataset>
    counts_wide = counts_wide.fillna(0)
    rename_map = {}
    for col in counts_wide.columns:
        if col != "cluster_id":
            rename_map[col] = f"count_{col}"
    counts_wide = counts_wide.rename(columns=rename_map)

    # -------------------------------------------------------------------------
    # Merge examples, dimensions, and counts; this is what the rest of the script expects.
    merged = (
        sample_per_cluster
        .merge(dim_mode, on="cluster_id", how="left")
        .merge(counts_wide, on="cluster_id", how="left")
    )
    merged = merged.sort_values("cluster_id").reset_index(drop=True)

    if max_clusters is not None:
        merged = merged.head(max_clusters).copy()

    return merged



def build_prompt(
    cluster_id: int,
    dimension: str | None,
    sentences: List[str],
    counts: Dict[str, int],
    user_template: str,
) -> str:
    """Fill the DIMENSION and TARGET_EXAMPLES slots in the USER template for this cluster."""
    examples_block = "\n".join(f"- \"{s}\"" for s in sentences)
    counts_str = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    target_block = f"""Cluster sentences:
{examples_block}

Cluster counts by dataset: {counts_str}"""

    prompt = user_template

    # Inject the coarse bucket / dimension if requested by the template.
    if "{{DIMENSION}}" in prompt:
        dim_value = dimension if dimension is not None else "other"
        prompt = prompt.replace("{{DIMENSION}}", str(dim_value))

    # Inject the example sentences block.
    if "{{TARGET_EXAMPLES}}" in prompt:
        prompt = prompt.replace("{{TARGET_EXAMPLES}}", target_block)
    else:
        # Fallback: append if placeholder is missing
        prompt = prompt + "\n\n" + target_block

    return prompt



def clean_json_string(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        raw = raw.lstrip("json").strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1:
        return raw[start : end + 1]
    return raw


def call_openrouter(
    system_prompt: str,
    user_prompt: str,
    api_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    http_referer: str,
    x_title: str,
) -> Dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": http_referer,
        "X-Title": x_title,
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    response = requests.post(api_url, headers=headers, json=payload, timeout=120)
    if response.status_code != 200:
        try:
            err = response.json()
        except Exception:
            err = response.text
        raise requests.HTTPError(f"OpenRouter error {response.status_code}: {err}")
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content)

    try:
        cleaned = clean_json_string(content)
        result = json.loads(cleaned)
        if not isinstance(result, dict) or "label" not in result or "summary" not in result:
            raise ValueError("JSON missing expected keys")
        # Confidence is optional but encouraged by the prompt.
        conf_value = result.get("confidence")
        try:
            confidence = float(conf_value) if conf_value is not None else None
        except (TypeError, ValueError):
            confidence = None
        return {
            "label": str(result.get("label", "")).strip() or "parse_error",
            "summary": str(result.get("summary", "")).strip(),
            "confidence": confidence,
        }
    except Exception:
        return {
            "label": "parse_error",
            "summary": clean_json_string(content).strip()[:500],
            "confidence": None,
        }


def process_dataset(
    dataset_label: str,
    clusters_path: Path,
    flat_path: Path,
    output_path: Path,
    max_clusters: int | None,
    samples_per_dataset: int,
    system_prompt: str,
    user_template: str,
    api_params: dict,
    rate_limit_sleep: float,
) -> None:
    print(f"[{dataset_label}] Loading clusters and flat samples")
    df_flat = pd.read_parquet(flat_path)
    df_clusters = load_clusters(
        clusters_path,
        max_clusters,
        df_flat=df_flat,
        samples_per_dataset=samples_per_dataset,
    )

    # If no per-dataset count columns, create one from cluster_size or defaults.
    dataset_count_cols = [c for c in df_clusters.columns if c.startswith("count_")]

    df_clusters["cluster_type"] = f"only_{dataset_label}"

    labels: List[str] = []
    summaries: List[str] = []
    confidences: List[float | None] = []

    for _, row in df_clusters.iterrows():
        cluster_id = int(row["cluster_id"])
        sentences = _ensure_sentence_list(row["sample_stereotypes"])
        count_cols = {
            col: int(row[col])
            for col in df_clusters.columns
            if col.startswith("count_")
        }
        # Coarse bucket / dimension for this cluster (may be None if not available)
        dimension = row.get("dimension", None) if "dimension" in df_clusters.columns else None

        user_prompt = build_prompt(cluster_id, dimension, sentences, count_cols, user_template)
        result = call_openrouter(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            **api_params,
        )
        labels.append(result["label"])
        summaries.append(result["summary"])
        confidences.append(result.get("confidence"))
        time.sleep(rate_limit_sleep)

    df_out = df_clusters.copy()
    df_out["label"] = labels
    df_out["summary"] = summaries
    df_out["label_confidence"] = confidences

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".csv":
        df_out.to_csv(output_path, index=False)
    else:
        df_out.to_parquet(output_path, index=False)
    print(f"[{dataset_label}] Saved {len(df_out)} cluster summaries to {output_path}")


def main() -> None:
    args = parse_args()
    api_key = get_api_key()

    api_params = {
        "api_url": args.api_url,
        "api_key": api_key,
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "http_referer": args.http_referer,
        "x_title": args.x_title,
    }

    datasets = []
    if args.datasets in {"hansard", "both"}:
        if not args.hansard_clusters or not args.hansard_flat or not args.hansard_output:
            raise ValueError("Provide --hansard-clusters, --hansard-flat, and --hansard-output when labeling Hansard.")
        if not args.hansard_prompt_file:
            raise ValueError("Provide --hansard-prompt-file when labeling Hansard clusters.")
        hansard_system, hansard_user = load_prompt_sections(args.hansard_prompt_file)
        datasets.append(
            ("hansard", args.hansard_clusters, args.hansard_flat, args.hansard_output, hansard_system, hansard_user)
        )
    if args.datasets in {"trh", "both"}:
        if not args.trh_clusters or not args.trh_flat or not args.trh_output:
            raise ValueError("Provide --trh-clusters, --trh-flat, and --trh-output when labeling TRH.")
        if not args.trh_prompt_file:
            raise ValueError("Provide --trh-prompt-file when labeling TRH clusters.")
        trh_system, trh_user = load_prompt_sections(args.trh_prompt_file)
        datasets.append(
            ("trh", args.trh_clusters, args.trh_flat, args.trh_output, trh_system, trh_user)
        )

    for ds_label, clusters_path, flat_path, output_path, system_prompt, user_template in datasets:
        process_dataset(
            ds_label,
            clusters_path,
            flat_path,
            output_path,
            args.max_clusters,
            args.n_samples,
            system_prompt,
            user_template,
            api_params,
            args.rate_limit_sleep,
        )


if __name__ == "__main__":
    main()
