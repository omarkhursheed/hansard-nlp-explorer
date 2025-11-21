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
        "--clusters",
        type=Path,
        required=True,
        help="Parquet file produced by compare_stereotypes.py (--output-clusters).",
    )
    parser.add_argument(
        "--flat-samples",
        type=Path,
        default=None,
        help="Optional Parquet with cluster membership (output-flat) to select top stereotype samples by frequency.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output Parquet/CSV with cluster-level labels and summaries.",
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
    return parser.parse_args()


def build_samples_from_flat(df_flat: pd.DataFrame, max_per_dataset: int = 2) -> pd.DataFrame:
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


def load_clusters(path: Path, max_clusters: int | None, df_flat: pd.DataFrame | None = None) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"cluster_id", "sample_stereotypes", "dataset", "cluster_dataset_count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Clusters file is missing columns: {missing}")

    # Build example sentences per cluster:
    # - If a flat membership file is provided, pick top stereotypes by count (max_per_dataset=2).
    # - Otherwise, fall back to the first sample_stereotypes entry for the cluster.
    if df_flat is not None:
        sample_per_cluster = build_samples_from_flat(df_flat, max_per_dataset=2)
    else:
        sample_per_cluster = (
            df.groupby("cluster_id")["sample_stereotypes"]
            .first()
            .reset_index()
        )

    counts_wide = (
        df.groupby(["cluster_id", "dataset"])["cluster_dataset_count"]
        .sum()
        .unstack(fill_value=0)
    )

    # Prefix dataset columns for clarity, e.g. hansard -> count_hansard.
    counts_wide = counts_wide.rename(columns=lambda d: f"count_{d}")
    counts_wide = counts_wide.reset_index()

    merged = sample_per_cluster.merge(counts_wide, on="cluster_id", how="left")
    merged = merged.sort_values("cluster_id").reset_index(drop=True)

    if max_clusters is not None:
        merged = merged.head(max_clusters).copy()

    return merged


FEW_SHOT_CLUSTERS = """
Example 1:
Cluster sentences:
- "Women are too emotional to make calm decisions."
- "Females cannot think rationally when their feelings are involved."
Summary JSON:
{"label": "emotional/irrational", "summary": "Portrays women as overly emotional and incapable of calm, rational decision-making."}

Example 2:
Cluster sentences:
- "Girls are naturally weak and unfit for hard labour."
- "Women lack the physical strength for serious work."
Summary JSON:
{"label": "physically weak", "summary": "Frames women as physically fragile and unsuitable for demanding work."}

Example 3:
Cluster sentences:
- "Women must obey their husbands in all things."
- "Females were created to submit to male authority."
Summary JSON:
{"label": "submissive/domestic obedience", "summary": "Insists women are naturally subordinate to men and belong in obedient domestic roles."}
""".strip()


def build_prompt(cluster_id: int, sentences: List[str], counts: Dict[str, int]) -> str:
    examples_block = "\n".join(f"- \"{s}\"" for s in sentences)
    counts_str = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    return f"""
You are analysing clusters of stereotypes about women drawn from two historical text corpora.

Follow the style of these few-shot examples:

{FEW_SHOT_CLUSTERS}

Now analyse this cluster (id {cluster_id}).

Representative stereotype sentences:
{examples_block}

Cluster counts by dataset: {counts_str}.

Identify the main stereotype expressed across these sentences. Use a short label like
"emotional/irrational", "physically weak", "submissive/domestic obedience", "moral virtue", "manipulative/scheming",
or invent a concise label if it does not fit the examples.

Return JSON of the form:
{{
  "label": "<short stereotype label>",
  "summary": "<1-2 sentence explanation of the stereotype and how it portrays women>"
}}
""".strip()


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
    prompt: str,
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
                "content": "You are a concise analyst who labels and explains clusters of stereotypes about women.",
            },
            {"role": "user", "content": prompt},
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
        return {
            "label": str(result.get("label", "")).strip() or "parse_error",
            "summary": str(result.get("summary", "")).strip(),
        }
    except Exception:
        return {
            "label": "parse_error",
            "summary": clean_json_string(content).strip()[:500],
        }


def main() -> None:
    args = parse_args()
    api_key = get_api_key()

    df_flat = pd.read_parquet(args.flat_samples) if args.flat_samples else None
    df_clusters = load_clusters(args.clusters, args.max_clusters, df_flat=df_flat)

    # Infer whether each cluster is shared across datasets or specific to one.
    dataset_count_cols = [c for c in df_clusters.columns if c.startswith("count_")]

    def infer_cluster_type(row) -> str:
        active = [c for c in dataset_count_cols if row.get(c, 0) > 0]
        if len(active) >= 2:
            return "both"
        if len(active) == 1:
            dataset_name = active[0].replace("count_", "", 1)
            return f"only_{dataset_name}"
        return "none"

    if dataset_count_cols:
        df_clusters["cluster_type"] = df_clusters.apply(infer_cluster_type, axis=1)
    else:
        df_clusters["cluster_type"] = "unknown"

    labels: List[str] = []
    summaries: List[str] = []

    for _, row in df_clusters.iterrows():
        cluster_id = int(row["cluster_id"])
        sentences = _ensure_sentence_list(row["sample_stereotypes"])

        # Collect dataset counts (any column starting with count_).
        count_cols = {col: int(row[col]) for col in df_clusters.columns if col.startswith("count_")}

        prompt = build_prompt(cluster_id, sentences, count_cols)
        result = call_openrouter(
            prompt=prompt,
            api_url=args.api_url,
            api_key=api_key,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            http_referer=args.http_referer,
            x_title=args.x_title,
        )
        labels.append(result["label"])
        summaries.append(result["summary"])
        time.sleep(args.rate_limit_sleep)

    df_out = df_clusters.copy()
    df_out["label"] = labels
    df_out["summary"] = summaries

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".csv":
        df_out.to_csv(args.output, index=False)
    else:
        df_out.to_parquet(args.output, index=False)
    print(f"Saved {len(df_out)} cluster summaries to {args.output}")


if __name__ == "__main__":
    main()
