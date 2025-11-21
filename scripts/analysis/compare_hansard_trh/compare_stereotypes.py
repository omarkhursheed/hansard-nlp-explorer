#!/usr/bin/env python3
"""
Combine Hansard + TRH stereotype canonicals, embed, cluster, and compare.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flatten and cluster Hansard/TRH stereotypes.")
    parser.add_argument("--hansard", type=Path, required=True, help="Hansard canonical stereotypes Parquet.")
    parser.add_argument("--trh", type=Path, required=True, help="TRH canonical stereotypes Parquet/JSON.")
    parser.add_argument("--output-flat", type=Path, required=True, help="Output Parquet with combined stereotypes.")
    parser.add_argument("--output-clusters", type=Path, required=True, help="Output Parquet with cluster assignments.")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-mpnet-base-v2", help="Embedding model.")
    parser.add_argument("--n-clusters", type=int, default=20, help="Number of clusters for k-means.")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size.")
    return parser.parse_args()


def load_hansard(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # Hansard canonical outputs use `stereotype_text` for the text column.
    required = {"stereotype_text", "dimension", "polarity", "confidence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Hansard file is missing columns: {missing}")

    if "count" not in df.columns:
        df["count"] = 1
    return df


def load_trh(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".json":
        df = pd.read_json(path)
    else:
        df = pd.read_parquet(path)

    # TRH canonical files also use `stereotype_text` for the text column.
    required = {"stereotype_text", "dimension", "polarity", "confidence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"TRH file is missing columns: {missing}")

    if "count" not in df.columns:
        df["count"] = 1
    return df


def normalize_df(df: pd.DataFrame, dataset_label: str) -> pd.DataFrame:
    df = df.copy()
    df["dataset"] = dataset_label
    cols = ["dataset", "stereotype_text", "dimension", "polarity", "confidence", "count"]
    return df[cols]


def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def main() -> None:
    args = parse_args()

    df_h = normalize_df(load_hansard(args.hansard), "hansard")
    df_t = normalize_df(load_trh(args.trh), "trh")
    print(f"Loaded {len(df_h)} Hansard and {len(df_t)} TRH stereotypes")

    # Combine into a single table with dataset markers.
    df_flat = pd.concat([df_h, df_t], ignore_index=True)

    # Compute unique stereotype texts (across both datasets) for clustering.
    unique = df_flat.drop_duplicates(subset=["stereotype_text"]).reset_index(drop=True)
    print(f"Found {len(unique)} unique stereotype texts for clustering")

    print(f"Loading embedding model: {args.model}")
    model = SentenceTransformer(args.model)
    print("Embedding stereotype texts…")
    embeddings = embed_texts(model, unique["stereotype_text"].tolist(), batch_size=args.batch_size)

    print(f"Clustering into {args.n_clusters} clusters with k-means")
    km = KMeans(n_clusters=args.n_clusters, random_state=42)
    labels = km.fit_predict(embeddings)
    unique["cluster_id"] = labels

    # Attach cluster_id back to each (dataset, stereotype_text, …) row so that
    # we know which stereotypes (and from which dataset) belong to which cluster.
    df_membership = df_flat.merge(
        unique[["stereotype_text", "cluster_id"]],
        on="stereotype_text",
        how="left",
    )

    # Save the combined flat table with cluster assignments.
    args.output_flat.parent.mkdir(parents=True, exist_ok=True)
    df_membership.to_parquet(args.output_flat, index=False)
    print(f"Wrote combined stereotypes with cluster_ids to {args.output_flat}")

    # Summarise cluster composition by dataset and provide a few example texts.
    cluster_counts = (
        df_membership.groupby(["cluster_id", "dataset"])
        .size()
        .rename("cluster_dataset_count")
        .reset_index()
    )
    cluster_summary = (
        df_membership.groupby("cluster_id")["stereotype_text"]
        .apply(lambda x: list(x.head(5)))
        .rename("sample_stereotypes")
        .reset_index()
    )

    result = cluster_summary.merge(cluster_counts, on="cluster_id", how="left").sort_values(
        ["cluster_id", "dataset"]
    )
    args.output_clusters.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output_clusters, index=False)
    print(f"Wrote cluster summaries to {args.output_clusters}")


if __name__ == "__main__":
    main()
