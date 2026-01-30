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
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flatten and cluster Hansard/TRH stereotypes separately.")
    parser.add_argument("--hansard", type=Path, required=True, help="Hansard stereotypes Parquet.")
    parser.add_argument("--trh", type=Path, required=True, help="TRH stereotypes Parquet/JSON.")
    parser.add_argument("--hansard-flat", type=Path, required=True, help="Output Parquet with Hansard stereotypes + cluster_id.")
    parser.add_argument("--hansard-clusters", type=Path, required=True, help="Output Parquet with Hansard cluster summaries.")
    parser.add_argument("--trh-flat", type=Path, required=True, help="Output Parquet with TRH stereotypes + cluster_id.")
    parser.add_argument("--trh-clusters", type=Path, required=True, help="Output Parquet with TRH cluster summaries.")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-mpnet-base-v2", help="Embedding model.")
    parser.add_argument("--n-clusters-hansard", type=int, default=20, help="Number of clusters for Hansard k-means.")
    parser.add_argument("--n-clusters-trh", type=int, default=20, help="Number of clusters for TRH k-means.")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size.")
    return parser.parse_args()


def load_hansard(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"stereotype_text", "dimension", "polarity", "confidence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Hansard file is missing columns: {missing}")

    if "count" not in df.columns:
        df["count"] = 1
    before = len(df)
    df = df[df["confidence"] >= 0.5].reset_index(drop=True)
    dropped = before - len(df)
    print(f"[Hansard] Dropped {dropped} low-confidence rows (<0.5); remaining {len(df)}")
    return df


def load_trh(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".json":
        df = pd.read_json(path)
    else:
        df = pd.read_parquet(path)

    # TRH files also use `stereotype_text` for the text column.
    required = {"stereotype_text", "dimension", "polarity", "confidence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"TRH file is missing columns: {missing}")

    if "count" not in df.columns:
        df["count"] = 1
    before = len(df)
    df = df[df["confidence"] >= 0.5].reset_index(drop=True)
    dropped = before - len(df)
    print(f"[TRH] Dropped {dropped} low-confidence rows (<0.5); remaining {len(df)}")
    return df


def normalize_df(df: pd.DataFrame, dataset_label: str) -> pd.DataFrame:
    df = df.copy()
    df["dataset"] = dataset_label
    cols = ["dataset"] + [c for c in df.columns if c != "dataset"]
    return df[cols]


def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def cluster_dataset(
    df: pd.DataFrame,
    dataset_label: str,
    model: SentenceTransformer,
    n_clusters: int,
    batch_size: int,
    output_flat: Path,
    output_clusters: Path,
) -> None:
    """
    Cluster stereotypes for a single dataset, but *within* dimension buckets.

    - Embeds all unique (stereotype_text, dimension) pairs once.
    - For each dimension, runs KMeans on the subset of embeddings.
    - Assigns globally unique cluster_id across all dimensions.
    - Writes a flat membership file and a cluster summary file.
    """
    print(f"[{dataset_label}] Starting clustering over {len(df)} rows")

    # Make sure we don't collapse the same text with different dimensions
    if "dimension" not in df.columns:
        raise ValueError(f"{dataset_label} dataframe is missing 'dimension' column")

    unique = df.drop_duplicates(subset=["stereotype_text", "dimension"]).reset_index(drop=True)
    print(f"[{dataset_label}] Unique (stereotype_text, dimension): {len(unique)}")

    # Embed all unique texts once
    embeddings = embed_texts(
        model,
        unique["stereotype_text"].tolist(),
        batch_size=batch_size,
    )

    # Cluster *within* each dimension bucket
    cluster_id_counter = 0
    cluster_pieces: list[pd.DataFrame] = []

    for dim_value, sub in unique.groupby("dimension"):
        sub = sub.copy()
        idx = sub.index.to_numpy()
        sub_emb = embeddings[idx]

        n_items = len(sub)
        if n_items == 0:
            continue

        # Choose number of clusters for this bucket:
        # at most n_clusters, but not more than the number of items
        k = min(n_clusters, n_items)

        print(f"[{dataset_label}] Clustering dimension='{dim_value}' with {n_items} items into {k} clusters")

        km = KMeans(n_clusters=k, random_state=42)
        local_labels = km.fit_predict(sub_emb)

        # Map local cluster IDs to global cluster IDs
        sub["cluster_id"] = local_labels + cluster_id_counter
        cluster_id_counter += k

        cluster_pieces.append(sub[["stereotype_text", "dimension", "cluster_id"]])

    # Concatenate per-dimension cluster assignments
    clustered_unique = pd.concat(cluster_pieces, ignore_index=True)

    # Merge cluster_id back onto the full dataframe (keep all metadata + count)
    df_membership = df.merge(
        clustered_unique,
        on=["stereotype_text", "dimension"],
        how="left",
    )

    output_flat.parent.mkdir(parents=True, exist_ok=True)
    df_membership.to_parquet(output_flat, index=False)
    print(f"[{dataset_label}] Wrote flat stereotypes with clusters to {output_flat}")

    # Build cluster summaries (include dimension for inspection)
    cluster_summary = (
        df_membership.groupby(["cluster_id", "dimension"])["stereotype_text"]
        .apply(lambda x: list(x.head(5)))
        .rename("sample_stereotypes")
        .reset_index()
    )
    cluster_sizes = (
        df_membership.groupby(["cluster_id", "dimension"])
        .size()
        .rename("cluster_size")
        .reset_index()
    )

    result = (
        cluster_summary
        .merge(cluster_sizes, on=["cluster_id", "dimension"], how="left")
        .sort_values("cluster_id")
    )
    output_clusters.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_clusters, index=False)
    print(f"[{dataset_label}] Wrote cluster summaries to {output_clusters}")



def main() -> None:
    args = parse_args()

    df_h = normalize_df(load_hansard(args.hansard), "hansard")
    df_t = normalize_df(load_trh(args.trh), "trh")
    print(f"Loaded {len(df_h)} Hansard and {len(df_t)} TRH stereotypes")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model: {args.model} on {device}")
    model = SentenceTransformer(args.model, device=device)

    cluster_dataset(
        df_h,
        "hansard",
        model,
        args.n_clusters_hansard,
        args.batch_size,
        args.hansard_flat,
        args.hansard_clusters,
    )
    cluster_dataset(
        df_t,
        "trh",
        model,
        args.n_clusters_trh,
        args.batch_size,
        args.trh_flat,
        args.trh_clusters,
    )


if __name__ == "__main__":
    main()
