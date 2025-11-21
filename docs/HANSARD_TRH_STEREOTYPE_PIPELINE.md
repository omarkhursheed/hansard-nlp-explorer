# Hansard–TRH Stereotype Comparison Pipeline

This document describes the end‑to‑end pipeline for:

1. Extracting stereotypes about women from the **TRH** corpus and **Hansard**.
2. Embedding and clustering stereotypes to find shared vs dataset‑specific themes.
3. Labelling clusters with an LLM.
4. Exploring the results in the `stereotype_comparison_analysis.ipynb` notebook.

The code lives under `scripts/analysis/compare_hansard_trh/` and `notebooks/`.

---

## 0. Prerequisites

- **Environment**
  - Use the `hansard` conda env (or equivalent) with:
    - `pandas`, `pyarrow`
    - `requests`
    - `sentence-transformers`
    - `concurry`
    - `scikit-learn`
    - `matplotlib`, `seaborn` (for the notebook)

- **OpenRouter API key**
  - Obtain an API key from OpenRouter and set it in your shell:
    ```bash
    export OPENROUTER_API_KEY="sk-or-xxxxxxxx"
    ```
  - The extraction and labelling scripts all read `OPENROUTER_API_KEY` from the environment.

- **Prompts**
  - TRH extraction prompt (example): `prompts/trh_bias_women_filter_prompt.md`
  - Hansard extraction prompt: `prompts/hansard_stereotype_extraction_prompt.md`

You should run all commands below from the project root.

---

## 1. Extract stereotypes from TRH

Script:  
`scripts/analysis/compare_hansard_trh/extract_bias_TRH_data.py`

**Purpose**
- Filter a large TRH text file to statements about women (and synonyms).
- Split into chunks, call an LLM (Gemini Flash via OpenRouter) with a few‑shot prompt.
- Parse the returned JSON and write one row per stereotype instance.

**Expected input**
- A plain text file with toxic/attitudinal statements (one per line or separated by blank lines).
- A SYSTEM/USER prompt file with `SYSTEM` and `USER` sections.

**Typical command**

```bash
conda activate hansard

python scripts/analysis/compare_hansard_trh/extract_bias_TRH_data.py \
  --text-file src/hansard/data/toxicity_rabbit_hole_data/rabbit_hole_corpus.txt \
  --output outputs/trh/trh_stereotypes.parquet \
  --prompt-file prompts/trh_bias_women_filter_prompt.md \
  --cache-file outputs/trh/cache/trh_cache.parquet \
  --model google/gemini-2.5-flash \
  --chunk-size 2500 \
  --chunk-overlap 400 \
  --max-workers 4 \
  --calls-per-minute 30 \
  --max-tokens 1600 \
  --max-tokens-ceiling 2600 \
  --max-tokens-step 400
```

**Output**
- `outputs/trh/trh_stereotypes.parquet` (path is up to you) with columns like:
  - `chunk_id`
  - `stereotype_text`
  - `dimension` (e.g. `emotion`, `morality`, `political competence`, …)
  - `polarity` (`positive`, `negative`, `ambivalent`)
  - `confidence` (0–1)

Each row is treated as one stereotype instance; any aggregate `count` is added later.

**Useful options**
- `--test-chunk` — run a single chunk end‑to‑end to debug the prompt.
- `--cache-file` — JSON cache of LLM responses (skips re‑calling for unchanged chunks).
- `--llm-retries`, `--max-tokens-step`, `--max-tokens-ceiling` — robust handling of JSON parse failures.

---

## 2. Extract stereotypes from Hansard

Script:  
`scripts/analysis/compare_hansard_trh/extract_stereotypes_hansard.py`

**Purpose**
- Take Hansard **argument‑mining results** (with a `reasons` column).
- Build a compact evidence block from rationales + quotes.
- Call an LLM via OpenRouter to extract structured stereotypes.
- Write:
  - a row‑level file (one row per stereotype instance), and
  - a canonical file (deduplicated stereotypes with counts).

**Expected input**
- A Parquet file with at least:
  - `reasons` (numpy/JSON‑like structure with `rationale`, `quotes`, `stance_label`, …)
  - Optional metadata columns (`speech_id`, `debate_id`, `speaker`, `gender`, `year`, …).
- Prompt file: `prompts/hansard_stereotype_extraction_prompt.md`.

**Typical command**

```bash
conda activate hansard

python scripts/analysis/compare_hansard_trh/extract_stereotypes_hansard.py \
  --input src/hansard/data/analysis_data/arg_mining/arg_mining_results_full_claude.parquet \
  --output outputs/trh/hansard_stereotypes.parquet \
  --prompt-file prompts/hansard_stereotype_extraction_prompt.md \
  --cache outputs/trh/cache/hansard_cache.parquet \
  --model google/gemini-2.5-flash \
  --max-workers 4 \
  --calls-per-minute 30 \
  --max-rows 100 \   # optional for testing
  --debug-samples 5   # optional: print evidence blocks for early rows
```

**Outputs**
- `outputs/hansard/hansard_stereotypes.parquet`:
  - `row_id`, `stereotype_text`, `dimension`, `polarity`, `confidence`, `dataset="hansard"`, plus any attached metadata.
- `outputs/hansard/hansard_stereotypes_canonical.parquet`:
  - `stereotype_text`, `dimension`, `polarity`, `confidence` (mean), `count` (number of occurrences).

**Useful options**
- `--max-rows` — run only a subset of rows for a cheap test.
- `--debug-samples` — print raw/parsed reasons and evidence blocks for debugging.
- `--cache` — shelve cache of LLM responses by evidence hash.
- `--llm-retries`, `--max-tokens-step`, `--max-tokens-ceiling` — automatic retries with larger token budgets.

---

## 3. Embed and cluster stereotypes (Hansard + TRH)

Script:  
`scripts/analysis/compare_hansard_trh/compare_stereotypes.py`

**Purpose**
- Combine Hansard and TRH stereotype canonicals.
- Embed stereotype texts with a SentenceTransformer model.
- Cluster into `k` stereotype clusters using k‑means.
- Output:
  - a flat file with each instance and its `cluster_id`, and
  - a cluster‑level summary with sample texts and per‑dataset counts.

**Expected inputs**
- Hansard canonical file from step 2:
  - e.g. `outputs/hansard/hansard_stereotypes_canonical.parquet`, with columns:
    - `stereotype_text`, `dimension`, `polarity`, `confidence`, `count`
- TRH stereotypes file from step 1:
  - e.g. `outputs/trh/trh_stereotypes.parquet`, with:
    - `stereotype_text`, `dimension`, `polarity`, `confidence`
  - `compare_stereotypes.py` will add `count = 1` if missing.

**Typical command**

```bash
conda activate hansard

python scripts/analysis/compare_hansard_trh/compare_stereotypes.py \
  --hansard outputs/trh/hansard_stereotypes_canonical.parquet \
  --trh outputs/trh/trh_stereotypes.parquet \
  --output-flat outputs/trh/stereotypes_flattened.parquet \
  --output-clusters outputs/trh/stereotype_clusters.parquet \
  --model sentence-transformers/all-mpnet-base-v2 \
  --n-clusters 50
```

**Outputs**
- `--output-flat` (e.g. `outputs/trh/stereotypes_flattened.parquet`):
  - columns: `dataset` (`hansard` / `trh`), `stereotype_text`, `dimension`, `polarity`, `confidence`, `count`, `cluster_id`.
- `--output-clusters` (e.g. `outputs/trh/stereotype_clusters.parquet`):
  - `cluster_id`
  - `sample_stereotypes` (up to 5 example texts from the cluster)
  - `dataset`, `cluster_dataset_count` (one row per `(cluster_id, dataset)` pair).

---

## 4. Label clusters with an LLM

Script:  
`scripts/analysis/compare_hansard_trh/summarize_stereotype_clusters.py`

**Purpose**
- For each `cluster_id`, send representative stereotype sentences + per‑dataset counts to the LLM.
- Ask for:
  - a short **stereotype label** (e.g. `emotional/irrational`, `physically weak`, `politically immature`), and
  - a 1–2 sentence **summary**.
- Infer whether the cluster appears in:
  - **both** datasets,
  - **only_hansard**, or
  - **only_trh**.

**Expected input**
- Cluster summary from step 3 (`--output-clusters`).

**Typical command**

```bash
conda activate hansard

python scripts/analysis/compare_hansard_trh/summarize_stereotype_clusters.py \
  --clusters outputs/trh/stereotype_clusters.parquet \
  --output outputs/trh/stereotype_clusters_labeled.parquet \
  --model google/gemini-2.5-flash \
  --max-clusters 100   # optional for a cheap first pass
```

**Output**
- `outputs/trh/stereotype_clusters_labeled.parquet` with one row per cluster:
  - `cluster_id`
  - `sample_stereotypes`
  - `count_hansard`, `count_trh`, …
  - `cluster_type` ∈ `{both, only_hansard, only_trh, none}`
  - `label` (short stereotype name)
  - `summary` (1–2 sentence explanation)

**Notes**
- `cluster_type` is derived from the `count_*` columns:
  - `both` → cluster has stereotypes from ≥2 datasets (Hansard & TRH).
  - `only_hansard` → only Hansard rows in that cluster.
  - `only_trh` → only TRH rows in that cluster.

---

## 5. Interactive analysis in the notebook

Notebook:  
`notebooks/stereotype_comparison_analysis.ipynb`

**Purpose**
- Join instance‑level stereotypes with cluster labels.
- Compare the distribution of labels and dimensions across Hansard and TRH.
- Visualize shared vs dataset‑specific stereotypes and inspect example sentences.

**Configuration**

At the top of the notebook, set the input paths to match your pipeline outputs (e.g. from steps 3–4):

```python
from pathlib import Path

# Paths to inputs produced by the CLI scripts
flat_path = project_root / "outputs" / "trh/stereotypes_flattened.parquet"     # compare_stereotypes.py --output-flat
clusters_path = project_root / "outputs" / "trh/stereotype_clusters_labeled.parquet"  # summarize_stereotype_clusters.py --output
```

**What the notebook does**

1. **Setup & load**
   - Detects `project_root`, adds `src` to `sys.path`.
   - Loads `df_flat` and `df_clusters`.

2. **Join**
   - Merges on `cluster_id` to produce `df_joined` with:
     - `dataset`, `stereotype_text`, `dimension`, `polarity`, `confidence`, `count`, `cluster_id`
     - `label`, `summary`, `cluster_type`.

3. **Aggregation**
   - Builds a `summary` table:
     ```python
     summary = (
         df_joined
         .groupby(["label", "cluster_type", "dataset"], dropna=False)
         .agg(
             n_rows=("stereotype_text", "size"),
             n_unique_stereotypes=("stereotype_text", "nunique"),
             total_count=("count", "sum"),
         )
         .reset_index()
     )
     ```

4. **Visualizations**
   - **Top shared stereotypes** (cluster_type = `both`) by dataset.
   - **Shared vs dataset‑specific stereotypes** (`both`, `only_hansard`, `only_trh`) per dataset.
   - **Dimension distribution** by dataset.
   - **Polarity distribution** by dataset.
   - **Confidence boxplot** by dataset.

4. **Example inspection**
   - Choose a `label` and `cluster_type` (e.g. `"inferior/subordinate (racialized)"`, `"both"`).
   - See up to 20 example stereotype sentences with their dataset, dimension, polarity, and `cluster_id`.

---

## Summary of the full pipeline

1. **TRH extraction**  
   `extract_bias_TRH_data.py` → `trh_stereotypes.parquet`
2. **Hansard extraction**  
   `extract_stereotypes_hansard.py` → `hansard_stereotypes.parquet` + `hansard_stereotypes_canonical.parquet`
3. **Embedding & clustering**  
   `compare_stereotypes.py` → `stereotypes_flattened.parquet` + `stereotype_clusters.parquet`
4. **Cluster labelling**  
   `summarize_stereotype_clusters.py` → `stereotype_clusters_labeled.parquet`
5. **Interactive analysis**  
   `stereotype_comparison_analysis.ipynb` → tables, plots, and qualitative inspection of shared vs differing stereotypes.

