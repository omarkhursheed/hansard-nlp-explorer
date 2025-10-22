"""
Hansard Turnwise Attention Analysis
===================================

This script analyzes parliamentary speaking behavior using the
turnwise debates dataset (`turnwise_debates_matched_speakers.parquet`).

It computes:
1. Per-debate speaker turn counts and expected baselines by gender.
2. Normalized participation scores per speaker and per debate.
3. Weighted and unweighted gender-level summaries (group-level parity).
4. Per-decade and per-chamber breakdowns of normalized speaking turns.
5. Both speaker-level and group-level normalization datasets are saved
   for downstream visualization or statistical analysis.

Outputs:
    - OUT_PATH_SPEAKER : speaker-level normalized participation
    - OUT_PATH_GROUP   : group-level (gender) normalized participation
"""

import pandas as pd
import numpy as np
from pathlib import Path

INPUT_PATH = Path("src/hansard/data/processed_fixed/cleaned_data/turnwise_debates_matched_speakers.parquet")
OUT_PATH_SPEAKER = Path("src/hansard/data/analysis_data/attention_turn_freq_ind.parquet")
OUT_PATH_GROUP = Path("src/hansard/data/analysis_data/attention_turn_freq_group.parquet")

# ==== Load data ====
print("🔹 Loading parquet…")
df = pd.read_parquet(INPUT_PATH)

print("Original shape:", df.shape)
print(df["gender"].value_counts())

# Remove unknown gender speakers before analysis
df = df[df["gender"].isin(["M", "F"])]
print("After removing UNK:", df.shape)
print(df["gender"].value_counts())

# ---------------------------------------------------------------------
"""1) Compute per-debate speaker turns.
Each row = (debate_id, speaker, gender) with observed number of speaking turns."""
speaker_turns = (
    df.groupby(['debate_id', 'speaker', 'gender'], dropna=False)
      .size()
      .reset_index(name='turns')
)

# ---------------------------------------------------------------------
"""2) Debate-level totals: total turns and number of unique speakers per debate."""
debate_totals = (
    speaker_turns.groupby('debate_id', as_index=False)
    .agg(total_turns=('turns', 'sum'),
         total_speakers=('speaker', 'nunique'))
)

# ---------------------------------------------------------------------
"""3) Compute gender-level expected baselines within each debate.
   (a) Count distinct speakers per gender
   (b) Compute share of total speakers
   (c) Estimate expected turns by gender"""
gender_speaker_counts = (
    speaker_turns.groupby(['debate_id', 'gender'], as_index=False)
    .agg(n_speakers=('speaker', 'nunique'),
         gender_turns=('turns', 'sum'))
)

gender_participation = gender_speaker_counts.merge(debate_totals, on='debate_id', how='left')
gender_participation['participation_share'] = (
    gender_participation['n_speakers'] / gender_participation['total_speakers']
)
gender_participation['expected_turns_gender'] = (
    gender_participation['participation_share'] * gender_participation['total_turns']
)

# ---------------------------------------------------------------------
"""4) Normalize speaker turns by gender-expected baseline (within debate)."""
speaker_norm = speaker_turns.merge(
    gender_participation[['debate_id', 'gender', 'expected_turns_gender']],
    on=['debate_id', 'gender'],
    how='left'
)

speaker_norm['norm_turns'] = speaker_norm['turns'] / np.where(
    speaker_norm['expected_turns_gender'] == 0, np.nan, speaker_norm['expected_turns_gender']
)

# ---------------------------------------------------------------------
"""4b) Flat per-speaker normalization.
Baseline = equal share of turns among all speakers in that debate."""
speaker_flat = speaker_turns.merge(
    debate_totals[['debate_id','total_turns','total_speakers']],
    on='debate_id', how='left'
)
speaker_flat['expected_turns_flat'] = speaker_flat['total_turns'] / speaker_flat['total_speakers']
speaker_flat['flat_norm_turns'] = speaker_flat['turns'] / np.where(
    speaker_flat['expected_turns_flat'] == 0,
    np.nan,
    speaker_flat['expected_turns_flat']
)
speaker_flat_meta = speaker_flat.merge(
    df[['debate_id','year','decade','chamber']].drop_duplicates('debate_id'),
    on='debate_id', how='left'
)

# ---------------------------------------------------------------------
"""5) Aggregate per-speaker statistics across all debates."""
speaker_scores = (
    speaker_norm.groupby(['speaker', 'gender'], dropna=False)
    .agg(
        mean_norm_turns=('norm_turns', 'mean'),
        debates_participated=('debate_id', 'nunique'),
        total_turns=('turns', 'sum')
    )
    .reset_index()
)

# ---------------------------------------------------------------------
"""6) Gender-level comparison (unweighted averages across speakers)."""
gender_summary = (
    speaker_scores.groupby('gender', dropna=False)['mean_norm_turns']
    .mean()
    .reset_index(name='avg_mean_norm_turns')
)

# ---------------------------------------------------------------------
"""7) Weighted averages: weight speaker-level norm_turns by debate size."""
speaker_norm_w = speaker_norm.merge(debate_totals[['debate_id', 'total_turns']], on='debate_id', how='left')
w = speaker_norm_w['total_turns']

speaker_scores_weighted = (
    speaker_norm_w.groupby(['speaker', 'gender'], dropna=False)
    .apply(lambda g: np.average(g['norm_turns'].dropna(), weights=w.loc[g.index].fillna(0))
           if g['norm_turns'].notna().any() else np.nan)
    .reset_index(name='mean_norm_turns_weighted')
)
gender_summary_weighted = (
    speaker_scores_weighted.groupby('gender', dropna=False)['mean_norm_turns_weighted']
    .mean()
    .reset_index(name='avg_mean_norm_turns_weighted')
)

# ---------------------------------------------------------------------
"""8) Temporal stratification: compute per-decade averages of normalized turns."""
speaker_norm_meta = speaker_norm.merge(
    df[['debate_id','year','decade','chamber']].drop_duplicates('debate_id'),
    on='debate_id',
    how='left'
)
per_decade = (
    speaker_norm_meta.groupby(['decade','speaker','gender'], dropna=False)['norm_turns']
    .mean()
    .reset_index(name='mean_norm_turns_decade')
)
per_decade_gender = (
    per_decade.groupby(['decade','gender'], dropna=False)['mean_norm_turns_decade']
    .mean()
    .reset_index()
    .sort_values(['decade','gender'])
)

# ---------------------------------------------------------------------
"""8b) Group-level normalization: compare aggregate turns for M vs F per debate."""
group_turns = (
    df.groupby(['debate_id','gender'])
      .size()
      .reset_index(name='observed_turns')
)
gender_speaker_counts = (
    df.groupby(['debate_id','gender'])['speaker']
      .nunique()
      .reset_index(name='n_speakers')
)
debate_totals_group = (
    df.groupby('debate_id')
      .agg(total_turns=('speaker','count'),
           total_speakers=('speaker','nunique'))
      .reset_index()
)
gender_participation_group = gender_speaker_counts.merge(debate_totals_group, on='debate_id', how='left')
gender_participation_group['participation_share'] = (
    gender_participation_group['n_speakers'] /
    gender_participation_group.groupby('debate_id')['n_speakers'].transform('sum')
)
gender_participation_group['expected_turns'] = (
    gender_participation_group['participation_share'] * gender_participation_group['total_turns']
)
group_norm = group_turns.merge(
    gender_participation_group[['debate_id','gender','expected_turns']],
    on=['debate_id','gender'],
    how='left'
)
group_norm['group_norm_turns'] = group_norm['observed_turns'] / np.where(
    group_norm['expected_turns']==0, np.nan, group_norm['expected_turns']
)
group_norm = group_norm.merge(
    df[['debate_id','year','decade','chamber']].drop_duplicates('debate_id'),
    on='debate_id', how='left'
)

# ---------------------------------------------------------------------
"""9) Print intermediate summaries for validation and sanity checks."""
print("Speaker-level (unweighted) mean normalized turns:")
print(speaker_scores.head())

print("\nGender summary (speaker-weighted, unweighted across debates):")
print(gender_summary)

print("\nGender summary (speaker-weighted, debate-size weighted):")
print(gender_summary_weighted.head())

print("\nPer-decade gender comparison (unweighted across speakers):")
print(per_decade_gender.head())

avg_group = group_norm.groupby('gender')['group_norm_turns'].mean().reset_index()
print("\nGroup-level normalized turns (average across debates):")
print(avg_group)

avg_group_weighted = group_norm.groupby('gender', dropna=False) \
    .apply(lambda g: np.average(g['group_norm_turns'], weights=g['expected_turns'])) \
    .reset_index(name='avg_group_norm_weighted')
print("\nGroup-level (weighted by expected turns):")
print(avg_group_weighted)

global_obs = group_norm.groupby('gender', dropna=False)['observed_turns'].sum()
global_exp = group_norm.groupby('gender', dropna=False)['expected_turns'].sum()
global_ratio = (global_obs / global_exp).reset_index(name='global_observed_over_expected')
print("\nGlobal observed / expected turns (sanity check):")
print(global_ratio)

raw_counts = df['gender'].value_counts().rename_axis('gender').reset_index(name='raw_turns')
raw_counts['raw_share'] = raw_counts['raw_turns'] / raw_counts['raw_turns'].sum()
print("\nRaw total turns and shares (no normalization):")
print(raw_counts)

# ---------------------------------------------------------------------
"""Flat baseline summaries: compute averages using equal-turn baseline."""
speaker_scores_flat = (
    speaker_flat.groupby(['speaker','gender'], dropna=False)
    .agg(
        mean_flat_norm_turns=('flat_norm_turns','mean'),
        debates_participated=('debate_id','nunique'),
        total_turns=('turns','sum')
    )
    .reset_index()
)
gender_summary_flat = (
    speaker_scores_flat.groupby('gender', dropna=False)['mean_flat_norm_turns']
    .mean()
    .reset_index(name='avg_mean_flat_norm_turns')
)
print("\nGender summary (flat per-speaker baseline):")
print(gender_summary_flat)

# ---------------------------------------------------------------------
"""10) Save outputs: individual- and group-level normalized datasets."""
speaker_norm_meta = speaker_norm.merge(
    df[['debate_id','year','decade','month','reference_date','chamber','title','topic']]
      .drop_duplicates('debate_id'),
    on='debate_id',
    how='left'
)
speaker_norm_meta = speaker_norm_meta.merge(
    speaker_flat[['debate_id','speaker','flat_norm_turns','expected_turns_flat']],
    on=['debate_id','speaker'], how='left'
)
speaker_norm_meta.to_parquet(OUT_PATH_SPEAKER, index=False)
group_norm.to_parquet(OUT_PATH_GROUP, index=False)

print("Final datasets saved.")
