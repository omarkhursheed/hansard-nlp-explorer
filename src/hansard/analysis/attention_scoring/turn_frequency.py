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

# -------- 1) Per-debate speaker turns --------
# One row per (debate_id, speaker, gender) with observed turns
speaker_turns = (
    df.groupby(['debate_id', 'speaker', 'gender'], dropna=False)
      .size()
      .reset_index(name='turns')
)

# -------- 2) Debate-level totals --------
debate_totals = (
    speaker_turns.groupby('debate_id', as_index=False)
    .agg(total_turns=('turns', 'sum'),
         total_speakers=('speaker', 'nunique'))
)

# -------- 3) Gender-level expected baseline inside each debate --------
# (a) count distinct speakers per gender within each debate
gender_speaker_counts = (
    speaker_turns.groupby(['debate_id', 'gender'], as_index=False)
    .agg(n_speakers=('speaker', 'nunique'),
         gender_turns=('turns', 'sum'))
)

# (b) merge debate totals to compute participation share (by #speakers)
gender_participation = gender_speaker_counts.merge(debate_totals, on='debate_id', how='left')
gender_participation['participation_share'] = (
    gender_participation['n_speakers'] / gender_participation['total_speakers']
)

# (c) expected turns for each gender in that debate
gender_participation['expected_turns_gender'] = (
    gender_participation['participation_share'] * gender_participation['total_turns']
)

# -------- 4) Normalize speaker turns by gender-expected baseline (within debate) --------
speaker_norm = speaker_turns.merge(
    gender_participation[['debate_id', 'gender', 'expected_turns_gender']],
    on=['debate_id', 'gender'],
    how='left'
)

# if a debate has only one gender, expected_turns_gender = total_turns, which is fine.
# Guard against divide-by-zero
speaker_norm['norm_turns'] = speaker_norm['turns'] / np.where(
    speaker_norm['expected_turns_gender'] == 0, np.nan, speaker_norm['expected_turns_gender']
)

# Optional: cap or clip extreme values if desired
# speaker_norm['norm_turns'] = speaker_norm['norm_turns'].clip(upper=5)

# -------- 4b) Flat per-speaker normalization (baseline = equal share for all MPs) --------
# Merge in total_turns and total_speakers for each debate
speaker_flat = speaker_turns.merge(
    debate_totals[['debate_id','total_turns','total_speakers']],
    on='debate_id', how='left'
)

# Expected turns = equal share per speaker
speaker_flat['expected_turns_flat'] = speaker_flat['total_turns'] / speaker_flat['total_speakers']

# Flat normalization: observed / expected
speaker_flat['flat_norm_turns'] = speaker_flat['turns'] / np.where(
    speaker_flat['expected_turns_flat'] == 0,
    np.nan,
    speaker_flat['expected_turns_flat']
)

# Attach metadata (year, decade, chamber) for consistency
speaker_flat_meta = speaker_flat.merge(
    df[['debate_id','year','decade','chamber']].drop_duplicates('debate_id'),
    on='debate_id', how='left'
)


# -------- 5) Average per speaker across the debates they attended --------
speaker_scores = (
    speaker_norm.groupby(['speaker', 'gender'], dropna=False)
    .agg(
        mean_norm_turns=('norm_turns', 'mean'),     # unweighted average across debates
        debates_participated=('debate_id', 'nunique'),
        total_turns=('turns', 'sum')                # raw turns (diagnostic)
    )
    .reset_index()
)

# -------- 6) Compare groups by gender (speaker-weighted) --------
gender_summary = (
    speaker_scores.groupby('gender', dropna=False)['mean_norm_turns']
    .mean()
    .reset_index(name='avg_mean_norm_turns')
)

# -------- 7) Weighted speaker averages by debate size/importance --------
# Example A: weight each speaker's per-debate norm_turns by debate total_turns
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

# -------- 8) Stratify by time / chamber --------
# Per-decade gender comparison (unweighted)
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

# -------- 8b) Group-level normalized turns (M vs F per debate) --------
# Observed turns per gender per debate
group_turns = (
    df.groupby(['debate_id','gender'])
      .size()
      .reset_index(name='observed_turns')
)

# Number of unique speakers per gender per debate
gender_speaker_counts = (
    df.groupby(['debate_id','gender'])['speaker']
      .nunique()
      .reset_index(name='n_speakers')
)

# Debate totals
debate_totals_group = (
    df.groupby('debate_id')
      .agg(total_turns=('speaker','count'),
           total_speakers=('speaker','nunique'))
      .reset_index()
)

# Participation share by gender (within each debate)
gender_participation_group = gender_speaker_counts.merge(debate_totals_group, on='debate_id', how='left')
gender_participation_group['participation_share'] = (
    gender_participation_group['n_speakers'] /
    gender_participation_group.groupby('debate_id')['n_speakers'].transform('sum')
)

# Expected turns for each gender
gender_participation_group['expected_turns'] = (
    gender_participation_group['participation_share'] * gender_participation_group['total_turns']
)

# Merge observed + expected
group_norm = group_turns.merge(
    gender_participation_group[['debate_id','gender','expected_turns']],
    on=['debate_id','gender'],
    how='left'
)

# Compute normalized turns at group level
group_norm['group_norm_turns'] = group_norm['observed_turns'] / np.where(
    group_norm['expected_turns']==0, np.nan, group_norm['expected_turns']
)

# Attach metadata (year, decade, chamber)
group_norm = group_norm.merge(
    df[['debate_id','year','decade','chamber']].drop_duplicates('debate_id'),
    on='debate_id', how='left'
)

# -------- 9) Outputs --------
print("Speaker-level (unweighted) mean normalized turns:")
print(speaker_scores.head())

print("\nGender summary (speaker-weighted, unweighted across debates):")
print(gender_summary)

print("\nGender summary (speaker-weighted, debate-size weighted):")
print(gender_summary_weighted.head())

print("\nPer-decade gender comparison (unweighted across speakers):")
print(per_decade_gender.head())

# Print overall averages by gender (macro-average, unweighted over debates with that gender present)
avg_group = group_norm.groupby('gender')['group_norm_turns'].mean().reset_index()
print("\nGroup-level normalized turns (average across debates):")
print(avg_group)

# --- Sanity checks for group-level norm ---

# 1) Micro-average weighted by expected turns (equivalent to sum(obs)/sum(exp))
avg_group_weighted = group_norm.groupby('gender', dropna=False) \
    .apply(lambda g: np.average(g['group_norm_turns'], weights=g['expected_turns'])) \
    .reset_index(name='avg_group_norm_weighted')
print("\nGroup-level (weighted by expected turns):")
print(avg_group_weighted)

# 2) Explicit global ratio = sum(observed)/sum(expected) — should match (1)
global_obs = group_norm.groupby('gender', dropna=False)['observed_turns'].sum()
global_exp = group_norm.groupby('gender', dropna=False)['expected_turns'].sum()
global_ratio = (global_obs / global_exp).reset_index(name='global_observed_over_expected')
print("\nGlobal observed / expected turns (sanity check):")
print(global_ratio)

# 3) Also print raw global shares to compare with your 2.95M vs 116k intuition
raw_counts = df['gender'].value_counts().rename_axis('gender').reset_index(name='raw_turns')
raw_counts['raw_share'] = raw_counts['raw_turns'] / raw_counts['raw_turns'].sum()
print("\nRaw total turns and shares (no normalization):")
print(raw_counts)

# -------- Flat baseline summaries --------
# Average per speaker across debates
speaker_scores_flat = (
    speaker_flat.groupby(['speaker','gender'], dropna=False)
    .agg(
        mean_flat_norm_turns=('flat_norm_turns','mean'),
        debates_participated=('debate_id','nunique'),
        total_turns=('turns','sum')
    )
    .reset_index())

# Gender-level summary (per-speaker averages)
gender_summary_flat = (
    speaker_scores_flat.groupby('gender', dropna=False)['mean_flat_norm_turns']
    .mean()
    .reset_index(name='avg_mean_flat_norm_turns'))

print("\nGender summary (flat per-speaker baseline):")
print(gender_summary_flat)


# -------- 10) Save file --------

# Merge in debate metadata so the final dataset is self-contained
speaker_norm_meta = speaker_norm.merge(
    df[['debate_id','year','decade','month','reference_date','chamber','title','topic']]
      .drop_duplicates('debate_id'),
    on='debate_id',
    how='left'
)

# Merge flat baseline into speaker_norm_meta
speaker_norm_meta = speaker_norm_meta.merge(
    speaker_flat[['debate_id','speaker','flat_norm_turns','expected_turns_flat']],
    on=['debate_id','speaker'], how='left'
)

# Save final dataset (one row per speaker per debate with normalized turns and metadata)
speaker_norm_meta.to_parquet(OUT_PATH_SPEAKER, index=False)
group_norm.to_parquet(OUT_PATH_GROUP, index=False)

print("Final datasets saved.")