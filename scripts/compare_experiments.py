"""
Compare classification results across different LLM experiments.
"""
import pandas as pd
import json
from pathlib import Path

# Experiment configuration
EXPERIMENTS = {
    "gpt4o_mini": {
        "name": "GPT-4o-mini (baseline)",
        "results": "outputs/llm_classification/full_results_v6_context_3_expanded.parquet",
        "log": None,  # No log file for this one
    },
    "claude_sonnet_45": {
        "name": "Claude Sonnet 4.5 (pilot)",
        "results": "outputs/llm_classification/claude_sonnet_45_pilot_results.parquet",
        "log": "outputs/llm_classification/claude_sonnet_45_pilot_log.json",
    },
}

def analyze_experiment(results_path, log_path=None):
    """Analyze a single experiment."""
    df = pd.read_parquet(results_path)

    analysis = {
        "total_speeches": len(df),
        "successful": df['api_success'].sum() if 'api_success' in df else len(df),
        "failed": (~df['api_success']).sum() if 'api_success' in df else 0,
    }

    # Stance distribution
    stance_counts = df['stance'].value_counts()
    analysis["stance_distribution"] = stance_counts.to_dict()
    analysis["stance_percentages"] = {
        k: f"{100*v/len(df):.1f}%"
        for k, v in stance_counts.items()
    }

    # Women AGAINST suffrage (critical test case)
    if 'gender' in df.columns:
        women_against = df[(df['gender'] == 'F') & (df['stance'] == 'against')]
        analysis["women_against_count"] = len(women_against)
        analysis["women_against_speeches"] = women_against[['speech_id', 'speaker', 'year']].to_dict('records')
    else:
        analysis["women_against_count"] = "N/A (no gender data)"

    # Load log if available
    if log_path and Path(log_path).exists():
        with open(log_path) as f:
            log_data = json.load(f)
            analysis["tokens"] = log_data.get("total_tokens")
            analysis["elapsed_minutes"] = log_data.get("elapsed_minutes")
            analysis["model"] = log_data.get("model")

    return analysis

def compare_women_against(exp1_df, exp2_df):
    """Compare women AGAINST classifications between two experiments."""
    women1 = set(exp1_df[(exp1_df['gender'] == 'F') & (exp1_df['stance'] == 'against')]['speech_id'])
    women2 = set(exp2_df[(exp2_df['gender'] == 'F') & (exp2_df['stance'] == 'against')]['speech_id'])

    only_in_exp1 = women1 - women2
    only_in_exp2 = women2 - women1
    in_both = women1 & women2

    return {
        "only_in_exp1": len(only_in_exp1),
        "only_in_exp2": len(only_in_exp2),
        "in_both": len(in_both),
        "speeches_only_in_exp1": list(only_in_exp1),
        "speeches_only_in_exp2": list(only_in_exp2),
    }

def main():
    print("="*70)
    print("EXPERIMENT COMPARISON")
    print("="*70)

    # Analyze each experiment
    results = {}
    for exp_id, exp_config in EXPERIMENTS.items():
        results_path = exp_config["results"]
        if not Path(results_path).exists():
            print(f"\nSkipping {exp_id}: {results_path} not found")
            continue

        print(f"\n{exp_config['name']}")
        print("-" * 70)

        analysis = analyze_experiment(results_path, exp_config.get("log"))
        results[exp_id] = analysis

        print(f"Speeches: {analysis['successful']}/{analysis['total_speeches']} successful")
        if analysis.get("tokens"):
            print(f"Tokens: {analysis['tokens']:,}")
        if analysis.get("elapsed_minutes"):
            print(f"Time: {analysis['elapsed_minutes']:.1f} minutes")

        print(f"\nStance distribution:")
        for stance, pct in analysis["stance_percentages"].items():
            count = analysis["stance_distribution"][stance]
            print(f"  {stance:12s}: {count:4d} ({pct})")

        if isinstance(analysis["women_against_count"], int):
            print(f"\nWomen AGAINST suffrage: {analysis['women_against_count']} speeches")

    # Compare if we have both datasets with same speeches
    print("\n" + "="*70)
    print("COMPARISON ANALYSIS")
    print("="*70)

    # Check if we can compare (pilot data overlaps)
    gpt_df = pd.read_parquet(EXPERIMENTS["gpt4o_mini"]["results"])
    claude_df = pd.read_parquet(EXPERIMENTS["claude_sonnet_45"]["results"])

    # Find overlapping speeches
    gpt_ids = set(gpt_df['speech_id'])
    claude_ids = set(claude_df['speech_id'])
    overlap_ids = gpt_ids & claude_ids

    if len(overlap_ids) == 0:
        print("\nNo overlapping speeches - cannot compare directly")
        print(f"GPT-4o-mini: {len(gpt_ids)} speeches")
        print(f"Claude: {len(claude_ids)} speeches")
        return

    print(f"\nOverlapping speeches: {len(overlap_ids)}")

    # Filter to overlap
    gpt_overlap = gpt_df[gpt_df['speech_id'].isin(overlap_ids)]
    claude_overlap = claude_df[claude_df['speech_id'].isin(overlap_ids)]

    # Merge on speech_id
    merged = gpt_overlap.merge(
        claude_overlap,
        on='speech_id',
        suffixes=('_gpt', '_claude')
    )

    print(f"\nStance agreement:")
    agreement = (merged['stance_gpt'] == merged['stance_claude']).sum()
    print(f"  Same stance: {agreement}/{len(merged)} ({100*agreement/len(merged):.1f}%)")

    # Where they disagree
    disagree = merged[merged['stance_gpt'] != merged['stance_claude']]
    if len(disagree) > 0:
        print(f"\nDisagreements: {len(disagree)}")
        print("\nMost common disagreement patterns:")
        patterns = disagree.groupby(['stance_gpt', 'stance_claude']).size().sort_values(ascending=False)
        for (gpt_stance, claude_stance), count in patterns.head(10).items():
            print(f"  GPT: {gpt_stance:12s} -> Claude: {claude_stance:12s} ({count} speeches)")

    # Women AGAINST comparison
    if 'gender_gpt' in merged.columns or 'gender_claude' in merged.columns:
        gender_col = 'gender_gpt' if 'gender_gpt' in merged.columns else 'gender_claude'
        women_against_gpt = merged[(merged[gender_col] == 'F') & (merged['stance_gpt'] == 'against')]
        women_against_claude = merged[(merged[gender_col] == 'F') & (merged['stance_claude'] == 'against')]

        print(f"\nWomen AGAINST suffrage (overlapping speeches only):")
        print(f"  GPT-4o-mini: {len(women_against_gpt)}")
        print(f"  Claude Sonnet 4.5: {len(women_against_claude)}")

        # Show examples where they disagree
        women_disagree = merged[
            (merged[gender_col] == 'F') &
            (merged['stance_gpt'] != merged['stance_claude']) &
            ((merged['stance_gpt'] == 'against') | (merged['stance_claude'] == 'against'))
        ]

        if len(women_disagree) > 0:
            print(f"\nWomen where GPT and Claude disagree on AGAINST classification:")
            for idx, row in women_disagree.iterrows():
                print(f"\n  Speech: {row['speech_id']}")
                speaker = row.get('speaker_gpt', row.get('speaker_claude', 'N/A'))
                year = row.get('year_gpt', row.get('year_claude', 'N/A'))
                print(f"  Speaker: {speaker}, Year: {year}")
                print(f"  GPT stance: {row['stance_gpt']}")
                print(f"  Claude stance: {row['stance_claude']}")

if __name__ == "__main__":
    main()
