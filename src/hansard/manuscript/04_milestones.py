#!/usr/bin/env python3
"""
Generate milestone-focused visualizations for manuscript.

Analyzes female MP participation around major historical events:
1. 1919: Nancy Astor (first woman MP seated)
2. 1928: Equal voting rights
3. 1939-1945: WW2 period
4. 1979: Thatcher becomes PM
5. 1997: Blair's 101 women MPs

Usage:
    python3 04_milestones.py [--window YEARS]

Output:
    manuscript_figures/milestones_*.png - Event-focused visualizations
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from data_loader import load_speeches
from utils import setup_plot_style, COLORS, save_figure, MILESTONES


def compute_milestone_statistics(speeches_df, milestone_years, window=5):
    """
    Compute before/after statistics for each milestone.

    Args:
        speeches_df: DataFrame with all speeches
        milestone_years: Dict of {name: year}
        window: Years before/after to analyze

    Returns:
        DataFrame with statistics for each milestone
    """
    print(f"Computing milestone statistics (window = {window} years)...")

    results = []

    for key, milestone in milestone_years.items():
        year = milestone['year']
        print(f"  Analyzing {key} ({year})...")

        # Before period
        before = speeches_df[
            (speeches_df['year'] >= year - window) &
            (speeches_df['year'] < year) &
            (speeches_df['gender'].notna())
        ]

        # After period
        after = speeches_df[
            (speeches_df['year'] > year) &
            (speeches_df['year'] <= year + window) &
            (speeches_df['gender'].notna())
        ]

        # Compute statistics
        before_total = len(before)
        before_female = (before['gender'] == 'f').sum()
        before_male = (before['gender'] == 'm').sum()
        before_pct = before_female / (before_female + before_male) * 100 if (before_female + before_male) > 0 else 0

        after_total = len(after)
        after_female = (after['gender'] == 'f').sum()
        after_male = (after['gender'] == 'm').sum()
        after_pct = after_female / (after_female + after_male) * 100 if (after_female + after_male) > 0 else 0

        results.append({
            'milestone': milestone.get('label', key),
            'year': year,
            'before_total': before_total,
            'before_female': before_female,
            'before_male': before_male,
            'before_pct': before_pct,
            'after_total': after_total,
            'after_female': after_female,
            'after_male': after_male,
            'after_pct': after_pct,
            'change_pct': after_pct - before_pct
        })

    return pd.DataFrame(results)


def plot_milestone_changes(milestone_stats):
    """
    Plot before/after female speaking proportion for each milestone.

    Args:
        milestone_stats: DataFrame with milestone statistics
    """
    print("Creating milestone changes plot...")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create grouped bar chart
    x = np.arange(len(milestone_stats))
    width = 0.35

    bars1 = ax.bar(
        x - width/2,
        milestone_stats['before_pct'],
        width,
        label='Before',
        color=COLORS['male'],
        alpha=0.7
    )

    bars2 = ax.bar(
        x + width/2,
        milestone_stats['after_pct'],
        width,
        label='After',
        color=COLORS['female'],
        alpha=0.7
    )

    # Customize
    ax.set_xlabel('Historical Milestone')
    ax.set_ylabel('Female Speaking Proportion (%)')
    ax.set_title('Female MP Speaking Proportion Before/After Major Events')
    ax.set_xticks(x)
    ax.set_xticklabels(milestone_stats['milestone'], rotation=45, ha='right')
    ax.legend()

    # Add percentage change annotations
    for i, row in milestone_stats.iterrows():
        if row['change_pct'] != 0:
            y_pos = max(row['before_pct'], row['after_pct']) + 0.5
            ax.text(
                i,
                y_pos,
                f"+{row['change_pct']:.1f}%" if row['change_pct'] > 0 else f"{row['change_pct']:.1f}%",
                ha='center',
                va='bottom',
                fontsize=9,
                color=COLORS['text']
            )

    return fig


def plot_milestone_timeseries(speeches_df, milestone_years, window=10):
    """
    Plot continuous timeseries with milestone markers.

    Args:
        speeches_df: DataFrame with all speeches
        milestone_years: Dict of milestone information
        window: Years before/after first/last milestone to show
    """
    print("Creating milestone timeseries plot...")

    # Compute annual speaking proportions
    gendered = speeches_df[speeches_df['gender'].notna()].copy()

    annual = gendered.groupby('year').apply(
        lambda x: pd.Series({
            'total': len(x),
            'female': (x['gender'] == 'f').sum(),
            'male': (x['gender'] == 'm').sum(),
            'female_pct': (x['gender'] == 'f').sum() / len(x) * 100
        })
    ).reset_index()

    # Filter to range around milestones
    milestone_year_list = [m['year'] for m in milestone_years.values()]
    min_year = min(milestone_year_list) - window
    max_year = max(milestone_year_list) + window
    annual = annual[(annual['year'] >= min_year) & (annual['year'] <= max_year)]

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot trend line
    ax.plot(
        annual['year'],
        annual['female_pct'],
        color=COLORS['female'],
        linewidth=2.5,
        label='Female speaking proportion'
    )

    # Add milestone markers
    for key, milestone in milestone_years.items():
        year = milestone['year']
        if year >= min_year and year <= max_year:
            ax.axvline(
                year,
                color=COLORS['muted'],
                linestyle='--',
                alpha=0.6,
                linewidth=1.5
            )

            # Add label
            y_pos = ax.get_ylim()[1] * 0.9
            ax.text(
                year,
                y_pos,
                f"{year}\n{milestone['label']}",
                rotation=0,
                ha='center',
                va='top',
                fontsize=9,
                color=COLORS['text'],
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor='white',
                    edgecolor=COLORS['muted'],
                    alpha=0.8
                )
            )

    ax.set_xlabel('Year')
    ax.set_ylabel('Female Speaking Proportion (%)')
    ax.set_title('Female MP Participation Around Historical Milestones')
    ax.legend(loc='upper left')

    return fig


def plot_milestone_small_multiples(speeches_df, milestone_years, window=5):
    """
    Create small multiples showing trend around each milestone.

    Args:
        speeches_df: DataFrame with all speeches
        milestone_years: Dict of milestone information
        window: Years before/after to show for each milestone
    """
    print("Creating small multiples plot...")

    n_milestones = len(milestone_years)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for idx, (key, milestone) in enumerate(milestone_years.items()):
        if idx >= len(axes):
            break

        ax = axes[idx]
        year = milestone['year']

        # Get data for this window
        window_data = speeches_df[
            (speeches_df['year'] >= year - window) &
            (speeches_df['year'] <= year + window) &
            (speeches_df['gender'].notna())
        ]

        # Compute annual proportions
        annual = window_data.groupby('year').apply(
            lambda x: pd.Series({
                'female_pct': (x['gender'] == 'f').sum() / len(x) * 100 if len(x) > 0 else 0
            })
        ).reset_index()

        # Plot
        ax.plot(
            annual['year'],
            annual['female_pct'],
            color=COLORS['female'],
            linewidth=2,
            marker='o',
            markersize=4
        )

        # Add milestone marker
        ax.axvline(
            year,
            color=COLORS['muted'],
            linestyle='--',
            alpha=0.6
        )

        # Customize
        ax.set_title(f"{year}: {milestone['label']}", fontsize=10)
        ax.set_xlabel('Year', fontsize=9)
        ax.set_ylabel('Female %', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_xlim(year - window, year + window)

    # Remove extra subplot if odd number
    if n_milestones < len(axes):
        fig.delaxes(axes[-1])

    fig.suptitle('Female Speaking Proportion Around Historical Events', fontsize=14)

    return fig


def main():
    """Generate milestone-focused visualizations."""
    parser = argparse.ArgumentParser(description='Generate milestone visualizations')
    parser.add_argument(
        '--window',
        type=int,
        default=5,
        help='Years before/after milestone to analyze (default: 5)'
    )
    parser.add_argument(
        '--include-lords',
        action='store_true',
        help='Include Lords speeches (default: Commons only)'
    )
    args = parser.parse_args()

    print("=" * 70)
    print("GENERATING MILESTONE VISUALIZATIONS")
    print("=" * 70)

    # Setup plotting style
    setup_plot_style()

    # Load speeches (Commons only by default)
    chamber = None if args.include_lords else 'Commons'
    print(f"\nLoading speeches ({'Commons only' if chamber else 'All chambers'})...")
    speeches = load_speeches(chamber=chamber)
    print(f"  Loaded {len(speeches):,} speeches")

    # Filter to relevant milestones (skip dataset_end for visualization)
    milestone_years = {
        k: v for k, v in MILESTONES.items()
        if k != 'dataset_end'
    }

    # Compute statistics
    milestone_stats = compute_milestone_statistics(speeches, milestone_years, args.window)

    # Save statistics
    stats_path = save_figure.__globals__['get_output_dir']() / 'milestone_statistics.csv'
    milestone_stats.to_csv(stats_path, index=False)
    print(f"\nSaved statistics to: {stats_path}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # 1. Before/after comparison
    fig1 = plot_milestone_changes(milestone_stats)
    save_figure(fig1, 'milestones_before_after.png')
    plt.close(fig1)

    # 2. Timeseries with milestones
    fig2 = plot_milestone_timeseries(speeches, milestone_years, window=10)
    save_figure(fig2, 'milestones_timeseries.png')
    plt.close(fig2)

    # 3. Small multiples
    fig3 = plot_milestone_small_multiples(speeches, milestone_years, window=args.window)
    save_figure(fig3, 'milestones_small_multiples.png')
    plt.close(fig3)

    print("\n" + "=" * 70)
    print("MILESTONE VISUALIZATIONS COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - milestones_before_after.png")
    print("  - milestones_timeseries.png")
    print("  - milestones_small_multiples.png")
    print("  - milestone_statistics.csv")


if __name__ == '__main__':
    main()
