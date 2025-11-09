#!/usr/bin/env python3
"""
Generate temporal trend visualizations for manuscript.

Creates Figure 1: Female MP participation over time
- Female MP presence rate in Parliament (from external data)
- Female speaking proportion in Hansard debates

Usage:
    python3 02_temporal_trends.py [--start YEAR] [--end YEAR]

Output:
    manuscript_figures/temporal_*.png - Trend visualizations
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import load_speeches
from utils import setup_plot_style, COLORS, save_figure, MILESTONES


def compute_annual_speaking_proportions(speeches_df):
    """
    Compute annual female speaking proportion.
    Excludes speeches with unknown gender.

    Returns:
        DataFrame with columns: year, total_speeches, female_speeches,
                                 male_speeches, female_proportion
    """
    print("Computing annual speaking proportions...")

    # Filter to only speeches with known gender
    gendered = speeches_df[speeches_df['gender'].notna()].copy()

    # Group by year
    annual = gendered.groupby('year').agg({
        'speech_id': 'count',
        'gender': lambda x: (x == 'f').sum(),  # Count female
    }).rename(columns={
        'speech_id': 'total_speeches',
        'gender': 'female_speeches'
    })

    # Add male speeches
    annual['male_speeches'] = gendered.groupby('year')['gender'].apply(
        lambda x: (x == 'm').sum()
    )

    # Calculate proportion
    annual['female_proportion'] = (
        annual['female_speeches'] /
        (annual['female_speeches'] + annual['male_speeches']) * 100
    )

    annual = annual.reset_index()
    print(f"  Computed {len(annual)} years of data")

    return annual


def plot_temporal_trends_single_axis(annual_data, start_year=1900):
    """
    Plot temporal trends with single y-axis (speaking proportion only).

    Args:
        annual_data: DataFrame with annual statistics
        start_year: Starting year for visualization (default 1900)
    """
    print("Creating single-axis temporal plot...")

    # Filter to start year
    data = annual_data[annual_data['year'] >= start_year].copy()

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot female speaking proportion
    ax.plot(
        data['year'],
        data['female_proportion'],
        color=COLORS['female'],
        linewidth=2,
        label='Female speaking proportion (%)'
    )

    # Add milestone markers
    for key, milestone in MILESTONES.items():
        if milestone['year'] >= start_year and milestone['year'] <= data['year'].max():
            ax.axvline(
                milestone['year'],
                color=COLORS['muted'],
                linestyle='--',
                alpha=0.5,
                linewidth=1
            )
            # Add label at top
            ax.text(
                milestone['year'],
                ax.get_ylim()[1] * 0.95,
                milestone['label'],
                rotation=90,
                verticalalignment='top',
                fontsize=8,
                color=COLORS['text'],
                alpha=0.7
            )

    ax.set_xlabel('Year')
    ax.set_ylabel('Female Speaking Proportion (%)')
    ax.set_title('Female MP Speaking Proportion in Hansard Debates')
    ax.legend(loc='upper left')
    ax.set_xlim(start_year, data['year'].max())
    ax.set_ylim(0, max(data['female_proportion']) * 1.1)

    return fig


def plot_temporal_trends_dual_axis(annual_data, mp_data=None, start_year=1900):
    """
    Plot temporal trends with dual y-axes.

    Args:
        annual_data: DataFrame with annual statistics
        mp_data: Optional DataFrame with MP presence data (year, female_mp_pct)
        start_year: Starting year for visualization
    """
    print("Creating dual-axis temporal plot...")

    # Filter to start year
    data = annual_data[annual_data['year'] >= start_year].copy()

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Left axis: Female speaking proportion
    color = COLORS['female']
    ax1.plot(
        data['year'],
        data['female_proportion'],
        color=color,
        linewidth=2.5,
        label='Speaking proportion'
    )
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Female Speaking Proportion in Debates (%)', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xlim(start_year, data['year'].max())

    # Right axis: Female MP presence (if provided)
    if mp_data is not None:
        ax2 = ax1.twinx()
        mp_subset = mp_data[
            (mp_data['year'] >= start_year) &
            (mp_data['year'] <= data['year'].max())
        ]
        color = COLORS['male']
        ax2.plot(
            mp_subset['year'],
            mp_subset['female_mp_pct'],
            color=color,
            linewidth=2.5,
            linestyle='--',
            label='MP presence'
        )
        ax2.set_ylabel('Female MPs in Parliament (%)', color=color)
        ax2.tick_params(axis='y', labelcolor=color)

    # Add milestone markers
    for key, milestone in MILESTONES.items():
        if milestone['year'] >= start_year and milestone['year'] <= data['year'].max():
            ax1.axvline(
                milestone['year'],
                color=COLORS['muted'],
                linestyle=':',
                alpha=0.4,
                linewidth=1.5
            )

    ax1.set_title('Female Political Participation Over Time')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    if mp_data is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    else:
        ax1.legend(loc='upper left')

    return fig


def plot_absolute_counts(annual_data, start_year=1900):
    """
    Plot absolute speech counts by gender.

    Args:
        annual_data: DataFrame with annual statistics
        start_year: Starting year for visualization
    """
    print("Creating absolute counts plot...")

    data = annual_data[annual_data['year'] >= start_year].copy()

    fig, ax = plt.subplots(figsize=(12, 6))

    # Stacked area plot
    ax.fill_between(
        data['year'],
        0,
        data['male_speeches'],
        color=COLORS['male'],
        alpha=0.7,
        label='Male speeches'
    )
    ax.fill_between(
        data['year'],
        data['male_speeches'],
        data['male_speeches'] + data['female_speeches'],
        color=COLORS['female'],
        alpha=0.7,
        label='Female speeches'
    )

    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Speeches')
    ax.set_title('Hansard Speech Volume by Gender')
    ax.legend(loc='upper left')
    ax.set_xlim(start_year, data['year'].max())

    # Format y-axis with commas
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

    return fig


def main():
    """Generate temporal trend visualizations."""
    parser = argparse.ArgumentParser(description='Generate temporal trend visualizations')
    parser.add_argument('--start', type=int, default=1900, help='Start year (default: 1900)')
    parser.add_argument('--end', type=int, default=2005, help='End year (default: 2005)')
    parser.add_argument('--include-lords', action='store_true', help='Include Lords speeches (default: Commons only)')
    args = parser.parse_args()

    print("=" * 70)
    print("GENERATING TEMPORAL TREND VISUALIZATIONS")
    print("=" * 70)

    # Setup plotting style
    setup_plot_style()

    # Load all speeches (Commons only by default)
    chamber = None if args.include_lords else 'Commons'
    print(f"\nLoading speeches from {args.start} to {args.end} ({'Commons only' if chamber else 'All chambers'})...")
    speeches = load_speeches(year_range=(args.start, args.end), chamber=chamber)
    print(f"  Loaded {len(speeches):,} speeches")

    # Compute annual statistics
    annual_data = compute_annual_speaking_proportions(speeches)

    # Generate visualizations
    print("\nGenerating visualizations...")

    # 1. Single axis: Speaking proportion only
    fig1 = plot_temporal_trends_single_axis(annual_data, start_year=args.start)
    save_figure(fig1, 'temporal_speaking_proportion.png')
    plt.close(fig1)

    # 2. Dual axis: Speaking + MP presence (placeholder for MP data)
    # NOTE: User needs to provide MP presence data if desired
    fig2 = plot_temporal_trends_dual_axis(annual_data, mp_data=None, start_year=args.start)
    save_figure(fig2, 'temporal_dual_axis.png')
    plt.close(fig2)

    # 3. Absolute counts
    fig3 = plot_absolute_counts(annual_data, start_year=args.start)
    save_figure(fig3, 'temporal_absolute_counts.png')
    plt.close(fig3)

    print("\n" + "=" * 70)
    print("TEMPORAL VISUALIZATIONS COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - temporal_speaking_proportion.png")
    print("  - temporal_dual_axis.png")
    print("  - temporal_absolute_counts.png")


if __name__ == '__main__':
    main()
