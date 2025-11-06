#!/usr/bin/env python3
"""
Shared utilities for manuscript visualizations.
Enforces professional styling standards from CLAUDE.md.
"""

import matplotlib.pyplot as plt
from pathlib import Path


# Color scheme options - user can choose
COLOR_SCHEMES = {
    'default': {
        # Default matplotlib colors
        'male': '#1f77b4',      # Matplotlib blue
        'female': '#ff7f0e',    # Matplotlib orange
        'background': '#FFFFFF',
        'grid': '#E5E7EB',
        'text': '#1F2937',
        'muted': '#9CA3AF',
    },
    'blue_pink': {
        # Blue/pink gender colors
        'male': '#3B82C4',
        'female': '#EC4899',
        'background': '#FFFFFF',
        'grid': '#E5E7EB',
        'text': '#1F2937',
        'muted': '#9CA3AF',
    },
    'colorblind': {
        # Colorblind-friendly palette
        'male': '#0173B2',      # Blue
        'female': '#DE8F05',    # Orange
        'background': '#FFFFFF',
        'grid': '#E5E7EB',
        'text': '#1F2937',
        'muted': '#9CA3AF',
    },
    'viridis': {
        # Viridis-inspired
        'male': '#440154',      # Purple
        'female': '#FDE724',    # Yellow
        'background': '#FFFFFF',
        'grid': '#E5E7EB',
        'text': '#1F2937',
        'muted': '#9CA3AF',
    },
    'nature': {
        # Nature colors
        'male': '#2C5F2D',      # Forest green
        'female': '#854442',    # Terracotta
        'background': '#FFFFFF',
        'grid': '#E5E7EB',
        'text': '#1F2937',
        'muted': '#9CA3AF',
    }
}

# Current color scheme (change this after testing)
CURRENT_SCHEME = 'default'
COLORS = COLOR_SCHEMES[CURRENT_SCHEME]


def setup_plot_style():
    """
    Configure matplotlib for professional publication-ready figures.
    Call this at the start of each script.
    """
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica Neue', 'Arial', 'DejaVu Sans']
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['axes.facecolor'] = COLORS['background']
    plt.rcParams['figure.facecolor'] = COLORS['background']
    plt.rcParams['text.color'] = COLORS['text']
    plt.rcParams['axes.labelcolor'] = COLORS['text']
    plt.rcParams['xtick.color'] = COLORS['text']
    plt.rcParams['ytick.color'] = COLORS['text']


def get_output_dir() -> Path:
    """Get manuscript_figures output directory."""
    project_root = Path(__file__).resolve().parents[3]
    output_dir = project_root / 'manuscript_figures'
    output_dir.mkdir(exist_ok=True)
    return output_dir


def save_figure(fig, filename: str, tight: bool = True):
    """
    Save figure with consistent settings.

    Args:
        fig: Matplotlib figure object
        filename: Output filename (e.g., 'temporal_trends.png')
        tight: Use tight_layout
    """
    output_path = get_output_dir() / filename

    if tight:
        fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none'
    )
    print(f"Saved: {output_path}")


# Historical milestones for reference
MILESTONES = {
    'astor': {
        'year': 1919,
        'label': 'Nancy Astor\n(First woman MP)',
        'description': 'Nancy Astor becomes first woman to take seat in Parliament'
    },
    'equal_suffrage': {
        'year': 1928,
        'label': 'Equal Voting Rights',
        'description': 'Equal franchise for women and men'
    },
    'ww2': {
        'year': 1939,
        'end_year': 1945,
        'label': 'World War II',
        'description': 'World War II period'
    },
    'thatcher': {
        'year': 1979,
        'label': 'Thatcher PM',
        'description': 'Margaret Thatcher becomes Prime Minister'
    },
    'blair': {
        'year': 1997,
        'label': "Blair's 101 Women",
        'description': '101 Labour women MPs elected'
    },
    'dataset_end': {
        'year': 2005,
        'label': 'Dataset End',
        'description': 'End of Hansard dataset'
    }
}


if __name__ == '__main__':
    print("Testing manuscript utils...")

    # Test style setup
    setup_plot_style()
    print("Plot style configured")

    # Test output directory
    output_dir = get_output_dir()
    print(f"Output directory: {output_dir}")

    # Show color palette
    print("\nColor palette:")
    for name, color in COLORS.items():
        print(f"  {name}: {color}")

    # Show milestones
    print("\nHistorical milestones:")
    for key, milestone in MILESTONES.items():
        print(f"  {milestone['year']}: {milestone['description']}")

    print("\nUtils test complete!")
