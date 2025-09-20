#!/usr/bin/env python3
"""
MP Temporal Gender Analysis
============================

Analyzes gender patterns over time using only confirmed MPs matched
against the authoritative gendered house members dataset.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MPTemporalAnalyzer:
    """Analyze temporal gender patterns for confirmed MPs only."""
    
    def __init__(self):
        self.data_path = Path('data')
        self.results_path = Path('analysis/results')
        self.results_path.mkdir(parents=True, exist_ok=True)
        
    def load_mp_data(self):
        """Load MP-only speaker data with authoritative gender assignments."""
        
        # Load MP speakers with gender
        mp_path = self.data_path / 'mp_speakers_gendered.parquet'
        logger.info(f"Loading MP speakers from {mp_path}")
        mp_df = pd.read_parquet(mp_path)
        
        logger.info(f"Loaded {len(mp_df):,} confirmed MPs")
        
        # Gender distribution
        gender_dist = mp_df['gender'].value_counts()
        logger.info("\nGender distribution:")
        for gender, count in gender_dist.items():
            percentage = (count / len(mp_df)) * 100
            logger.info(f"  {gender}: {count:,} ({percentage:.1f}%)")
        
        return mp_df
    
    def calculate_yearly_proportions(self, mp_df):
        """Calculate gender proportions by year."""
        
        # Get year range
        min_year = mp_df['first_year'].min()
        max_year = mp_df['last_year'].max()
        
        yearly_data = []
        
        for year in range(min_year, max_year + 1):
            # Find MPs active in this year
            active_mps = mp_df[
                (mp_df['first_year'] <= year) & 
                (mp_df['last_year'] >= year)
            ].copy()
            
            if len(active_mps) == 0:
                continue
            
            # Calculate speech activity weighting
            active_mps['years_active'] = active_mps['last_year'] - active_mps['first_year'] + 1
            active_mps['speeches_per_year'] = active_mps['total_speeches'] / active_mps['years_active']
            
            # Calculate weighted counts
            total_weighted = active_mps['speeches_per_year'].sum()
            female_weighted = active_mps[active_mps['gender'] == 'F']['speeches_per_year'].sum()
            male_weighted = active_mps[active_mps['gender'] == 'M']['speeches_per_year'].sum()
            
            # Simple counts
            total_mps = len(active_mps)
            female_mps = len(active_mps[active_mps['gender'] == 'F'])
            male_mps = len(active_mps[active_mps['gender'] == 'M'])
            
            female_proportion = (female_mps / total_mps) * 100 if total_mps > 0 else 0
            female_speech_proportion = (female_weighted / total_weighted) * 100 if total_weighted > 0 else 0
            
            yearly_data.append({
                'year': year,
                'total_mps': total_mps,
                'female_mps': female_mps,
                'male_mps': male_mps,
                'female_proportion': round(female_proportion, 2),
                'female_speech_proportion': round(female_speech_proportion, 2),
                'total_speeches_weighted': round(total_weighted),
                'female_speeches_weighted': round(female_weighted)
            })
        
        temporal_df = pd.DataFrame(yearly_data)
        logger.info(f"\nCalculated proportions for {len(temporal_df)} years ({min_year}-{max_year})")
        
        # Log key milestones
        first_female = temporal_df[temporal_df['female_mps'] > 0]
        if len(first_female) > 0:
            first_year = first_female['year'].min()
            logger.info(f"First female MP(s) in dataset: {first_year}")
        
        # Peak representation
        if temporal_df['female_proportion'].max() > 0:
            peak_idx = temporal_df['female_proportion'].idxmax()
            peak_year = temporal_df.loc[peak_idx, 'year']
            peak_prop = temporal_df.loc[peak_idx, 'female_proportion']
            peak_count = temporal_df.loc[peak_idx, 'female_mps']
            logger.info(f"Peak female representation: {peak_prop:.1f}% ({peak_count} MPs) in {peak_year}")
        
        return temporal_df
    
    def create_visualization(self, temporal_df, mp_df):
        """Create comprehensive temporal visualization."""
        
        plt.style.use('default')
        fig = plt.figure(figsize=(16, 12))
        
        # Create subplots
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 2, 1.5], hspace=0.3, wspace=0.25)
        ax1 = fig.add_subplot(gs[0, :])  # Top spanning both columns
        ax2 = fig.add_subplot(gs[1, 0])  # Middle left
        ax3 = fig.add_subplot(gs[1, 1])  # Middle right
        ax4 = fig.add_subplot(gs[2, :])  # Bottom spanning both columns
        
        # Plot 1: Main temporal trend - Female MP proportion
        ax1.plot(temporal_df['year'], temporal_df['female_proportion'], 
                linewidth=2.5, color='#e91e63', marker='o', markersize=3,
                markerfacecolor='white', markeredgecolor='#e91e63', markeredgewidth=1.5,
                label='Female MPs (%)')
        
        # Historical milestones
        milestones = [
            (1918, "Women's partial\nsuffrage (30+)", '#2196f3'),
            (1919, "Nancy Astor\nFirst woman MP", '#00bcd4'),
            (1928, "Equal franchise\n(age 21)", '#ff9800'),
            (1979, "Margaret Thatcher\nbecomes PM", '#4caf50'),
            (1997, "Blair's Labour\n'Blair Babes'", '#9c27b0')
        ]
        
        for year, label, color in milestones:
            if temporal_df['year'].min() <= year <= temporal_df['year'].max():
                ax1.axvline(x=year, color=color, linestyle='--', alpha=0.5, linewidth=1.5)
                ax1.text(year, ax1.get_ylim()[1] * 0.95, label, 
                        rotation=0, fontsize=8, color=color,
                        ha='center', va='top', weight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor=color, alpha=0.8))
        
        ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Female MPs (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Gender Representation in UK Parliament (Confirmed MPs Only)\n' + 
                     'Based on Authoritative House Members Dataset', 
                     fontsize=14, fontweight='bold', pad=15)
        ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax1.set_xlim(temporal_df['year'].min() - 2, temporal_df['year'].max() + 2)
        
        # Plot 2: Speech participation
        ax2.plot(temporal_df['year'], temporal_df['female_speech_proportion'], 
                linewidth=2, color='#9c27b0', marker='s', markersize=2,
                label='Female speech share (%)')
        
        ax2.set_xlabel('Year', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Female Speech Share (%)', fontsize=11, fontweight='bold')
        ax2.set_title('Share of Parliamentary Speech by Gender', 
                     fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Absolute numbers
        ax3.bar(temporal_df['year'], temporal_df['female_mps'], 
               color='#ff4081', alpha=0.7, width=1)
        ax3.set_xlabel('Year', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Number of Female MPs', fontsize=11, fontweight='bold')
        ax3.set_title('Absolute Number of Female MPs', 
                     fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Top female MPs by decade
        decades = [(1920, 1929), (1930, 1939), (1940, 1949), (1950, 1959), 
                  (1960, 1969), (1970, 1979), (1980, 1989), (1990, 1999), (2000, 2005)]
        
        decade_top_women = []
        for start, end in decades:
            decade_mps = mp_df[
                (mp_df['gender'] == 'F') &
                (mp_df['first_year'] <= end) &
                (mp_df['last_year'] >= start)
            ].nlargest(3, 'total_speeches')
            
            if len(decade_mps) > 0:
                decade_top_women.append({
                    'decade': f"{start}s",
                    'top_mp': decade_mps.iloc[0]['speaker_name'] if len(decade_mps) > 0 else '',
                    'speeches': decade_mps.iloc[0]['total_speeches'] if len(decade_mps) > 0 else 0
                })
        
        if decade_top_women:
            decade_df = pd.DataFrame(decade_top_women)
            x_pos = np.arange(len(decade_df))
            ax4.bar(x_pos, decade_df['speeches'], color='#673ab7', alpha=0.7)
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(decade_df['decade'], rotation=45)
            ax4.set_ylabel('Speeches', fontsize=11, fontweight='bold')
            ax4.set_title('Most Active Female MP by Decade (by speech count)', 
                         fontsize=12, fontweight='bold')
            
            # Add MP names on bars
            for i, (_, row) in enumerate(decade_df.iterrows()):
                if row['speeches'] > 0:
                    name = row['top_mp'].replace('Dame ', '').replace('Mrs. ', '').replace('Miss ', '')
                    if len(name) > 15:
                        name = name[:15] + '...'
                    ax4.text(i, row['speeches'] + 20, name, 
                           ha='center', va='bottom', fontsize=8, rotation=90)
        
        # Add summary statistics
        peak_prop = temporal_df['female_proportion'].max()
        peak_year = temporal_df.loc[temporal_df['female_proportion'].idxmax(), 'year']
        final_prop = temporal_df.iloc[-1]['female_proportion']
        final_year = temporal_df.iloc[-1]['year']
        
        stats_text = f"Data Quality: MP-only dataset (no references/non-MPs)\n"
        stats_text += f"Matched MPs: {len(mp_df):,} ({mp_df['gender'].value_counts()['F']} female)\n"
        stats_text += f"Peak: {peak_prop:.1f}% ({peak_year})\n"
        stats_text += f"Final: {final_prop:.1f}% ({final_year})"
        
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
        
        plt.suptitle(f'MP Gender Analysis (Authoritative Dataset)\nGenerated: {datetime.now().strftime("%Y-%m-%d")}',
                    fontsize=15, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        return fig
    
    def save_results(self, temporal_df, fig, mp_df):
        """Save analysis results."""
        
        # Save visualization
        viz_path = self.results_path / 'mp_gender_temporal_authoritative.png'
        fig.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"\nSaved visualization to {viz_path}")
        
        # Save data
        data_path = self.results_path / 'mp_gender_temporal_authoritative.json'
        
        # Calculate key statistics
        female_mps = mp_df[mp_df['gender'] == 'F'].sort_values('total_speeches', ascending=False)
        
        results = {
            'metadata': {
                'analysis_type': 'mp_temporal_gender_analysis_authoritative',
                'data_source': 'mp_speakers_gendered.parquet',
                'gender_source': 'house_members_gendered_updated.parquet',
                'generated_at': datetime.now().isoformat(),
                'years_analyzed': f"{temporal_df['year'].min()}-{temporal_df['year'].max()}"
            },
            'data_quality': {
                'total_speakers_original': 59373,
                'matched_to_mps': len(mp_df),
                'match_rate': f"{len(mp_df)/59373*100:.1f}%",
                'gender_distribution': {
                    'female': int(mp_df['gender'].value_counts().get('F', 0)),
                    'male': int(mp_df['gender'].value_counts().get('M', 0))
                }
            },
            'key_statistics': {
                'first_female_mp_year': int(temporal_df[temporal_df['female_mps'] > 0]['year'].min()),
                'peak_year': int(temporal_df.loc[temporal_df['female_proportion'].idxmax(), 'year']),
                'peak_female_proportion': float(temporal_df['female_proportion'].max()),
                'final_year_proportion': float(temporal_df.iloc[-1]['female_proportion'])
            },
            'top_female_mps': [
                {
                    'name': row['speaker_name'],
                    'speeches': int(row['total_speeches']),
                    'years': f"{row['first_year']}-{row['last_year']}"
                }
                for _, row in female_mps.head(20).iterrows()
            ],
            'temporal_data': temporal_df.to_dict('records')
        }
        
        with open(data_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved data to {data_path}")
        
        return results
    
    def run_analysis(self):
        """Run complete MP temporal gender analysis."""
        
        logger.info("="*60)
        logger.info("MP TEMPORAL GENDER ANALYSIS")
        logger.info("Using Authoritative Gendered House Members Dataset")
        logger.info("="*60)
        
        # Load MP data
        mp_df = self.load_mp_data()
        
        # Calculate yearly proportions
        temporal_df = self.calculate_yearly_proportions(mp_df)
        
        # Create visualization
        fig = self.create_visualization(temporal_df, mp_df)
        
        # Save results
        results = self.save_results(temporal_df, fig, mp_df)
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("ANALYSIS COMPLETE")
        logger.info("="*50)
        logger.info(f"Analyzed {len(mp_df):,} confirmed MPs")
        logger.info(f"Female MPs: {mp_df['gender'].value_counts().get('F', 0):,}")
        logger.info(f"Male MPs: {mp_df['gender'].value_counts().get('M', 0):,}")
        logger.info(f"Years covered: {temporal_df['year'].min()}-{temporal_df['year'].max()}")
        
        # Top female MPs
        female_mps = mp_df[mp_df['gender'] == 'F'].sort_values('total_speeches', ascending=False)
        logger.info("\nTop 5 female MPs by speech count:")
        for _, mp in female_mps.head(5).iterrows():
            logger.info(f"  {mp['speaker_name']:30s} | {mp['total_speeches']:,} speeches")
        
        logger.info("\nVisualization saved to: analysis/results/mp_gender_temporal_authoritative.png")
        logger.info("Data saved to: analysis/results/mp_gender_temporal_authoritative.json")


if __name__ == "__main__":
    analyzer = MPTemporalAnalyzer()
    analyzer.run_analysis()