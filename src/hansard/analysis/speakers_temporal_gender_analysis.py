#!/usr/bin/env python3
"""
Speakers Temporal Gender Analysis
==================================

Analyzes gender patterns over time in the deduplicated speakers dataset
using title-based gender inference.

Only includes speakers where gender can be confidently inferred from titles.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
import logging
from datetime import datetime
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SpeakerGenderInference:
    """Infer gender from speaker names using titles and patterns."""
    
    def __init__(self):
        # Define clear gender indicators
        self.male_titles = {
            'Mr.', 'Sir', 'Lord', 'Earl', 'Viscount', 'Marquess', 'Duke', 'Baron',
            'Major', 'Captain', 'Colonel', 'General', 'Lieut.', 'Lieutenant',
            'Commander', 'Admiral', 'Reverend', 'Rev.', 'Father'
        }
        
        self.female_titles = {
            'Mrs.', 'Miss', 'Ms.', 'Lady', 'Baroness', 'Countess', 'Duchess',
            'Dame', 'Sister'
        }
        
        # Some names have explicit gender indicators
        self.female_first_names = {
            'diane', 'margaret', 'barbara', 'betty', 'clare', 'claire', 'harriet',
            'virginia', 'edith', 'eleanor', 'elizabeth', 'ellen', 'emma', 'florence',
            'gwyneth', 'helen', 'irene', 'jacqui', 'janet', 'jean', 'jenny', 'joan',
            'judith', 'julia', 'julie', 'kate', 'laura', 'linda', 'lorna', 'lynn',
            'maria', 'marion', 'mary', 'maureen', 'nancy', 'patricia', 'rachel',
            'rosie', 'sally', 'sarah', 'shirley', 'susan', 'tessa', 'valerie'
        }
        
    def infer_gender(self, name: str) -> str:
        """
        Infer gender from a speaker name.
        Returns 'M', 'F', or 'U' (unknown/uncertain)
        """
        if not name or pd.isna(name):
            return 'U'
        
        name = str(name)
        
        # Check for explicit titles
        for title in self.male_titles:
            if name.startswith(title + ' '):
                return 'M'
        
        for title in self.female_titles:
            if name.startswith(title + ' '):
                return 'F'
        
        # Check for known female first names (with confidence)
        name_lower = name.lower()
        for female_name in self.female_first_names:
            # Check if the name contains this female name as a complete word
            if re.search(r'\b' + female_name + r'\b', name_lower):
                # Double-check it's not preceded by Mr.
                if not any(title in name for title in ['Mr.', 'Sir', 'Lord']):
                    return 'F'
        
        # Special case: "the Lord Chancellor" and similar roles are traditionally male
        if 'chancellor' in name_lower and 'lord' in name_lower:
            return 'M'
        
        # If we can't determine with confidence, return unknown
        return 'U'


class TemporalGenderAnalyzer:
    """Analyze temporal gender patterns in parliamentary speakers."""
    
    def __init__(self):
        self.data_path = Path('/Users/omarkhursheed/workplace/hansard-nlp-explorer/src/hansard/data')
        self.results_path = Path('analysis/results')
        self.results_path.mkdir(parents=True, exist_ok=True)
        self.gender_inferrer = SpeakerGenderInference()
        
    def load_and_infer_gender(self):
        """Load deduplicated speakers and infer gender."""
        
        # Load deduplicated and fixed speakers (with realistic career spans)
        speakers_path = self.data_path / 'speakers_deduplicated_fixed.parquet'
        logger.info(f"Loading deduplicated speakers from {speakers_path}")
        df = pd.read_parquet(speakers_path)
        
        logger.info(f"Loaded {len(df):,} deduplicated speakers")
        
        # Infer gender for each speaker
        logger.info("Inferring gender from speaker names...")
        df['inferred_gender'] = df['normalized_name'].apply(self.gender_inferrer.infer_gender)
        
        # Statistics on gender inference
        gender_counts = df['inferred_gender'].value_counts()
        logger.info("\nGender inference results:")
        for gender, count in gender_counts.items():
            percentage = (count / len(df)) * 100
            logger.info(f"  {gender}: {count:,} ({percentage:.1f}%)")
        
        # Filter to only confident gender assignments
        confident_df = df[df['inferred_gender'].isin(['M', 'F'])].copy()
        logger.info(f"\nUsing {len(confident_df):,} speakers with confident gender assignment")
        
        return confident_df
    
    def calculate_yearly_proportions(self, df):
        """Calculate gender proportions by year."""
        
        # Get year range
        min_year = df['first_year'].min()
        max_year = df['last_year'].max()
        
        yearly_data = []
        
        for year in range(min_year, max_year + 1):
            # Find speakers active in this year
            active_speakers = df[
                (df['first_year'] <= year) & 
                (df['last_year'] >= year)
            ].copy()
            
            if len(active_speakers) == 0:
                continue
            
            # Weight by speech count for that year
            # Approximate: distribute speeches evenly across active years
            active_speakers['years_active'] = active_speakers['last_year'] - active_speakers['first_year'] + 1
            active_speakers['speeches_per_year'] = active_speakers['total_speeches'] / active_speakers['years_active']
            
            # Calculate weighted counts
            total_weighted = active_speakers['speeches_per_year'].sum()
            female_weighted = active_speakers[active_speakers['inferred_gender'] == 'F']['speeches_per_year'].sum()
            male_weighted = active_speakers[active_speakers['inferred_gender'] == 'M']['speeches_per_year'].sum()
            
            # Also get simple counts
            total_speakers = len(active_speakers)
            female_speakers = len(active_speakers[active_speakers['inferred_gender'] == 'F'])
            male_speakers = len(active_speakers[active_speakers['inferred_gender'] == 'M'])
            
            female_proportion = (female_speakers / total_speakers) * 100 if total_speakers > 0 else 0
            female_speech_proportion = (female_weighted / total_weighted) * 100 if total_weighted > 0 else 0
            
            yearly_data.append({
                'year': year,
                'total_speakers': total_speakers,
                'female_speakers': female_speakers,
                'male_speakers': male_speakers,
                'female_proportion': round(female_proportion, 2),
                'female_speech_proportion': round(female_speech_proportion, 2),
                'total_speeches_weighted': round(total_weighted),
                'female_speeches_weighted': round(female_weighted)
            })
        
        temporal_df = pd.DataFrame(yearly_data)
        logger.info(f"\nCalculated proportions for {len(temporal_df)} years ({min_year}-{max_year})")
        
        return temporal_df
    
    def create_visualization(self, temporal_df):
        """Create temporal visualization of gender patterns."""
        
        plt.style.use('default')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Plot 1: Proportion of female speakers
        ax1.plot(temporal_df['year'], temporal_df['female_proportion'], 
                linewidth=2.5, color='#e91e63', marker='o', markersize=3,
                markerfacecolor='white', markeredgecolor='#e91e63', markeredgewidth=1.5,
                label='Female speakers (%)')
        
        # Historical milestones
        milestones = [
            (1918, "Women's partial suffrage\n(age 30+)", '#2196f3'),
            (1928, "Women's full suffrage\n(age 21+)", '#ff9800'),
            (1979, "Margaret Thatcher\nbecomes PM", '#4caf50')
        ]
        
        for year, label, color in milestones:
            if year in temporal_df['year'].values:
                for ax in [ax1, ax2]:
                    ax.axvline(x=year, color=color, linestyle='--', alpha=0.5, linewidth=1.5)
                    ax.text(year, ax.get_ylim()[1] * 0.95, label, 
                           rotation=0, fontsize=9, color=color,
                           ha='center', va='top', weight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                   edgecolor=color, alpha=0.8))
        
        # Formatting for ax1
        ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Female Speakers (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Gender Representation in UK Parliament Speakers (Title-Based Inference)\n' + 
                     'Proportion of Female Speakers Over Time', 
                     fontsize=14, fontweight='bold', pad=15)
        ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax1.set_xlim(temporal_df['year'].min() - 2, temporal_df['year'].max() + 2)
        
        # Set meaningful y-axis range
        max_prop = temporal_df['female_proportion'].max()
        ax1.set_ylim(0, max(max_prop * 1.3, 5))
        
        # Plot 2: Speech participation (weighted by activity)
        ax2.plot(temporal_df['year'], temporal_df['female_speech_proportion'], 
                linewidth=2.5, color='#9c27b0', marker='s', markersize=3,
                markerfacecolor='white', markeredgecolor='#9c27b0', markeredgewidth=1.5,
                label='Female speech share (%)')
        
        # Formatting for ax2
        ax2.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Female Speech Share (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Share of Parliamentary Speech by Gender\n' + 
                     '(Weighted by Number of Speeches)', 
                     fontsize=14, fontweight='bold', pad=15)
        ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax2.set_xlim(temporal_df['year'].min() - 2, temporal_df['year'].max() + 2)
        
        max_speech_prop = temporal_df['female_speech_proportion'].max()
        ax2.set_ylim(0, max(max_speech_prop * 1.3, 5))
        
        # Add summary statistics
        peak_prop = temporal_df['female_proportion'].max()
        peak_year = temporal_df.loc[temporal_df['female_proportion'].idxmax(), 'year']
        first_female_year = temporal_df[temporal_df['female_proportion'] > 0]['year'].min() if any(temporal_df['female_proportion'] > 0) else 'None'
        
        stats_text = f"Peak female speakers: {peak_prop:.1f}% ({peak_year})\n"
        stats_text += f"First female speaker: {first_female_year}\n"
        stats_text += f"Final year (2005): {temporal_df.iloc[-1]['female_proportion']:.1f}%"
        
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightpink', alpha=0.7))
        
        # Highlight key periods
        for ax in [ax1, ax2]:
            ax.axvspan(1914, 1918, alpha=0.1, color='red', label='WWI')
            ax.axvspan(1939, 1945, alpha=0.1, color='red', label='WWII')
        
        plt.tight_layout()
        
        return fig
    
    def save_results(self, temporal_df, fig):
        """Save analysis results."""
        
        # Save visualization
        viz_path = self.results_path / 'speakers_gender_temporal.png'
        fig.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"\nSaved visualization to {viz_path}")
        
        # Save data
        data_path = self.results_path / 'speakers_gender_temporal.json'
        
        # Calculate key statistics
        peak_idx = temporal_df['female_proportion'].idxmax() if temporal_df['female_proportion'].max() > 0 else 0
        first_female = temporal_df[temporal_df['female_proportion'] > 0]
        
        results = {
            'metadata': {
                'analysis_type': 'speakers_temporal_gender_analysis',
                'data_source': 'speakers_deduplicated.parquet',
                'gender_inference': 'title_based',
                'generated_at': datetime.now().isoformat(),
                'years_analyzed': f"{temporal_df['year'].min()}-{temporal_df['year'].max()}"
            },
            'key_statistics': {
                'first_female_year': int(first_female['year'].min()) if len(first_female) > 0 else None,
                'peak_year': int(temporal_df.loc[peak_idx, 'year']),
                'peak_female_proportion': float(temporal_df.loc[peak_idx, 'female_proportion']),
                'peak_female_count': int(temporal_df.loc[peak_idx, 'female_speakers']),
                'final_year_proportion': float(temporal_df.iloc[-1]['female_proportion']),
                'final_year_count': int(temporal_df.iloc[-1]['female_speakers'])
            },
            'temporal_data': temporal_df.to_dict('records')
        }
        
        with open(data_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved data to {data_path}")
        
        return results
    
    def run_analysis(self):
        """Run complete temporal gender analysis."""
        
        logger.info("="*60)
        logger.info("TEMPORAL GENDER ANALYSIS - PARLIAMENTARY SPEAKERS")
        logger.info("Using Title-Based Gender Inference")
        logger.info("="*60)
        
        # Load data and infer gender
        df = self.load_and_infer_gender()
        
        # Calculate yearly proportions
        temporal_df = self.calculate_yearly_proportions(df)
        
        # Create visualization
        fig = self.create_visualization(temporal_df)
        
        # Save results
        results = self.save_results(temporal_df, fig)
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("ANALYSIS SUMMARY")
        logger.info("="*50)
        
        stats = results['key_statistics']
        if stats['first_female_year']:
            logger.info(f"First female speaker detected: {stats['first_female_year']}")
        logger.info(f"Peak female representation: {stats['peak_female_proportion']:.1f}% in {stats['peak_year']}")
        logger.info(f"Final year (2005): {stats['final_year_proportion']:.1f}%")
        
        # Show some example female speakers identified
        female_speakers = df[df['inferred_gender'] == 'F'].sort_values('total_speeches', ascending=False)
        logger.info(f"\nTop female speakers identified (by speech count):")
        for _, speaker in female_speakers.head(10).iterrows():
            logger.info(f"  {speaker['normalized_name']:30s} | {speaker['total_speeches']:,} speeches | {speaker['first_year']}-{speaker['last_year']}")
        
        plt.show()
        
        return temporal_df, results


def main():
    """Main function."""
    analyzer = TemporalGenderAnalyzer()
    temporal_df, results = analyzer.run_analysis()


if __name__ == "__main__":
    main()