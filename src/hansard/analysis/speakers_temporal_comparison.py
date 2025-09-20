#!/usr/bin/env python3
"""
Speakers vs MPs Temporal Comparison
===================================

Compares female representation in:
1. Actual MP composition (from house_members_gendered.parquet)
2. Speaking participation (from speakers_master.parquet with gender inference)

Creates side-by-side graphs to show if female MPs spoke proportionally to their numbers.

Usage:
    python speakers_temporal_comparison.py

Output:
    - speakers_vs_mps_comparison.png: Side-by-side temporal graphs
    - speakers_temporal_data.json: Speaking participation data
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpeakersTemporalAnalyzer:
    """Analyzer comparing MP representation vs speaking participation"""
    
    def __init__(self):
        self.speakers_path = Path("data/processed_fixed/metadata/speakers_master.parquet")
        self.mps_path = Path("data/house_members_gendered.parquet")
        self.results_path = Path("analysis/results")
        self.results_path.mkdir(parents=True, exist_ok=True)
        
    def identify_speaker_gender(self, df):
        """Identify speaker gender using title-based heuristics"""
        
        # Initialize gender column
        df['gender_inferred'] = 'unknown'
        
        # Female indicators
        female_titles = [
            'mrs', 'miss', 'ms', 'lady', 'dame', 'duchess', 'countess', 'baroness',
            'viscountess', 'marchioness', 'princess'
        ]
        
        # Male indicators  
        male_titles = [
            'mr', 'sir', 'lord', 'duke', 'earl', 'baron', 'viscount', 'marquess',
            'prince', 'count', 'dr', 'prof', 'captain', 'major', 'colonel', 'general'
        ]
        
        # Apply gender identification
        speaker_lower = df['speaker_name'].str.lower().fillna('')
        
        # Check for female titles
        female_mask = False
        for title in female_titles:
            female_mask |= speaker_lower.str.contains(f'\\b{title}\\b', na=False, regex=True)
        df.loc[female_mask, 'gender_inferred'] = 'F'
        
        # Check for male titles (but don't override female)
        male_mask = False
        for title in male_titles:
            male_mask |= speaker_lower.str.contains(f'\\b{title}\\b', na=False, regex=True)
        df.loc[male_mask & (df['gender_inferred'] == 'unknown'), 'gender_inferred'] = 'M'
        
        # Report results
        gender_counts = df['gender_inferred'].value_counts()
        logger.info("Speaker gender identification:")
        for gender, count in gender_counts.items():
            percentage = (count / len(df)) * 100
            logger.info(f"  {gender}: {count:,} ({percentage:.1f}%)")
        
        return df
    
    def load_speakers_data(self):
        """Load and process speakers data"""
        
        if not self.speakers_path.exists():
            raise FileNotFoundError(f"Speakers dataset not found: {self.speakers_path}")
        
        logger.info(f"Loading speakers data from {self.speakers_path}")
        df = pd.read_parquet(self.speakers_path)
        logger.info(f"Loaded {len(df):,} speaker records")
        
        # Identify gender
        df = self.identify_speaker_gender(df)
        
        # Filter to Commons only for fair comparison with MP data
        commons_df = df[df['chamber'] == 'Commons'].copy()
        logger.info(f"Filtered to {len(commons_df):,} Commons speaker records")
        
        return commons_df
    
    def load_mps_data(self):
        """Load MP composition data (reuse from previous analysis)"""
        
        if not self.mps_path.exists():
            raise FileNotFoundError(f"MPs dataset not found: {self.mps_path}")
        
        logger.info(f"Loading MP composition data from {self.mps_path}")
        df = pd.read_parquet(self.mps_path)
        
        # Filter for MPs only
        mp_df = df[df['post_role'] == 'Member of Parliament'].copy()
        logger.info(f"Loaded {len(mp_df):,} MP records")
        
        # Convert dates
        mp_df['start_date'] = pd.to_datetime(mp_df['start_date'], format='mixed', errors='coerce')
        mp_df['end_date'] = pd.to_datetime(mp_df['end_date'], format='mixed', errors='coerce')
        mp_df['start_year'] = mp_df['start_date'].dt.year
        mp_df['end_year'] = mp_df['end_date'].dt.year
        
        return mp_df
    
    def calculate_speakers_proportions(self, speakers_df):
        """Calculate female speaking proportions by year"""
        
        yearly_data = []
        
        for year in sorted(speakers_df['year'].unique()):
            year_speakers = speakers_df[speakers_df['year'] == year]
            
            # Count unique speakers per year
            unique_speakers = year_speakers.drop_duplicates(subset=['speaker_name'])
            
            total_speakers = len(unique_speakers)
            female_speakers = len(unique_speakers[unique_speakers['gender_inferred'] == 'F'])
            male_speakers = len(unique_speakers[unique_speakers['gender_inferred'] == 'M'])
            unknown_speakers = len(unique_speakers[unique_speakers['gender_inferred'] == 'unknown'])
            
            if total_speakers > 0:
                female_proportion = (female_speakers / total_speakers) * 100
                yearly_data.append({
                    'year': int(year),
                    'total_speakers': total_speakers,
                    'female_speakers': female_speakers,
                    'male_speakers': male_speakers,
                    'unknown_speakers': unknown_speakers,
                    'female_proportion': round(female_proportion, 2)
                })
        
        return pd.DataFrame(yearly_data)
    
    def calculate_mps_proportions(self, mp_df):
        """Calculate female MP proportions by year (same logic as before)"""
        
        min_year = mp_df['start_year'].min()
        max_year = mp_df['end_year'].max()
        
        yearly_data = []
        
        for year in range(min_year, max_year + 1):
            active_mps = mp_df[
                (mp_df['start_year'] <= year) & 
                ((mp_df['end_year'] >= year) | mp_df['end_year'].isna())
            ].copy()
            
            if len(active_mps) == 0:
                continue
            
            unique_mps = active_mps.drop_duplicates(subset=['person_id'])
            
            total_mps = len(unique_mps)
            female_mps = len(unique_mps[unique_mps['gender_inferred'] == 'F'])
            male_mps = len(unique_mps[unique_mps['gender_inferred'] == 'M'])
            
            female_proportion = (female_mps / total_mps) * 100 if total_mps > 0 else 0
            
            yearly_data.append({
                'year': year,
                'total_mps': total_mps,
                'female_mps': female_mps,
                'male_mps': male_mps,
                'female_proportion': round(female_proportion, 2)
            })
        
        return pd.DataFrame(yearly_data)
    
    def create_comparison_visualization(self, speakers_df, mps_df):
        """Create side-by-side comparison of speakers vs MPs"""
        
        plt.style.use('default')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Left plot: MP Composition
        ax1.plot(mps_df['year'], mps_df['female_proportion'], 
                linewidth=3, color='#d62728', marker='o', markersize=4,
                markerfacecolor='white', markeredgecolor='#d62728', markeredgewidth=2)
        
        ax1.set_title('Female MP Representation\n(Actual Parliamentary Composition)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Female MPs (%)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(1800, 2010)
        ax1.set_ylim(0, max(mps_df['female_proportion'].max() * 1.1, 15))
        
        # Right plot: Speaking Participation
        ax2.plot(speakers_df['year'], speakers_df['female_proportion'], 
                linewidth=3, color='#1f77b4', marker='s', markersize=4,
                markerfacecolor='white', markeredgecolor='#1f77b4', markeredgewidth=2)
        
        ax2.set_title('Female Speaking Participation\n(Who Actually Spoke in Debates)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax2.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Female Speakers (%)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(1800, 2010)
        ax2.set_ylim(0, max(speakers_df['female_proportion'].max() * 1.1, 15))
        
        # Add milestone markers to both plots
        milestones = [(1918, "1918"), (1928, "1928"), (1979, "1979"), (1997, "1997")]
        
        for year, label in milestones:
            for ax in [ax1, ax2]:
                ax.axvline(x=year, color='gray', linestyle='--', alpha=0.5)
                ax.text(year, ax.get_ylim()[1] * 0.9, label, rotation=45, 
                       fontsize=9, ha='center', va='bottom', alpha=0.7)
        
        # Add summary statistics
        mp_peak = mps_df['female_proportion'].max()
        mp_peak_year = mps_df.loc[mps_df['female_proportion'].idxmax(), 'year']
        
        speaker_peak = speakers_df['female_proportion'].max()
        speaker_peak_year = speakers_df.loc[speakers_df['female_proportion'].idxmax(), 'year']
        
        ax1.text(0.02, 0.98, f"Peak: {mp_peak:.1f}% ({mp_peak_year})", 
                transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        ax2.text(0.02, 0.98, f"Peak: {speaker_peak:.1f}% ({speaker_peak_year})", 
                transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.suptitle('Female Political Participation Comparison (1803-2005)\nMP Representation vs Speaking Participation', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        return fig
    
    def create_overlay_comparison(self, speakers_df, mps_df):
        """Create overlay comparison on single plot"""
        
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(15, 9))
        
        # Plot both lines
        ax.plot(mps_df['year'], mps_df['female_proportion'], 
                linewidth=3, color='#d62728', marker='o', markersize=4,
                label='MP Composition', alpha=0.8)
        
        ax.plot(speakers_df['year'], speakers_df['female_proportion'], 
                linewidth=3, color='#1f77b4', marker='s', markersize=4,
                label='Speaking Participation', alpha=0.8)
        
        ax.set_xlabel('Year', fontsize=14, fontweight='bold')
        ax.set_ylabel('Female Representation (%)', fontsize=14, fontweight='bold')
        ax.set_title('Female Political Participation: MP Composition vs Speaking Participation (1803-2005)', 
                    fontsize=16, fontweight='bold', pad=25)
        
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12, loc='upper left')
        ax.set_xlim(1800, 2010)
        
        # Add milestone markers
        milestones = [
            (1918, "First women MPs\n(1918)"),
            (1928, "Equal voting rights\n(1928)"),
            (1979, "Thatcher becomes PM\n(1979)"),
            (1997, "Blair's 'Year of Woman'\n(1997)")
        ]
        
        for year, label in milestones:
            ax.axvline(x=year, color='gray', linestyle='--', alpha=0.6)
            ax.annotate(label, xy=(year, ax.get_ylim()[1] * 0.7), 
                       fontsize=9, ha='center', rotation=0)
        
        plt.tight_layout()
        
        return fig
    
    def save_results(self, speakers_df, mps_df):
        """Save comparison data"""
        
        # Find overlapping years for direct comparison
        common_years = set(speakers_df['year']) & set(mps_df['year'])
        
        comparison_data = []
        for year in sorted(common_years):
            mp_data = mps_df[mps_df['year'] == year].iloc[0]
            speaker_data = speakers_df[speakers_df['year'] == year]
            
            if len(speaker_data) > 0:
                speaker_data = speaker_data.iloc[0]
                
                # Calculate gap
                participation_gap = speaker_data['female_proportion'] - mp_data['female_proportion']
                
                comparison_data.append({
                    'year': int(year),
                    'mp_female_proportion': float(mp_data['female_proportion']),
                    'speaker_female_proportion': float(speaker_data['female_proportion']),
                    'participation_gap': round(participation_gap, 2),
                    'mp_female_count': int(mp_data['female_mps']),
                    'speaker_female_count': int(speaker_data['female_speakers'])
                })
        
        results = {
            'metadata': {
                'analysis_type': 'speakers_vs_mps_comparison',
                'generated_at': datetime.now().isoformat(),
                'years_compared': len(comparison_data),
                'data_sources': {
                    'mps': 'house_members_gendered.parquet',
                    'speakers': 'speakers_master.parquet with gender inference'
                }
            },
            'summary_statistics': {
                'average_participation_gap': round(np.mean([d['participation_gap'] for d in comparison_data]), 2),
                'max_participation_gap': max([d['participation_gap'] for d in comparison_data]),
                'min_participation_gap': min([d['participation_gap'] for d in comparison_data])
            },
            'comparison_data': comparison_data,
            'speakers_temporal_data': speakers_df.to_dict('records'),
            'mps_temporal_data': mps_df.to_dict('records')
        }
        
        output_path = self.results_path / "speakers_vs_mps_comparison.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved comparison data to {output_path}")
        return results
    
    def run_analysis(self):
        """Run complete comparison analysis"""
        
        logger.info("="*60)
        logger.info("SPEAKERS vs MPs TEMPORAL COMPARISON")
        logger.info("="*60)
        
        # Load data
        speakers_df = self.load_speakers_data()
        mps_df = self.load_mps_data()
        
        # Calculate proportions
        speakers_temporal = self.calculate_speakers_proportions(speakers_df)
        mps_temporal = self.calculate_mps_proportions(mps_df)
        
        logger.info(f"Speakers analysis: {len(speakers_temporal)} years")
        logger.info(f"MPs analysis: {len(mps_temporal)} years")
        
        # Create visualizations
        side_by_side_fig = self.create_comparison_visualization(speakers_temporal, mps_temporal)
        overlay_fig = self.create_overlay_comparison(speakers_temporal, mps_temporal)
        
        # Save visualizations
        side_by_side_fig.savefig(self.results_path / "speakers_vs_mps_side_by_side.png", 
                                dpi=300, bbox_inches='tight', facecolor='white')
        overlay_fig.savefig(self.results_path / "speakers_vs_mps_overlay.png", 
                           dpi=300, bbox_inches='tight', facecolor='white')
        
        # Save data
        results = self.save_results(speakers_temporal, mps_temporal)
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("COMPARISON SUMMARY")
        logger.info("="*50)
        
        summary = results['summary_statistics']
        logger.info(f"Average participation gap: {summary['average_participation_gap']:.2f}%")
        logger.info(f"Max participation gap: {summary['max_participation_gap']:.2f}%")
        logger.info(f"Min participation gap: {summary['min_participation_gap']:.2f}%")
        
        plt.show()
        
        return speakers_temporal, mps_temporal, results


def main():
    """Main function"""
    analyzer = SpeakersTemporalAnalyzer()
    speakers_data, mps_data, results = analyzer.run_analysis()


if __name__ == "__main__":
    main()