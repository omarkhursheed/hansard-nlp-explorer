#!/usr/bin/env python3
"""
Speakers Temporal Gender Analysis (Fixed for Historical Context)
================================================================

Analyzes gender patterns over time in the deduplicated speakers dataset
using title-based gender inference, with proper handling of historical constraints.

Key Historical Context:
- Before 1918: Women could not be MPs (female titles are debate references)
- 1918: Women over 30 gained right to vote and stand for Parliament
- 1919: Nancy Astor became first woman to take her seat
- 1928: Equal franchise (women could vote at 21, same as men)

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


class HistoricalGenderInference:
    """Infer gender from speaker names with historical context."""
    
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
        
        # Year when women could first become MPs
        self.WOMEN_MP_START_YEAR = 1918
        
    def infer_gender_with_context(self, name: str, year: int) -> tuple[str, str]:
        """
        Infer gender from a speaker name with historical context.
        
        Returns: (gender, category)
        - gender: 'M', 'F', or 'U' (unknown/uncertain)
        - category: 'mp', 'reference', or 'unknown'
        """
        if not name or pd.isna(name):
            return 'U', 'unknown'
        
        name = str(name)
        
        # Check for male titles
        for title in self.male_titles:
            if name.startswith(title + ' '):
                return 'M', 'mp'
        
        # Check for female titles
        for title in self.female_titles:
            if name.startswith(title + ' '):
                # Historical constraint: women couldn't be MPs before 1918
                if year < self.WOMEN_MP_START_YEAR:
                    # These are references to people in debates, not MPs
                    return 'F', 'reference'
                else:
                    return 'F', 'mp'
        
        # Special case: "the Lord Chancellor" and similar roles are traditionally male
        name_lower = name.lower()
        if 'chancellor' in name_lower and 'lord' in name_lower:
            return 'M', 'mp'
        
        # If we can't determine with confidence, return unknown
        return 'U', 'unknown'


class HistoricalTemporalAnalyzer:
    """Analyze temporal gender patterns with historical accuracy."""
    
    def __init__(self):
        self.data_path = Path('/Users/omarkhursheed/workplace/hansard-nlp-explorer/src/hansard/data')
        self.results_path = Path('analysis/results')
        self.results_path.mkdir(parents=True, exist_ok=True)
        self.gender_inferrer = HistoricalGenderInference()
        
    def load_and_infer_gender(self):
        """Load deduplicated speakers and infer gender with historical context."""
        
        # Load deduplicated and fixed speakers
        speakers_path = self.data_path / 'speakers_deduplicated_fixed.parquet'
        logger.info(f"Loading deduplicated speakers from {speakers_path}")
        df = pd.read_parquet(speakers_path)
        
        logger.info(f"Loaded {len(df):,} deduplicated speakers")
        
        # Infer gender for each speaker based on their active period
        logger.info("Inferring gender from speaker names with historical context...")
        
        # Use the midpoint of their career for gender inference
        df['mid_year'] = ((df['first_year'] + df['last_year']) / 2).astype(int)
        
        # Apply inference with context
        gender_results = df.apply(
            lambda row: self.gender_inferrer.infer_gender_with_context(
                row['normalized_name'], row['mid_year']
            ), axis=1
        )
        
        df['inferred_gender'] = gender_results.apply(lambda x: x[0])
        df['speaker_category'] = gender_results.apply(lambda x: x[1])
        
        # Statistics on gender inference
        logger.info("\nGender inference results:")
        for category in ['mp', 'reference', 'unknown']:
            cat_df = df[df['speaker_category'] == category]
            if len(cat_df) > 0:
                logger.info(f"\n{category.upper()}:")
                for gender in ['M', 'F', 'U']:
                    count = len(cat_df[cat_df['inferred_gender'] == gender])
                    if count > 0:
                        percentage = (count / len(df)) * 100
                        logger.info(f"  {gender}: {count:,} ({percentage:.1f}%)")
        
        # Log female references before 1918
        pre_1918_female_refs = df[
            (df['speaker_category'] == 'reference') & 
            (df['inferred_gender'] == 'F') &
            (df['first_year'] < 1918)
        ]
        if len(pre_1918_female_refs) > 0:
            logger.info(f"\nFound {len(pre_1918_female_refs)} female-titled references before 1918 (not MPs):")
            for _, speaker in pre_1918_female_refs.head(5).iterrows():
                logger.info(f"  {speaker['normalized_name']:30s} | {speaker['first_year']}-{speaker['last_year']}")
        
        # Filter to only actual MPs with confident gender assignment
        mp_df = df[
            (df['speaker_category'] == 'mp') & 
            (df['inferred_gender'].isin(['M', 'F']))
        ].copy()
        
        logger.info(f"\nUsing {len(mp_df):,} actual MPs with confident gender assignment")
        
        return mp_df, df  # Return both filtered MPs and full dataset for analysis
    
    def calculate_yearly_proportions(self, mp_df):
        """Calculate gender proportions by year for actual MPs only."""
        
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
            
            # Weight by speech count for that year
            active_mps['years_active'] = active_mps['last_year'] - active_mps['first_year'] + 1
            active_mps['speeches_per_year'] = active_mps['total_speeches'] / active_mps['years_active']
            
            # Calculate weighted counts
            total_weighted = active_mps['speeches_per_year'].sum()
            female_weighted = active_mps[active_mps['inferred_gender'] == 'F']['speeches_per_year'].sum()
            male_weighted = active_mps[active_mps['inferred_gender'] == 'M']['speeches_per_year'].sum()
            
            # Also get simple counts
            total_mps = len(active_mps)
            female_mps = len(active_mps[active_mps['inferred_gender'] == 'F'])
            male_mps = len(active_mps[active_mps['inferred_gender'] == 'M'])
            
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
        
        # Log first appearance of female MPs
        first_female = temporal_df[temporal_df['female_mps'] > 0]
        if len(first_female) > 0:
            first_year = first_female['year'].min()
            logger.info(f"First female MP(s) detected in dataset: {first_year}")
        
        return temporal_df
    
    def create_visualization(self, temporal_df, full_df):
        """Create temporal visualization with historical context."""
        
        plt.style.use('default')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Plot 1: Proportion of female MPs (actual MPs only)
        ax1.plot(temporal_df['year'], temporal_df['female_proportion'], 
                linewidth=2.5, color='#e91e63', marker='o', markersize=3,
                markerfacecolor='white', markeredgecolor='#e91e63', markeredgewidth=1.5,
                label='Female MPs (%)')
        
        # Historical milestones
        milestones = [
            (1918, "Women's partial suffrage\n(age 30+)", '#2196f3'),
            (1919, "Nancy Astor\nFirst woman MP", '#00bcd4'),
            (1928, "Women's full suffrage\n(age 21+)", '#ff9800'),
            (1979, "Margaret Thatcher\nbecomes PM", '#4caf50')
        ]
        
        for year, label, color in milestones:
            if temporal_df['year'].min() <= year <= temporal_df['year'].max():
                for ax in [ax1, ax2]:
                    ax.axvline(x=year, color=color, linestyle='--', alpha=0.5, linewidth=1.5)
                    ax.text(year, ax.get_ylim()[1] * 0.95, label, 
                           rotation=0, fontsize=9, color=color,
                           ha='center', va='top', weight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                   edgecolor=color, alpha=0.8))
        
        # Add shaded region for pre-1918 (no female MPs possible)
        if temporal_df['year'].min() < 1918:
            ax1.axvspan(temporal_df['year'].min(), min(1918, temporal_df['year'].max()), 
                       alpha=0.15, color='gray', zorder=0)
            ax1.text(min(1900, (temporal_df['year'].min() + 1918) / 2), 
                    ax1.get_ylim()[1] * 0.5,
                    'Women could not\nbe MPs', 
                    fontsize=10, color='gray', alpha=0.7,
                    ha='center', va='center', weight='bold',
                    rotation=0)
        
        # Formatting for ax1
        ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Female MPs (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Gender Representation in UK Parliament (Actual MPs Only)\n' + 
                     'Proportion of Female MPs Over Time', 
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
        
        # Add shaded region for pre-1918
        if temporal_df['year'].min() < 1918:
            ax2.axvspan(temporal_df['year'].min(), min(1918, temporal_df['year'].max()), 
                       alpha=0.15, color='gray', zorder=0)
        
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
        
        stats_text = f"Peak female MPs: {peak_prop:.1f}% ({peak_year})\n"
        stats_text += f"First female MP in data: {first_female_year}\n"
        stats_text += f"Final year (2005): {temporal_df.iloc[-1]['female_proportion']:.1f}%"
        
        # Count pre-1918 female references
        pre_1918_refs = full_df[
            (full_df['speaker_category'] == 'reference') & 
            (full_df['inferred_gender'] == 'F')
        ]
        if len(pre_1918_refs) > 0:
            stats_text += f"\n\nNote: {len(pre_1918_refs)} female-titled\nreferences (not MPs) excluded"
        
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightpink', alpha=0.7))
        
        # Highlight key periods
        for ax in [ax1, ax2]:
            ax.axvspan(1914, 1918, alpha=0.1, color='red', label='WWI')
            ax.axvspan(1939, 1945, alpha=0.1, color='red', label='WWII')
        
        plt.tight_layout()
        
        return fig
    
    def save_results(self, temporal_df, fig, full_df):
        """Save analysis results with historical context."""
        
        # Save visualization
        viz_path = self.results_path / 'speakers_gender_temporal_fixed.png'
        fig.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"\nSaved visualization to {viz_path}")
        
        # Save data
        data_path = self.results_path / 'speakers_gender_temporal_fixed.json'
        
        # Calculate key statistics
        peak_idx = temporal_df['female_proportion'].idxmax() if temporal_df['female_proportion'].max() > 0 else 0
        first_female = temporal_df[temporal_df['female_proportion'] > 0]
        
        # Count references vs MPs
        references = full_df[full_df['speaker_category'] == 'reference']
        female_refs = references[references['inferred_gender'] == 'F']
        
        results = {
            'metadata': {
                'analysis_type': 'speakers_temporal_gender_analysis_with_historical_context',
                'data_source': 'speakers_deduplicated_fixed.parquet',
                'gender_inference': 'title_based_with_historical_constraints',
                'generated_at': datetime.now().isoformat(),
                'years_analyzed': f"{temporal_df['year'].min()}-{temporal_df['year'].max()}"
            },
            'historical_context': {
                'women_mp_eligibility_start': 1918,
                'first_woman_mp': 'Nancy Astor (1919)',
                'note': 'Female-titled speakers before 1918 are references in debates, not MPs'
            },
            'speaker_categories': {
                'total_speakers': len(full_df),
                'actual_mps': len(full_df[full_df['speaker_category'] == 'mp']),
                'debate_references': len(references),
                'female_references_pre_1918': len(female_refs[female_refs['first_year'] < 1918]),
                'unknown_category': len(full_df[full_df['speaker_category'] == 'unknown'])
            },
            'key_statistics': {
                'first_female_mp_year': int(first_female['year'].min()) if len(first_female) > 0 else None,
                'peak_year': int(temporal_df.loc[peak_idx, 'year']),
                'peak_female_proportion': float(temporal_df.loc[peak_idx, 'female_proportion']),
                'peak_female_count': int(temporal_df.loc[peak_idx, 'female_mps']),
                'final_year_proportion': float(temporal_df.iloc[-1]['female_proportion']),
                'final_year_count': int(temporal_df.iloc[-1]['female_mps'])
            },
            'temporal_data': temporal_df.to_dict('records')
        }
        
        with open(data_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved data to {data_path}")
        
        return results
    
    def run_analysis(self):
        """Run complete temporal gender analysis with historical accuracy."""
        
        logger.info("="*60)
        logger.info("TEMPORAL GENDER ANALYSIS - PARLIAMENTARY SPEAKERS")
        logger.info("With Historical Context (Women MPs from 1918+)")
        logger.info("="*60)
        
        # Load data and infer gender with historical context
        mp_df, full_df = self.load_and_infer_gender()
        
        # Calculate yearly proportions for actual MPs only
        temporal_df = self.calculate_yearly_proportions(mp_df)
        
        # Create visualization
        fig = self.create_visualization(temporal_df, full_df)
        
        # Save results
        results = self.save_results(temporal_df, fig, full_df)
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("ANALYSIS SUMMARY")
        logger.info("="*50)
        
        stats = results['key_statistics']
        cats = results['speaker_categories']
        
        logger.info(f"\nSpeaker Categories:")
        logger.info(f"  Total speakers: {cats['total_speakers']:,}")
        logger.info(f"  Actual MPs: {cats['actual_mps']:,}")
        logger.info(f"  Debate references: {cats['debate_references']:,}")
        logger.info(f"  Female references pre-1918: {cats['female_references_pre_1918']:,}")
        
        if stats['first_female_mp_year']:
            logger.info(f"\nFirst female MP detected: {stats['first_female_mp_year']}")
        logger.info(f"Peak female representation: {stats['peak_female_proportion']:.1f}% in {stats['peak_year']}")
        logger.info(f"Final year (2005): {stats['final_year_proportion']:.1f}%")
        
        # Show some example female MPs identified
        female_mps = mp_df[mp_df['inferred_gender'] == 'F'].sort_values('total_speeches', ascending=False)
        if len(female_mps) > 0:
            logger.info(f"\nTop female MPs identified (by speech count):")
            for _, speaker in female_mps.head(10).iterrows():
                logger.info(f"  {speaker['normalized_name']:30s} | {speaker['total_speeches']:,} speeches | {speaker['first_year']}-{speaker['last_year']}")
        
        plt.show()
        
        return temporal_df, results


def main():
    """Main function."""
    analyzer = HistoricalTemporalAnalyzer()
    temporal_df, results = analyzer.run_analysis()


if __name__ == "__main__":
    main()