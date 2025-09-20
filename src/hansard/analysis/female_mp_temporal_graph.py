#!/usr/bin/env python3
"""
Female MP Temporal Graph
========================

Creates a clean temporal graph showing the proportion of female MPs over time
using the authoritative house_members_gendered.parquet dataset.

Usage:
    python female_mp_temporal_graph.py

Output:
    - female_mp_proportion_timeline.png: Clean temporal visualization
    - female_mp_temporal_data.json: Data for further analysis
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

class FemaleMPTemporalAnalyzer:
    """Analyzer for temporal trends in female MP representation using authoritative data"""
    
    def __init__(self):
        self.data_path = Path("data/house_members_gendered.parquet")
        self.results_path = Path("analysis/results")
        self.results_path.mkdir(parents=True, exist_ok=True)
        
    def load_mp_data(self):
        """Load and filter MP data from the gendered house members dataset"""
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Gendered house members dataset not found: {self.data_path}")
        
        logger.info(f"Loading gendered house members data from {self.data_path}")
        df = pd.read_parquet(self.data_path)
        
        # Filter for MPs only (exclude Peers, MSPs, etc.)
        mp_df = df[df['post_role'] == 'Member of Parliament'].copy()
        logger.info(f"Filtered to {len(mp_df):,} MP records from {len(df):,} total records")
        
        # Convert date columns with flexible parsing
        mp_df['start_date'] = pd.to_datetime(mp_df['start_date'], format='mixed', errors='coerce')
        mp_df['end_date'] = pd.to_datetime(mp_df['end_date'], format='mixed', errors='coerce')
        
        # Extract years
        mp_df['start_year'] = mp_df['start_date'].dt.year
        mp_df['end_year'] = mp_df['end_date'].dt.year
        
        logger.info(f"Date range: {mp_df['start_year'].min()} to {mp_df['end_year'].max()}")
        
        # Gender distribution
        gender_counts = mp_df['gender_inferred'].value_counts()
        logger.info("Gender distribution in MP data:")
        for gender, count in gender_counts.items():
            percentage = (count / len(mp_df)) * 100
            logger.info(f"  {gender}: {count:,} ({percentage:.1f}%)")
        
        return mp_df
    
    def calculate_yearly_proportions(self, mp_df):
        """Calculate female MP proportions by year using active memberships"""
        
        # Get the full year range
        min_year = mp_df['start_year'].min()
        max_year = mp_df['end_year'].max()
        
        yearly_data = []
        
        for year in range(min_year, max_year + 1):
            # Find MPs who were active during this year
            active_mps = mp_df[
                (mp_df['start_year'] <= year) & 
                ((mp_df['end_year'] >= year) | mp_df['end_year'].isna())
            ].copy()
            
            if len(active_mps) == 0:
                continue
            
            # Count unique people (avoid double-counting if someone had multiple constituencies)
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
        
        temporal_df = pd.DataFrame(yearly_data)
        logger.info(f"Calculated proportions for {len(temporal_df)} years ({min_year}-{max_year})")
        
        return temporal_df
    
    def create_visualization(self, temporal_df):
        """Create a clean temporal visualization"""
        
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(15, 9))
        
        # Main line plot with clean styling
        ax.plot(temporal_df['year'], temporal_df['female_proportion'], 
                linewidth=3, color='#d62728', marker='o', markersize=4, 
                markerfacecolor='white', markeredgecolor='#d62728', markeredgewidth=2)
        
        # Historical milestone markers
        milestones = [
            (1918, "First women MPs elected\n(Representation of People Act)", '#1f77b4', 'bottom'),
            (1928, "Equal voting rights\n(Equal Franchise Act)", '#ff7f0e', 'top'),
            (1979, "Margaret Thatcher\nbecomes PM", '#2ca02c', 'bottom'),
            (1997, "Blair's 'Year of the Woman'\n120 women MPs", '#9467bd', 'top')
        ]
        
        for year, label, color, va in milestones:
            if year in temporal_df['year'].values:
                proportion = temporal_df[temporal_df['year'] == year]['female_proportion'].iloc[0]
                ax.axvline(x=year, color=color, linestyle='--', alpha=0.6, linewidth=2)
                
                # Position annotation above or below line
                y_offset = 15 if va == 'top' else -25
                y_pos = proportion + (y_offset * 0.1)  # Scale by proportion range
                
                ax.annotate(label, xy=(year, proportion), xytext=(0, y_offset),
                           textcoords='offset points', fontsize=10, color=color,
                           ha='center', va=va, weight='bold',
                           bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                                   edgecolor=color, alpha=0.9))
        
        # Formatting
        ax.set_xlabel('Year', fontsize=14, fontweight='bold')
        ax.set_ylabel('Female MPs (%)', fontsize=14, fontweight='bold')
        ax.set_title('Female MP Representation in UK Parliament (1803-2005)\n' + 
                    'Based on Authoritative Political Database', 
                    fontsize=16, fontweight='bold', pad=25)
        
        # Grid and styling
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_xlim(temporal_df['year'].min() - 2, temporal_df['year'].max() + 2)
        
        # Set y-axis to show meaningful range
        max_prop = temporal_df['female_proportion'].max()
        ax.set_ylim(0, max(max_prop * 1.2, 15))
        
        # Summary statistics box
        peak_prop = temporal_df['female_proportion'].max()
        peak_year = temporal_df.loc[temporal_df['female_proportion'].idxmax(), 'year']
        
        # Find first female MP year
        first_female_year = temporal_df[temporal_df['female_proportion'] > 0]['year'].min()
        
        stats_text = f"First female MPs: {first_female_year}\n"
        stats_text += f"Peak representation: {peak_prop:.1f}% ({peak_year})\n"
        stats_text += f"Total years analyzed: {len(temporal_df)}\n"
        stats_text += f"Final year (2005): {temporal_df.iloc[-1]['female_proportion']:.1f}%"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        # Highlight key periods with background shading
        ax.axvspan(1914, 1918, alpha=0.1, color='red', label='WWI')
        ax.axvspan(1939, 1945, alpha=0.1, color='red', label='WWII')
        ax.axvspan(1979, 1990, alpha=0.1, color='green', label='Thatcher Era')
        
        plt.tight_layout()
        
        return fig, ax
    
    def save_visualization(self, fig, filename="female_mp_proportion_timeline.png"):
        """Save the visualization"""
        output_path = self.results_path / filename
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved visualization to {output_path}")
        
    def save_data(self, temporal_df, filename="female_mp_temporal_data.json"):
        """Save temporal data to JSON"""
        
        # Calculate key statistics
        peak_idx = temporal_df['female_proportion'].idxmax()
        first_female_idx = temporal_df[temporal_df['female_proportion'] > 0].index[0]
        
        results = {
            'metadata': {
                'analysis_type': 'female_mp_temporal_analysis',
                'data_source': 'house_members_gendered.parquet',
                'generated_at': datetime.now().isoformat(),
                'years_analyzed': f"{temporal_df['year'].min()}-{temporal_df['year'].max()}",
                'total_years': len(temporal_df)
            },
            'key_statistics': {
                'first_female_mp_year': int(temporal_df.loc[first_female_idx, 'year']),
                'first_female_count': int(temporal_df.loc[first_female_idx, 'female_mps']),
                'peak_year': int(temporal_df.loc[peak_idx, 'year']),
                'peak_proportion': float(temporal_df.loc[peak_idx, 'female_proportion']),
                'peak_count': int(temporal_df.loc[peak_idx, 'female_mps']),
                'final_year_proportion': float(temporal_df.iloc[-1]['female_proportion']),
                'final_year_count': int(temporal_df.iloc[-1]['female_mps'])
            },
            'historical_milestones': {
                '1918': temporal_df[temporal_df['year'] == 1918]['female_proportion'].iloc[0] if 1918 in temporal_df['year'].values else None,
                '1928': temporal_df[temporal_df['year'] == 1928]['female_proportion'].iloc[0] if 1928 in temporal_df['year'].values else None,
                '1979': temporal_df[temporal_df['year'] == 1979]['female_proportion'].iloc[0] if 1979 in temporal_df['year'].values else None,
                '1997': temporal_df[temporal_df['year'] == 1997]['female_proportion'].iloc[0] if 1997 in temporal_df['year'].values else None
            },
            'temporal_data': temporal_df.to_dict('records')
        }
        
        output_path = self.results_path / filename
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved temporal data to {output_path}")
        return results
    
    def run_analysis(self):
        """Run complete female MP temporal analysis"""
        
        logger.info("="*60)
        logger.info("FEMALE MP TEMPORAL ANALYSIS")
        logger.info("Using Authoritative Gendered House Members Dataset")
        logger.info("="*60)
        
        # Load data
        mp_df = self.load_mp_data()
        
        # Calculate yearly proportions
        temporal_df = self.calculate_yearly_proportions(mp_df)
        
        # Create visualization
        fig, ax = self.create_visualization(temporal_df)
        
        # Save results
        self.save_visualization(fig)
        results = self.save_data(temporal_df)
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("ANALYSIS SUMMARY")
        logger.info("="*50)
        
        stats = results['key_statistics']
        logger.info(f"First female MPs: {stats['first_female_count']} in {stats['first_female_mp_year']}")
        logger.info(f"Peak representation: {stats['peak_proportion']:.1f}% ({stats['peak_count']} MPs) in {stats['peak_year']}")
        logger.info(f"Final representation (2005): {stats['final_year_proportion']:.1f}% ({stats['final_year_count']} MPs)")
        
        logger.info("\nHistorical milestones:")
        milestones = results['historical_milestones']
        for year, proportion in milestones.items():
            if proportion is not None:
                logger.info(f"  {year}: {proportion:.2f}%")
        
        plt.show()
        
        return temporal_df, results


def main():
    """Main function"""
    analyzer = FemaleMPTemporalAnalyzer()
    temporal_df, results = analyzer.run_analysis()


if __name__ == "__main__":
    main()