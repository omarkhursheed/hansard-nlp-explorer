"""
Consolidated speaker processing module for Hansard data.
Combines functionality from multiple speaker-related scripts.
"""

import polars as pl
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import logging
from collections import defaultdict
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SpeakerProcessor:
    """Main class for all speaker-related processing."""

    def __init__(self, data_dir: Path = Path("data/processed")):
        self.data_dir = data_dir
        self.metadata_dir = data_dir / "metadata"
        self.content_dir = data_dir / "content"

    def create_mp_speakers(self, output_path: Optional[Path] = None,
                           filter_mps_only: bool = True) -> pl.DataFrame:
        """
        Create speaker dataset from debates, optionally filtering to MPs only.
        Consolidates functionality from create_mp_*.py scripts.
        """
        logger.info(f"Creating speaker dataset (MPs only: {filter_mps_only})")

        # Load debates metadata
        debates_path = self.metadata_dir / "debates_master.parquet"
        if not debates_path.exists():
            raise FileNotFoundError(f"Debates master file not found: {debates_path}")

        debates_df = pl.read_parquet(debates_path)

        # Extract unique speakers
        speakers = []
        for row in debates_df.iter_rows(named=True):
            if row['speakers']:
                for speaker in row['speakers']:
                    if not filter_mps_only or self._is_mp(speaker):
                        speakers.append({
                            'name': speaker,
                            'year': row['year'],
                            'chamber': row.get('chamber', 'unknown'),
                            'debate_id': row.get('hansard_ref', '')
                        })

        # Create DataFrame
        speakers_df = pl.DataFrame(speakers)

        # Deduplicate and aggregate
        speakers_agg = speakers_df.group_by(['name', 'chamber']).agg([
            pl.col('year').min().alias('first_appearance'),
            pl.col('year').max().alias('last_appearance'),
            pl.col('debate_id').n_unique().alias('debate_count')
        ])

        if output_path:
            speakers_agg.write_parquet(output_path)
            logger.info(f"Saved speaker dataset to {output_path}")

        return speakers_agg

    def normalize_speakers(self, speakers_df: pl.DataFrame) -> pl.DataFrame:
        """
        Normalize speaker names to handle variations.
        Consolidates normalize_speakers.py functionality.
        """
        logger.info("Normalizing speaker names")

        # Common normalizations
        normalizations = {
            'mr.': 'mr',
            'mrs.': 'mrs',
            'ms.': 'ms',
            'dr.': 'dr',
            'sir': 'sir',
            'lord': 'lord',
            'lady': 'lady'
        }

        def normalize_name(name: str) -> str:
            if name is None:
                return ""
            name_lower = name.lower().strip()
            for old, new in normalizations.items():
                name_lower = name_lower.replace(old, new)
            # Remove extra whitespace
            return ' '.join(name_lower.split())

        # Apply normalization with proper return type
        normalized_df = speakers_df.with_columns([
            pl.col('name').map_elements(normalize_name, return_dtype=pl.Utf8).alias('normalized_name')
        ])

        return normalized_df

    def deduplicate_speakers(self, speakers_df: pl.DataFrame) -> pl.DataFrame:
        """
        Deduplicate speaker records based on normalized names.
        Consolidates deduplicate_speakers.py functionality.
        """
        logger.info("Deduplicating speaker records")

        # Ensure normalized names exist
        if 'normalized_name' not in speakers_df.columns:
            speakers_df = self.normalize_speakers(speakers_df)

        # Build aggregation list based on available columns
        agg_list = [pl.col('name').first().alias('canonical_name')]

        if 'first_appearance' in speakers_df.columns:
            agg_list.append(pl.col('first_appearance').min().alias('first_appearance'))
        if 'last_appearance' in speakers_df.columns:
            agg_list.append(pl.col('last_appearance').max().alias('last_appearance'))
        if 'debate_count' in speakers_df.columns:
            agg_list.append(pl.col('debate_count').sum().alias('total_debates'))
        if 'primary_chamber' in speakers_df.columns:
            agg_list.append(pl.col('primary_chamber').mode().first().alias('primary_chamber'))
        elif 'chamber' in speakers_df.columns:
            agg_list.append(pl.col('chamber').mode().first().alias('primary_chamber'))

        # Group by normalized name and aggregate
        dedup_df = speakers_df.group_by('normalized_name').agg(agg_list)

        return dedup_df

    def extract_speaker_debates(self, speaker_name: str,
                                year_range: Optional[Tuple[int, int]] = None) -> List[Dict]:
        """
        Extract all debates for a specific speaker.
        Consolidates extract_speaker_debates.py functionality.
        """
        logger.info(f"Extracting debates for speaker: {speaker_name}")

        # Load debates
        debates_df = pl.read_parquet(self.metadata_dir / "debates_master.parquet")

        # Filter by speaker
        speaker_debates = debates_df.filter(
            pl.col('speakers').list.contains(speaker_name)
        )

        # Apply year range if specified
        if year_range:
            speaker_debates = speaker_debates.filter(
                (pl.col('year') >= year_range[0]) &
                (pl.col('year') <= year_range[1])
            )

        return speaker_debates.to_dicts()

    def validate_speaker_dataset(self, speakers_df: pl.DataFrame) -> Dict:
        """
        Validate speaker dataset for completeness and quality.
        Consolidates validate_speaker_dataset.py functionality.
        """
        logger.info("Validating speaker dataset")

        validation_report = {
            'total_speakers': len(speakers_df),
            'issues': []
        }

        # Check for debate count if column exists
        if 'debate_count' in speakers_df.columns:
            validation_report['speakers_with_debates'] = len(
                speakers_df.filter(pl.col('debate_count') > 0)
            )

        # Check date coverage if columns exist
        if 'first_appearance' in speakers_df.columns and 'last_appearance' in speakers_df.columns:
            validation_report['date_coverage'] = {
                'earliest': speakers_df['first_appearance'].min(),
                'latest': speakers_df['last_appearance'].max()
            }

        # Check chamber distribution if column exists
        if 'primary_chamber' in speakers_df.columns:
            validation_report['chamber_distribution'] = speakers_df['primary_chamber'].value_counts().to_dict()
        elif 'chamber' in speakers_df.columns:
            validation_report['chamber_distribution'] = speakers_df['chamber'].value_counts().to_dict()

        # Check for missing data
        null_counts = {
            col: speakers_df[col].null_count()
            for col in speakers_df.columns
        }

        for col, null_count in null_counts.items():
            if null_count > 0:
                validation_report['issues'].append(
                    f"Column '{col}' has {null_count} null values"
                )

        # Check for suspicious patterns (only if columns exist)
        if 'first_appearance' in speakers_df.columns and 'last_appearance' in speakers_df.columns:
            if len(speakers_df.filter(pl.col('first_appearance') > pl.col('last_appearance'))) > 0:
                validation_report['issues'].append(
                    "Some speakers have first_appearance > last_appearance"
                )

        return validation_report

    def check_mp_coverage(self, speakers_df: pl.DataFrame) -> Dict:
        """
        Analyze MP coverage in the dataset.
        Consolidates check_mp_coverage.py functionality.
        """
        logger.info("Checking MP coverage")

        # Rough heuristics for MP identification
        # Include mr., mrs. as they were commonly used for MPs
        mp_indicators = ['mp', 'member', 'hon.', 'right hon.', 'mr.', 'mrs.']

        def likely_mp(name: str) -> bool:
            if name is None:
                return False
            name_lower = name.lower()
            return any(indicator in name_lower for indicator in mp_indicators)

        # Apply MP detection with proper return type
        # Handle different possible column names for speaker name
        name_col = None
        if 'name' in speakers_df.columns:
            name_col = 'name'
        elif 'speaker_name' in speakers_df.columns:
            name_col = 'speaker_name'
        elif 'normalized_name' in speakers_df.columns:
            name_col = 'normalized_name'
        else:
            # If no name column, return empty coverage
            return {
                'total_speakers': len(speakers_df),
                'likely_mps': 0,
                'coverage_percentage': 0.0,
                'by_chamber': {},
                'error': 'No name column found'
            }

        with_mp_flag = speakers_df.with_columns([
            pl.col(name_col).map_elements(likely_mp, return_dtype=pl.Boolean).alias('likely_mp')
        ])

        coverage = {
            'total_speakers': len(speakers_df),
            'likely_mps': len(with_mp_flag.filter(pl.col('likely_mp'))),
            'coverage_percentage': (
                len(with_mp_flag.filter(pl.col('likely_mp'))) / len(speakers_df) * 100
            ),
            'by_chamber': {}
        }

        # Coverage by chamber
        chamber_col = 'primary_chamber' if 'primary_chamber' in speakers_df.columns else 'chamber'
        if chamber_col in speakers_df.columns:
            for chamber in speakers_df[chamber_col].unique():
                if chamber is not None:
                    chamber_df = with_mp_flag.filter(pl.col(chamber_col) == chamber)
                    coverage['by_chamber'][str(chamber)] = {
                        'total': len(chamber_df),
                        'likely_mps': len(chamber_df.filter(pl.col('likely_mp')))
                    }

        return coverage

    def _is_mp(self, name: str) -> bool:
        """Helper to determine if a speaker is likely an MP."""
        mp_indicators = ['mp', 'member', 'hon.', 'right hon.']
        name_lower = name.lower()
        return any(indicator in name_lower for indicator in mp_indicators)

    def generate_temporal_analysis(self, speakers_df: pl.DataFrame) -> pl.DataFrame:
        """
        Generate temporal analysis of speaker participation.
        Consolidates mp_temporal_gender_analysis.py functionality.
        """
        logger.info("Generating temporal analysis")

        # Group by year ranges
        year_bins = list(range(1800, 2010, 10))  # Decades

        temporal_data = []
        for i in range(len(year_bins) - 1):
            start_year = year_bins[i]
            end_year = year_bins[i + 1]

            period_speakers = speakers_df.filter(
                (pl.col('first_appearance') <= end_year) &
                (pl.col('last_appearance') >= start_year)
            )

            temporal_data.append({
                'period': f"{start_year}-{end_year}",
                'start_year': start_year,
                'end_year': end_year,
                'active_speakers': len(period_speakers),
                'new_speakers': len(period_speakers.filter(
                    (pl.col('first_appearance') >= start_year) &
                    (pl.col('first_appearance') < end_year)
                ))
            })

        return pl.DataFrame(temporal_data)


# Convenience functions for backward compatibility
def create_mp_speakers(output_path: Optional[Path] = None) -> pl.DataFrame:
    """Create MP speaker dataset."""
    processor = SpeakerProcessor()
    return processor.create_mp_speakers(output_path, filter_mps_only=True)


def normalize_speakers(speakers_df: pl.DataFrame) -> pl.DataFrame:
    """Normalize speaker names."""
    processor = SpeakerProcessor()
    return processor.normalize_speakers(speakers_df)


def deduplicate_speakers(speakers_df: pl.DataFrame) -> pl.DataFrame:
    """Deduplicate speaker records."""
    processor = SpeakerProcessor()
    return processor.deduplicate_speakers(speakers_df)


def validate_speaker_dataset(speakers_df: pl.DataFrame) -> Dict:
    """Validate speaker dataset."""
    processor = SpeakerProcessor()
    return processor.validate_speaker_dataset(speakers_df)


if __name__ == "__main__":
    # Example usage
    processor = SpeakerProcessor()

    # Create speaker dataset
    speakers = processor.create_mp_speakers()
    print(f"Created dataset with {len(speakers)} speakers")

    # Validate
    validation = processor.validate_speaker_dataset(speakers)
    print(f"Validation report: {validation}")

    # Check coverage
    coverage = processor.check_mp_coverage(speakers)
    print(f"MP coverage: {coverage['coverage_percentage']:.1f}%")