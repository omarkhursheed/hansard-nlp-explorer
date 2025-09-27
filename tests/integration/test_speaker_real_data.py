"""
Real data tests for speaker processing module.
Tests with actual Hansard data to ensure everything works as expected.
Skips when real dataset isn't available locally.
"""

import sys
from pathlib import Path
import polars as pl
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from hansard.speaker_processing import SpeakerProcessor


DATA_PRESENT = (Path("src/hansard/data/processed_fixed/metadata/debates_master.parquet").exists())


@pytest.mark.skipif(not DATA_PRESENT, reason="Real dataset not present")
def test_with_real_data():
    """Test speaker processing with actual Hansard data."""
    print("Testing speaker processing with real data...")

    # Use actual data path
    data_path = Path("src/hansard/data")
    if data_path.exists():
        processor = SpeakerProcessor(data_dir=data_path / "processed_fixed")

        # Test 1: Try to load real debates
        debates_path = data_path / "processed_fixed/metadata/debates_master.parquet"
        if debates_path.exists():
            print(f"✓ Found debates at {debates_path}")

            # Read actual data
            debates_df = pl.read_parquet(debates_path)
            print(f"✓ Loaded {len(debates_df)} debates")

            # Check actual columns
            print(f"  Columns: {debates_df.columns[:5]}...")

            # Test speaker extraction from real data
            if 'speakers' in debates_df.columns:
                # Get sample of debates with speakers
                with_speakers = debates_df.filter(
                    pl.col('speakers').is_not_null() &
                    (pl.col('speakers').list.len() > 0)
                )
                print(f"✓ Found {len(with_speakers)} debates with speakers")

                if len(with_speakers) > 0:
                    # Extract some real speakers
                    sample = with_speakers.head(10)
                    for row in sample.iter_rows(named=True):
                        speakers = row.get('speakers', [])
                        if speakers:
                            print(f"  Sample speakers: {speakers[:3]}")
                            break

                    # Test normalization with real speaker names
                    real_speakers = []
                    for row in with_speakers.head(100).iter_rows(named=True):
                        if row['speakers']:
                            for speaker in row['speakers']:
                                real_speakers.append({'name': speaker})

                    if real_speakers:
                        speakers_df = pl.DataFrame(real_speakers).unique()
                        normalized = processor.normalize_speakers(speakers_df)
                        print(f"✓ Normalized {len(normalized)} unique speakers")

                        # Check normalization worked
                        if 'normalized_name' in normalized.columns:
                            samples = normalized.head(5)
                            for orig, norm in zip(samples['name'], samples['normalized_name']):
                                print(f"    '{orig}' → '{norm}'")
                else:
                    print("⚠ No debates with speakers found")
            else:
                print("⚠ No 'speakers' column in debates")

        else:
            print(f"✗ Debates file not found at {debates_path}")

        # Test 2: Check existing speaker datasets
        speaker_files = [
            "speakers_deduplicated.parquet",
            "speakers_normalized.parquet",
            "mp_speakers_gendered.parquet"
        ]

        for file in speaker_files:
            file_path = data_path / file
            if file_path.exists():
                df = pl.read_parquet(file_path)
                print(f"✓ {file}: {len(df)} records, columns: {df.columns[:3]}...")

    else:
        print(f"✗ Data directory not found: {data_path}")

    print("\nTest complete!")


@pytest.mark.skipif(not DATA_PRESENT, reason="Real dataset not present")
def test_validation_with_real_data():
    """Test validation functions with real speaker data."""
    print("\nTesting validation with real speaker data...")

    data_path = Path("src/hansard/data")
    if (data_path / "speakers_normalized.parquet").exists():
        # Load real speaker data
        speakers_df = pl.read_parquet(data_path / "speakers_normalized.parquet")
        print(f"✓ Loaded {len(speakers_df)} normalized speakers")

        processor = SpeakerProcessor()

        # Run validation
        validation = processor.validate_speaker_dataset(speakers_df)
        print(f"✓ Validation complete:")
        print(f"  Total speakers: {validation.get('total_speakers', 0)}")

        if 'date_coverage' in validation:
            print(f"  Date range: {validation['date_coverage']}")

        if 'issues' in validation:
            print(f"  Issues found: {len(validation['issues'])}")
            for issue in validation['issues'][:3]:
                print(f"    - {issue}")

        # Test MP coverage
        coverage = processor.check_mp_coverage(speakers_df)
        print(f"✓ MP coverage analysis:")
        print(f"  Total speakers: {coverage['total_speakers']}")
        print(f"  Likely MPs: {coverage['likely_mps']}")
        print(f"  Coverage: {coverage['coverage_percentage']:.1f}%")

    else:
        print("✗ Speaker data not found")


@pytest.mark.skipif(not DATA_PRESENT, reason="Real dataset not present")
def test_temporal_analysis_real():
    """Test temporal analysis with real data."""
    print("\nTesting temporal analysis with real data...")

    data_path = Path("src/hansard/data")
    speakers_path = data_path / "speakers_normalized.parquet"

    if speakers_path.exists():
        speakers_df = pl.read_parquet(speakers_path)

        # Check if temporal columns exist
        if 'first_appearance' in speakers_df.columns:
            processor = SpeakerProcessor()
            temporal = processor.generate_temporal_analysis(speakers_df)

            print(f"✓ Temporal analysis generated:")
            print(f"  Periods analyzed: {len(temporal)}")

            # Show sample periods
            for row in temporal.head(5).iter_rows(named=True):
                print(f"  {row['period']}: {row['active_speakers']} speakers, "
                      f"{row['new_speakers']} new")

        else:
            print("⚠ No temporal columns in speaker data")
    else:
        print("✗ Speaker data not found")


if __name__ == "__main__":
    print("="*60)
    print("REAL DATA TESTS FOR SPEAKER PROCESSING")
    print("="*60)

    test_with_real_data()
    test_validation_with_real_data()
    test_temporal_analysis_real()

    print("\n" + "="*60)
    print("All real data tests completed!")
    print("="*60)
