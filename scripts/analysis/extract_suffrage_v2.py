#!/usr/bin/env python3
"""
Version 2: Improved suffrage extraction with two-tier approach.

Tier 1 (HIGH PRECISION): Explicit suffrage terms
Tier 2 (HIGH RECALL): Women + political terms with proximity

Balance precision and recall by using simpler, broader search then filtering.
"""

import pandas as pd
from pathlib import Path


class SuffrageExtractorV2:
    """Improved extractor with better recall."""

    def __init__(self, year_range=(1900, 1935), data_dir='data-hansard/derived_complete'):
        self.year_range = year_range
        self.data_dir = Path(data_dir)

    def extract_two_tier(self, chamber='Commons', min_word_count=50):
        """
        Two-tier extraction strategy:
        - Tier 1: High-precision explicit suffrage terms
        - Tier 2: Women + political terms (broader but may need filtering)

        Args:
            chamber: Filter to specific chamber
            min_word_count: Minimum words to count as substantive speech

        Returns:
            pd.DataFrame with additional 'confidence_tier' column
        """
        print(f"\nTwo-tier extraction: {self.year_range[0]}-{self.year_range[1]} ({chamber})")
        print(f"Minimum speech length: {min_word_count} words\n")

        all_speeches = []

        for year in range(self.year_range[0], self.year_range[1] + 1):
            speech_file = self.data_dir / 'speeches_complete' / f'speeches_{year}.parquet'

            if not speech_file.exists():
                continue

            speeches_df = pd.read_parquet(speech_file)

            # Filter chamber and length
            if chamber:
                speeches_df = speeches_df[speeches_df['chamber'] == chamber]
            speeches_df = speeches_df[speeches_df['word_count'] >= min_word_count]

            # TIER 1: HIGH PRECISION - Explicit suffrage terms
            tier1_pattern = (
                'women.*suffrage|female suffrage|suffrage.*women|'
                'votes for women|suffragette|suffragist|'
                'enfranchise.*women|women.*enfranchise|'
                'equal franchise|'
                'representation of the people.*women|'  # Must mention women
                'sex disqualification|'
                'women.*social.*political.*union'
            )

            tier1_match = speeches_df['text'].str.contains(tier1_pattern, case=False, na=False)

            # TIER 2: BROADER - Women + political/voting terms
            # Must have "women" or "female" AND voting-related term
            tier2_women = speeches_df['text'].str.contains('women|female', case=False, na=False)
            tier2_political = speeches_df['text'].str.contains(
                'suffrage|franchise|enfranchise|'
                'vote|votes|voting|voter|voters|electoral|electorate|'
                'representation.*people|parliamentary.*franchise',
                case=False, na=False
            )
            tier2_match = tier2_women & tier2_political & ~tier1_match  # Only new matches

            # Combine
            combined_match = tier1_match | tier2_match

            if combined_match.sum() > 0:
                year_speeches = speeches_df[combined_match].copy()

                # Add confidence tier
                year_speeches['confidence_tier'] = 'tier2'
                year_speeches.loc[tier1_match[combined_match], 'confidence_tier'] = 'tier1'

                print(f"  {year}: {combined_match.sum():4} speeches "
                      f"(Tier1: {tier1_match.sum():3}, Tier2: {tier2_match.sum():3})")

                all_speeches.append(year_speeches)
            else:
                print(f"  {year}: No speeches found")

        if all_speeches:
            speeches_combined = pd.concat(all_speeches, ignore_index=True)

            print(f"\n{'='*70}")
            print(f"EXTRACTION COMPLETE")
            print(f"{'='*70}")
            print(f"Total speeches: {len(speeches_combined):,}")
            print(f"  Tier 1 (high precision): {(speeches_combined['confidence_tier'] == 'tier1').sum():,}")
            print(f"  Tier 2 (broader): {(speeches_combined['confidence_tier'] == 'tier2').sum():,}")
            print(f"Date range: {speeches_combined['date'].min()} to {speeches_combined['date'].max()}")

            return speeches_combined
        else:
            print("\nNo speeches found!")
            return None


def run_extraction():
    """Run the improved extraction."""
    print("="*70)
    print("IMPROVED SUFFRAGE EXTRACTION v2")
    print("Two-tier approach for better recall with precision tracking")
    print("="*70)

    extractor = SuffrageExtractorV2(year_range=(1900, 1935))
    speeches = extractor.extract_two_tier(chamber='Commons', min_word_count=50)

    if speeches is not None:
        # Three-era analysis
        print(f"\n{'='*70}")
        print("THREE-ERA ANALYSIS")
        print(f"{'='*70}")

        pre_1918 = speeches[speeches['year'] < 1918]
        partial = speeches[(speeches['year'] >= 1918) & (speeches['year'] < 1928)]
        post_1928 = speeches[speeches['year'] >= 1928]

        for era_name, era_data in [
            ('Pre-1918 (no women vote)', pre_1918),
            ('1918-1927 (partial suffrage)', partial),
            ('Post-1928 (equal suffrage)', post_1928)
        ]:
            matched_era = era_data[era_data['matched_mp'] == True]
            tier1 = era_data[era_data['confidence_tier'] == 'tier1']
            tier2 = era_data[era_data['confidence_tier'] == 'tier2']

            print(f"\n{era_name}:")
            print(f"  Total speeches: {len(era_data):,}")
            print(f"    Tier 1: {len(tier1):,}, Tier 2: {len(tier2):,}")
            print(f"  Matched to MPs: {len(matched_era):,} ({len(matched_era)/len(era_data)*100:.1f}%)")

            if len(matched_era) > 0:
                gender = matched_era['gender'].value_counts()
                print(f"  Male: {gender.get('M', 0):,}, Female: {gender.get('F', 0):,}")

        # Save results
        print(f"\n{'='*70}")
        print("SAVING RESULTS")
        print(f"{'='*70}")

        output_dir = Path('outputs/suffrage_v2')
        output_dir.mkdir(parents=True, exist_ok=True)

        speeches.to_parquet(output_dir / 'speeches_all.parquet')

        # Also save tier-specific versions
        tier1 = speeches[speeches['confidence_tier'] == 'tier1']
        tier2 = speeches[speeches['confidence_tier'] == 'tier2']

        tier1.to_parquet(output_dir / 'speeches_tier1_high_precision.parquet')
        tier2.to_parquet(output_dir / 'speeches_tier2_broader.parquet')

        # CSV samples
        speeches.head(500).to_csv(output_dir / 'speeches_sample.csv', index=False)
        tier1.head(100).to_csv(output_dir / 'tier1_sample.csv', index=False)
        tier2.head(100).to_csv(output_dir / 'tier2_sample.csv', index=False)

        print(f"\nSaved to {output_dir}/")
        print(f"  - speeches_all.parquet ({len(speeches):,} speeches)")
        print(f"  - speeches_tier1_high_precision.parquet ({len(tier1):,} speeches)")
        print(f"  - speeches_tier2_broader.parquet ({len(tier2):,} speeches)")
        print(f"  - CSV samples for inspection")

        return speeches

    return None


if __name__ == '__main__':
    speeches = run_extraction()
