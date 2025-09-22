#!/usr/bin/env python3
"""
Analyze the gender truth data and understand what we can use for matching
"""

import pandas as pd
from pathlib import Path

def analyze_gender_data():
    """Deep dive into gender data for matching strategy"""

    data_path = Path("src/hansard/data/house_members_gendered_updated.parquet")
    df = pd.read_parquet(data_path)

    print("=== GENDER INFORMATION ===")
    print(f"Total records: {len(df)}")

    # Gender columns
    print("\nGender columns:")
    print(f"gender_inferred values: {df['gender_inferred'].value_counts().to_dict()}")
    print(f"gender_source values: {df['gender_source'].value_counts().to_dict()}")
    print(f"Records with gender: {df['gender_inferred'].notna().sum()} ({100*df['gender_inferred'].notna().sum()/len(df):.1f}%)")

    # Check unique persons
    print(f"\n=== UNIQUE PERSONS ===")
    print(f"Unique person_name values: {df['person_name'].nunique()}")
    print(f"Records per person (avg): {len(df) / df['person_name'].nunique():.1f}")

    # Group by person and check gender consistency
    person_genders = df.groupby('person_name')['gender_inferred'].agg(['nunique', 'first', 'count'])
    inconsistent = person_genders[person_genders['nunique'] > 1]
    if len(inconsistent) > 0:
        print(f"\nPersons with inconsistent gender: {len(inconsistent)}")
        print(inconsistent.head())
    else:
        print("\nAll persons have consistent gender across records")

    # Create unique person-gender mapping
    person_gender_map = df.dropna(subset=['gender_inferred']).groupby('person_name')['gender_inferred'].first()
    print(f"\n=== PERSON-GENDER MAPPING ===")
    print(f"Unique persons with gender: {len(person_gender_map)}")
    print(f"Gender distribution:")
    print(person_gender_map.value_counts())

    # Check IDs that might be useful for matching
    print("\n=== USEFUL IDs FOR MATCHING ===")
    id_cols = [c for c in df.columns if c.startswith('id_') or c.startswith('mem_id_')]
    for col in id_cols:
        non_null = df[col].notna().sum()
        if non_null > 10000:  # Only show IDs with good coverage
            print(f"{col}: {non_null} non-null ({100*non_null/len(df):.1f}%)")

    # Check aliases for matching
    print("\n=== ALIASES FOR MATCHING ===")
    aliases_non_null = df['aliases_norm'].notna().sum()
    print(f"Records with aliases_norm: {aliases_non_null} ({100*aliases_non_null/len(df):.1f}%)")

    # Sample aliases
    if aliases_non_null > 0:
        print("\nSample aliases_norm values:")
        for idx, val in df[df['aliases_norm'].notna()]['aliases_norm'].head(3).items():
            print(f"  {df.loc[idx, 'person_name']}: {val}")

    # Date ranges
    print("\n=== TEMPORAL COVERAGE ===")
    print(f"Birth years: {df['birth_year'].min()} to {df['birth_year'].max()}")
    print(f"Death years: {df['death_year'].min()} to {df['death_year'].max()}")

    # Check membership dates
    df['membership_start_year'] = pd.to_datetime(df['membership_start_date']).dt.year
    df['membership_end_year'] = pd.to_datetime(df['membership_end_date']).dt.year

    print(f"Membership years: {df['membership_start_year'].min():.0f} to {df['membership_end_year'].max():.0f}")

    return df, person_gender_map

if __name__ == "__main__":
    df, person_gender_map = analyze_gender_data()

    # Save the person-gender mapping
    print("\n=== SAVING PERSON-GENDER MAPPING ===")
    output_path = Path("src/hansard/data/person_gender_mapping.csv")
    person_gender_map.to_csv(output_path)
    print(f"Saved to {output_path}")