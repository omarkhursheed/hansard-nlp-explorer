#!/usr/bin/env python3
"""
Quick examination of the house_members_gendered.parquet file
"""

import pandas as pd
import numpy as np
from pathlib import Path

def examine_gendered_data():
    """Examine the structure and contents of the gendered house members data"""
    
    data_path = Path("src/hansard/data/house_members_gendered.parquet")
    
    if not data_path.exists():
        print(f"File not found: {data_path}")
        return
    
    print("Loading house_members_gendered.parquet...")
    df = pd.read_parquet(data_path)
    
    print(f"\n=== DATASET OVERVIEW ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    print(f"\n=== FIRST 10 ROWS ===")
    print(df.head(10))
    
    print(f"\n=== DATA TYPES ===")
    print(df.dtypes)
    
    print(f"\n=== SUMMARY STATISTICS ===")
    print(df.describe(include='all'))
    
    # Check for gender-related columns
    gender_columns = [col for col in df.columns if 'gender' in col.lower()]
    if gender_columns:
        print(f"\n=== GENDER COLUMNS ===")
        for col in gender_columns:
            print(f"\n{col}:")
            print(df[col].value_counts())
    
    # Check for date/year columns
    date_columns = [col for col in df.columns if any(word in col.lower() for word in ['date', 'year', 'time'])]
    if date_columns:
        print(f"\n=== DATE/TIME COLUMNS ===")
        for col in date_columns:
            print(f"\n{col}:")
            if df[col].dtype == 'object':
                print("Sample values:")
                print(df[col].dropna().head())
            else:
                print(f"Range: {df[col].min()} to {df[col].max()}")
    
    # Check for name/member columns
    name_columns = [col for col in df.columns if any(word in col.lower() for word in ['name', 'member', 'mp'])]
    if name_columns:
        print(f"\n=== NAME/MEMBER COLUMNS ===")
        for col in name_columns:
            print(f"\n{col} - unique values: {df[col].nunique()}")
            print("Sample values:")
            print(df[col].dropna().head())
    
    # Check for any female/male indicators
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_vals = df[col].dropna().unique()
            if any('female' in str(val).lower() or 'male' in str(val).lower() for val in unique_vals):
                print(f"\n=== POTENTIAL GENDER INDICATORS IN {col} ===")
                print(df[col].value_counts())
    
    print(f"\n=== MISSING VALUES ===")
    missing = df.isnull().sum()
    print(missing[missing > 0])
    
    return df

if __name__ == "__main__":
    df = examine_gendered_data()