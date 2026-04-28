#!/usr/bin/env python3
"""
Table 5: Sexism type by stance (% of sexist speeches within each stance).

Reproduces Table 5 from the paper by cross-tabulating the LLM sexism
classifications against speech stance (for/against/both suffrage).

Usage:
    python scripts/manuscript/05_table5_sexism_by_stance.py
"""

import pandas as pd
from pathlib import Path


def find_project_root() -> Path:
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / ".git").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parents[2]


ROOT = find_project_root()
CLASSIFICATION_DATA = ROOT / "outputs" / "llm_classification" / "v7_notrunc_results.parquet"

STANCES = ["for", "against", "both"]

AXIS_A_LABELS = ["hostile", "benevolent"]
AXIS_B_LABELS = [
    "paternalistic_prejudice",
    "contemptuous_prejudice",
    "admiration",
    "envious_prejudice",
]
AXIS_C_LABELS = ["descriptive", "prescriptive", "proscriptive"]

AXIS_B_DISPLAY = {
    "paternalistic_prejudice": "Paternalistic",
    "contemptuous_prejudice": "Contemptuous",
    "admiration": "Admiration",
    "envious_prejudice": "Envious",
}


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(CLASSIFICATION_DATA)
    return df[df["stance"].isin(STANCES)]


def pct(count: int, total: int) -> str:
    if total == 0:
        return "-"
    return f"{count / total * 100:.0f}%"


def compute_table(df: pd.DataFrame) -> None:
    header = f"{'':30s}"
    for stance in STANCES:
        n = len(df[df["stance"] == stance])
        header += f"  {stance.capitalize():>10s} (n={n:,})"
    print(header)
    print("-" * len(header))

    # Row 1: Sexist (% of stance)
    row = f"{'Sexist (% of stance)':30s}"
    for stance in STANCES:
        sub = df[df["stance"] == stance]
        sexist_n = (sub["binary"] == "sexist").sum()
        row += f"  {pct(sexist_n, len(sub)):>18s}"
    print(row)
    print()

    # Axis A: Ambivalent Sexism
    print("Axis A: Ambivalent Sexism")
    for label in AXIS_A_LABELS:
        row = f"  {label.capitalize():28s}"
        for stance in STANCES:
            sexist = df[(df["stance"] == stance) & (df["binary"] == "sexist")]
            count = (sexist["axis_a_label"] == label).sum()
            row += f"  {pct(count, len(sexist)):>18s}"
        print(row)
    print()

    # Axis B: Stereotype Content (denominator = all sexist, not just non-none)
    print("Axis B: Stereotype Content")
    for label in AXIS_B_LABELS:
        display = AXIS_B_DISPLAY[label]
        row = f"  {display:28s}"
        for stance in STANCES:
            sexist = df[(df["stance"] == stance) & (df["binary"] == "sexist")]
            count = (sexist["axis_b_label"] == label).sum()
            row += f"  {pct(count, len(sexist)):>18s}"
        print(row)
    print()

    # Axis C: Gender Norm Type
    print("Axis C: Gender Norm Type")
    for label in AXIS_C_LABELS:
        row = f"  {label.capitalize():28s}"
        for stance in STANCES:
            sexist = df[(df["stance"] == stance) & (df["binary"] == "sexist")]
            count = (sexist["axis_c_label"] == label).sum()
            row += f"  {pct(count, len(sexist)):>18s}"
        print(row)


def main():
    print("Table 5: Sexism type by stance (% of sexist speeches within each stance)")
    print("=" * 80)
    print(f"Data: {CLASSIFICATION_DATA.name}")
    print()

    df = load_data()
    print(f"Total speeches with stance: {len(df):,}")
    print(f"  Sexist: {(df['binary'] == 'sexist').sum():,}")
    print(f"  Not sexist: {(df['binary'] == 'not_sexist').sum():,}")
    print()

    compute_table(df)


if __name__ == "__main__":
    main()
