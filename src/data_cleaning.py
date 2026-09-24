"""
Load and clean transaction data from a CSV export.

Expected input columns (adjust based on the dataset we pick):
- description : str  (e.g. "STARBUCKS #4521")
- amount      : float
- date        : str
- category    : str  (only present in labeled training data)
"""

import pandas as pd


def load_transactions(csv_path: str) -> pd.DataFrame:
    """Load a transaction CSV into a DataFrame."""
    df = pd.read_csv(csv_path)
    return df


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop empty rows, standardize text, parse dates."""
    df = df.dropna(subset=["description"])
    df["description"] = df["description"].str.strip().str.upper()
    # TODO: parse date column once we finalize the dataset format
    return df


if __name__ == "__main__":
    # Quick manual test — update the path once we have real/sample data in /data
    df = load_transactions("data/sample_transactions.csv")
    df = clean_transactions(df)
    print(df.head())
