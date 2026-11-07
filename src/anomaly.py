import pandas as pd
from sklearn.ensemble import IsolationForest


def flag_unusual(df):
    """Flag unusual transactions using amount and category."""

    df = df.copy()

    # Check required columns
    if "amount" not in df.columns:
        raise ValueError("Missing amount column")

    if "category" not in df.columns:
        raise ValueError("Missing category column")

    # Prepare amount and category features
    features = df[["amount", "category"]].copy()
    features["amount"] = pd.to_numeric(
        features["amount"], errors="coerce"
    )

    # Handle missing values
    features["amount"] = features["amount"].fillna(
        features["amount"].median()
    ).fillna(0)

    features["category"] = features["category"].fillna("Other")

    # Convert category text to numerical columns
    features = pd.get_dummies(
        features,
        columns=["category"]
    )

    # Train Isolation Forest
    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42
    )

    predictions = model.fit_predict(features)

    # True means the transaction is unusual
    df["unusual"] = predictions == -1

    return df
