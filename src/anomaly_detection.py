from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# Locate the dataset
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "sample_transactions.csv"


def main():
    # Load transactions
    df = pd.read_csv(DATA_PATH)

    # Keep the columns we need
    df = df[["amt", "team_category", "is_fraud"]].copy()

    # Remove rows with missing or invalid values
    df["amt"] = pd.to_numeric(df["amt"], errors="coerce")
    df["is_fraud"] = pd.to_numeric(
        df["is_fraud"], errors="coerce"
    )
    df = df.dropna()
    df = df[df["is_fraud"].isin([0, 1])]

    print("Total transactions:", len(df))
    print("Fraud transactions:", int(df["is_fraud"].sum()))

    # Split into training and testing data
    train_df, test_df = train_test_split(
        df,
        test_size=0.3,
        random_state=42,
    )

    # Train on normal transactions only
    normal_train = train_df[train_df["is_fraud"] == 0]

    # Convert categories into numerical features
    X_train = pd.get_dummies(
        normal_train[["amt", "team_category"]],
        columns=["team_category"],
    )

    X_test = pd.get_dummies(
        test_df[["amt", "team_category"]],
        columns=["team_category"],
    )

    # Ensure training and test columns match
    X_test = X_test.reindex(
        columns=X_train.columns,
        fill_value=0
    )

    # Build and train Isolation Forest
    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42,
    )

    model.fit(X_train)

    # Predict unusual transactions
    predictions = model.predict(X_test)

    # Convert -1 (anomaly) to 1 and 1 (normal) to 0
    predicted_anomalies = (predictions == -1).astype(int)

    # Actual fraud labels
    actual_fraud = test_df["is_fraud"].astype(int)

    print("\nIsolation Forest Results")
    print("Test transactions:", len(test_df))
    print("Actual fraud in test set:", int(actual_fraud.sum()))
    print("Flagged anomalies:", int(predicted_anomalies.sum()))

    print("Precision:", round(
        precision_score(
            actual_fraud,
            predicted_anomalies,
            zero_division=0
        ), 3
    ))

    print("Recall:", round(
        recall_score(
            actual_fraud,
            predicted_anomalies,
            zero_division=0
        ), 3
    ))

    print("F1 Score:", round(
        f1_score(
            actual_fraud,
            predicted_anomalies,
            zero_division=0
        ), 3
    ))

    print("\nConfusion Matrix:")
    print(confusion_matrix(
        actual_fraud,
        predicted_anomalies,
        labels=[0, 1]
    ))


if __name__ == "__main__":
    main()
