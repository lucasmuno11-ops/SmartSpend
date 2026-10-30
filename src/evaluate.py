from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from categorize import (
    load_training_data,
    split_by_brand,
    train_categorizer,
)

# Find the dataset
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "real_merchants.csv"


def get_accuracy(train_df, test_df):
    model, vectorizer, X_test, y_test = train_categorizer(
        train_df["description"].tolist(),
        train_df["category"].tolist(),
        test_df["description"].tolist(),
        test_df["category"].tolist(),
    )

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    return accuracy, y_test, predictions



def main():
    # Load the merchant dataset
    df = load_training_data(DATA_PATH)

    # Test three different random states
    for seed in [42, 7, 99]:

        # Random split
        random_train, random_test = train_test_split(
            df,
            test_size=0.2,
            random_state=seed,
        )

        random_accuracy, _, _ = get_accuracy(
            random_train, random_test
        )
        

        # Unseen-brand split
        brand_train, brand_test = split_by_brand(
            df,
            test_size=0.2,
            random_state=seed,
        )

        brand_accuracy, y_test, predictions = get_accuracy(
            brand_train, brand_test
        )

        print(f"\nRandom State: {seed}")
        print(f"Random Split Accuracy: {random_accuracy:.2%}")
        print(f"Unseen Brand Accuracy: {brand_accuracy:.2%}")
        # Create a confusion matrix for seed 42
        if seed == 42:
            ConfusionMatrixDisplay.from_predictions(
                y_test,
                predictions,
                xticks_rotation=45,
                cmap="Blues",
                values_format="d"
            )

            plt.title("SmartSpend Unseen Brand Confusion Matrix")
            plt.tight_layout()

            # Save the chart
            output_folder = ROOT / "docs"
            output_folder.mkdir(exist_ok=True)

            plt.savefig(
                output_folder / "confusion_matrix.png",
                dpi=150,
                bbox_inches="tight"
            )

            plt.close()

            print("Confusion matrix saved!")



if __name__ == "__main__":
    main()
