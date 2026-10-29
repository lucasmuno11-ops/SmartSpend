"""
Text classification: predict a spending category from a transaction description.

Pipeline: TF-IDF (text -> numbers) -> Naive Bayes (numbers -> predicted category)
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split


def load_training_data(csv_path):
    """Load merchant descriptions, category labels, and brands from CSV."""
    df = pd.read_csv(csv_path)
    required_columns = {"description", "category", "brand"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    df = df[["description", "category", "brand"]].copy()
    df["description"] = df["description"].astype(str)
    df["category"] = df["category"].astype(str)
    df["brand"] = df["brand"].astype(str)
    return df


def split_by_brand(df, test_size=0.2, random_state=42):
    """Split merchants by brand so the test set contains unseen merchants."""
    brands = list(df["brand"].dropna().unique())  # new
    train_brands, test_brands = train_test_split(
        brands, test_size=test_size, random_state=random_state
    )

    train_df = df[df["brand"].isin(train_brands)].copy()
    test_df = df[df["brand"].isin(test_brands)].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Brand-based split produced an empty train or test set.")

    return train_df, test_df


def train_categorizer(train_descriptions, train_categories, test_descriptions, test_categories):
    """
    Train a TF-IDF + Naive Bayes categorizer.

    train_descriptions : list[str]
    train_categories   : list[str]
    test_descriptions  : list[str]
    test_categories    : list[str]
    """
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(train_descriptions)
    X_test_vec = vectorizer.transform(test_descriptions)

    model = MultinomialNB(fit_prior=False)
    model.fit(X_train_vec, train_categories)

    return model, vectorizer, X_test_vec, test_categories


def evaluate_review_cutoffs(model, X_test_vec, y_test, cutoffs=(0.2, 0.3, 0.4)):
    """Report labeled share and accuracy for different confidence cutoffs."""
    probabilities = model.predict_proba(X_test_vec)
    max_confidence = probabilities.max(axis=1)
    predictions = model.predict(X_test_vec)
    y_test_array = np.asarray(y_test)

    results = []
    for cutoff in cutoffs:
        labeled_mask = max_confidence >= cutoff
        labeled_rows = int(labeled_mask.sum())
        labeled_share = labeled_rows / len(labeled_mask)

        if labeled_rows == 0:
            accuracy = 0.0
        else:
            accuracy = accuracy_score(y_test_array[labeled_mask], predictions[labeled_mask])

        results.append(
            {
                "cutoff": cutoff,
                "rows_labeled": labeled_rows,
                "share_labeled": labeled_share,
                "accuracy": accuracy,
            }
        )

    return results


if __name__ == "__main__":
    csv_path = Path(__file__).resolve().parent.parent / "data" / "real_merchants.csv"
    df = load_training_data(csv_path)
    train_df, test_df = split_by_brand(df)

    model, vectorizer, X_test_vec, y_test = train_categorizer(
        train_df["description"].tolist(),
        train_df["category"].tolist(),
        test_df["description"].tolist(),
        test_df["category"].tolist(),
    )

    predictions = model.predict(X_test_vec)
    print(f"Train brands: {train_df['brand'].nunique()} | Test brands: {test_df['brand'].nunique()}")
    print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")
    print("Vocabulary size:", len(vectorizer.vocabulary_))
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    cutoffs = (0.2, 0.3, 0.4)
    print("\nNeeds review analysis using predict_proba max confidence:")
    for result in evaluate_review_cutoffs(model, X_test_vec, y_test, cutoffs=cutoffs):
        print(
            f"cutoff={result['cutoff']:.1f} | rows_labeled={result['rows_labeled']} | "
            f"share_labeled={result['share_labeled']:.3f} | accuracy={result['accuracy']:.3f}"
        )

    models_dir = Path(__file__).resolve().parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(model, models_dir / "categorizer_model.joblib")
    joblib.dump(vectorizer, models_dir / "categorizer_vectorizer.joblib")
    print(f"\nSaved model artifacts to: {models_dir}")
