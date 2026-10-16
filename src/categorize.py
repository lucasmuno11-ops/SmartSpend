"""
Text classification: predict a spending category from a transaction description.

Pipeline: TF-IDF (text -> numbers) -> Naive Bayes (numbers -> predicted category)
"""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split


def load_training_data(csv_path):
    """Load merchant descriptions and category labels from a CSV file."""
    df = pd.read_csv(csv_path)
    required_columns = {"description", "category"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    return df["description"].astype(str).tolist(), df["category"].astype(str).tolist()


def train_categorizer(descriptions, categories):
    """
    Train a TF-IDF + Naive Bayes categorizer.

    descriptions : list[str]  e.g. ["STARBUCKS #4521", "SHELL OIL 12345"]
    categories   : list[str]  e.g. ["Dining", "Transportation"]
    """
    X_train, X_test, y_train, y_test = train_test_split(
        descriptions, categories, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    return model, vectorizer, X_test_vec, y_test


if __name__ == "__main__":
    csv_path = Path(__file__).resolve().parent.parent / "data" / "real_merchants.csv"
    descriptions, categories = load_training_data(csv_path)

    model, vectorizer, X_test, y_test = train_categorizer(descriptions, categories)
    print(f"Trained on {len(descriptions)} examples.")
    print("Model trained. Vocabulary size:", len(vectorizer.vocabulary_))
