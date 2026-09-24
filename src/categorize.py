"""
Text classification: predict a spending category from a transaction description.

Pipeline: TF-IDF (text -> numbers) -> Naive Bayes (numbers -> predicted category)
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split


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
    # TODO: replace with real data once we have a dataset loaded via data_cleaning.py
    sample_descriptions = ["STARBUCKS", "SHELL OIL", "NETFLIX.COM", "TRADER JOES"]
    sample_categories = ["Dining", "Transportation", "Subscriptions", "Groceries"]

    model, vectorizer, X_test, y_test = train_categorizer(
        sample_descriptions, sample_categories
    )
    print("Model trained. Vocabulary size:", len(vectorizer.vocabulary_))
