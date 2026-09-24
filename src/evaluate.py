"""
Evaluate the categorizer's performance using precision, recall, and F1.
"""

from sklearn.metrics import classification_report


def evaluate_model(model, X_test, y_test):
    """Print a precision/recall/F1 report for the trained model."""
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))
    return predictions


if __name__ == "__main__":
    print("Run this after training a model in categorize.py — see README for usage.")
