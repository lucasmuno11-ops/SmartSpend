# SmartSpend Evaluation Results

## 1. Categorization Model

Model: TF-IDF + Multinomial Naive Bayes

Dataset: data/real_merchants.csv

### Accuracy Comparison

| Random State | Random Split | Unseen Brand Split |
|---|---|---|
| 42 | 98.99% | 55.98% |
| 7 | 98.29% | 54.23% |
| 99 | 98.99% | 57.48% |
| Average | 98.76% | 55.90% |

### Analysis

The categorization model performed significantly better
on random transaction splits than unseen-brand splits.

Random splitting achieved an average accuracy of 98.76%,
compared with 55.90% for unseen brands.

This 42.86 percentage-point difference suggests the model
relies heavily on recognizing familiar merchants.

Unseen-brand testing provides a more realistic measure
of how the model handles unfamiliar merchants.

## 2. Confusion Matrix

File: docs/confusion_matrix.png

The confusion matrix was generated using the unseen-brand
split with random state 42.

### Main Findings

The largest classification errors included:

- Entertainment predicted as Dining: 48 transactions
- Entertainment predicted as Other: 38 transactions
- Transportation predicted as Dining: 30 transactions
- Education predicted as Other: 20 transactions
- Other predicted as Dining: 19 transactions
- Transportation predicted as Shopping: 18 transactions

Entertainment was particularly difficult to classify,
with only 41 of 179 transactions correctly predicted.

Other weak categories included Housing, Other,
and Transportation.

### Analysis

The model frequently confused unfamiliar merchants with
Dining and Other. This suggests that transaction descriptions
do not always provide enough information to distinguish
spending categories.

Adding more diverse merchant examples may improve accuracy
on unfamiliar brands.

## 3. Isolation Forest

Dataset: data/sample_transactions.csv

Model: Isolation Forest

Features:
- Transaction amount (amt)
- Spending category (team_category)

Training data: Normal transactions only

Test size: 30%
Random state: 42
Contamination: 0.05

### Results

Total transactions: 2,000
Total fraud transactions: 4
Test transactions: 600
Fraud transactions in test set: 1
Flagged anomalies: 35

Precision: 0.029
Recall: 1.000
F1 Score: 0.056

### Confusion Matrix

True Negatives: 565
False Positives: 34
False Negatives: 0
True Positives: 1

### Analysis

The Isolation Forest correctly flagged the only fraudulent
transaction in the test set, achieving 100% recall.

However, it also incorrectly flagged 34 legitimate
transactions, resulting in low precision of 2.9%.

Because the test set contained only one fraudulent
transaction, these results are insufficient to determine
how reliably the model detects fraud.

The model is better described as an unusual transaction
detector rather than a confirmed fraud detector.

Future improvements could include testing on larger
datasets with more fraud cases and adjusting the
model's anomaly detection threshold.

## 4. Overall Conclusions

The SmartSpend categorization model performs well on
familiar merchants but struggles with unseen brands.

Expanding the training dataset with more diverse real-world
merchants could improve generalization.

The Isolation Forest successfully identified the single
fraudulent transaction in the test set but produced many
false positives.

Future work should focus on improving unseen-brand
accuracy and reducing unnecessary anomaly alerts.