# SmartSpend Evaluation Results

## 1. Categorization Model

Model: Character TF-IDF (`char_wb`, ngram range 2-5) + Logistic Regression

Dataset: data/real_merchants.csv

### Accuracy Comparison

| Random State | Random Split | Unseen Brand Split |
|---|---|---|
| 42 | 98.98% | 57.36% |
| 7 | 98.98% | 54.41% |
| 99 | 98.81% | 57.26% |
| Average | 98.92% | 56.34% |

### Analysis

The updated dataset contains 9,276 transactions across 1,121 merchant brands. Random-split accuracy averaged 98.92%, while unseen-brand accuracy averaged 56.34%. The character-based model improved unseen-brand accuracy over word TF-IDF + Naive Bayes on all three tested seeds.

## 2. Confusion Matrix

File: docs/confusion_matrix.png

The confusion matrix was generated using the unseen-brand
split with random state 42.

### Main Findings

The updated unseen-brand confusion matrix shows that the strongest categories are Nightlife, Dining, Shopping, Personal Care, Healthcare, and Groceries, which all have relatively high correct counts on the diagonal. Nightlife is especially strong, with 133 correct predictions and very few mix-ups. The model still struggles most with categories that overlap in wording or merchant type. The biggest error is Shopping being predicted as Dining (65 rows). Other common mix-ups include Transportation → Dining (33), Entertainment → Dining (26), Other → Dining (20), Housing → Dining (18), and Healthcare → Other (15). The matrix also shows that Transfers had no test rows in this split, so it could not really be evaluated here. Overall, the model is best on clear merchant types, but it still tends to over-predict Dining and sometimes Other when a merchant is unfamiliar.

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