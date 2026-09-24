# SmartSpend

An AI tool that automatically categorizes student spending and flags unusual transactions — built by a GatorAI (UF) Fall 2026 Applied Projects team.

## What it does
Students export their transaction history as a CSV (from a bank, Venmo, etc.), upload it, and SmartSpend:
1. Automatically sorts each transaction into a spending category (groceries, dining, subscriptions, etc.) using NLP
2. Flags transactions that look unusual (possible errors, duplicate charges, spending spikes)

## Tech stack
- **Python** — pandas (data handling), scikit-learn (ML)
- **Text Classification** — TF-IDF + Naive Bayes / Logistic Regression, to categorize transactions
- **Anomaly Detection** — Isolation Forest, to flag unusual transactions
- **Dashboard (stretch goal)** — Streamlit

## Project structure
```
smartspend/
├── data/         # sample/raw transaction data (not committed if real data)
├── src/          # main source code (data cleaning, model, evaluation)
├── notebooks/    # exploratory notebooks / experiments
├── docs/         # notes, meeting summaries, planning docs
├── requirements.txt
└── README.md
```

## Getting started
1. Clone the repo
2. Create a virtual environment: `python -m venv venv`
3. Activate it:
   - Mac/Linux: `source venv/bin/activate`
   - Windows: `venv\Scripts\activate`
4. Install dependencies: `pip install -r requirements.txt`

## Team
- **Project Lead:** Lucas
- Roles: ML/Model Development, Data Engineering, Evaluation, Dashboard, Product/Data Analyst

## Roadmap (10 weeks)
| Weeks | Focus |
|---|---|
| 1–2 | Data collection & cleaning |
| 3–5 | Build & tune classification model |
| 6–7 | Add anomaly detection |
| 8–9 | Build dashboard, polish |
| 10 | Demo & wrap-up |
