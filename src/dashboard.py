import streamlit as st
import pandas as pd
from pathlib import Path
from anomaly import flag_unusual
from data_cleaning import clean_transactions
import joblib

# Load the saved categorization model
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "categorizer_model.joblib"
VECTORIZER_PATH = ROOT / "models" / "categorizer_vectorizer.joblib"


@st.cache_resource
def load_categorizer():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

# Predict spending categories using the saved model
def predict_categories(df):
    df = df.copy()

    model, vectorizer = load_categorizer()

    # Preserve existing category labels for comparison
    if "category" in df.columns:
        df["actual_category"] = df["category"]

    # Convert descriptions into model inputs
    descriptions = df["description"].fillna("").astype(str)
    X = vectorizer.transform(descriptions)

    # Predict categories and confidence scores
    df["category"] = model.predict(X)
    probabilities = model.predict_proba(X)
    df["confidence"] = probabilities.max(axis=1)

    # Flag predictions below 40% confidence
    df["needs_review"] = df["confidence"] < 0.4

    return df



# Dashboard sidebar
with st.sidebar:
    st.title("💳 SmartSpend")
    st.write("AI-Powered Spending Analysis")

    st.divider()

    st.subheader("Dashboard Features")
    st.write("📋 Transaction Explorer")
    st.write("📊 Spending Analytics")
    st.write("🚨 Unusual Transaction Detection")

    st.divider()

    st.subheader("About")
    st.info(
        "SmartSpend is a student-focused financial "
        "analytics project that uses machine learning "
        "to categorize spending and identify unusual "
        "transactions."
    )

    st.caption("Dashboard Demo | Sample Transaction Data")

st.title("💳 SmartSpend")

st.subheader("AI-Powered Student Spending Dashboard")

st.write(
    "SmartSpend automatically categorizes spending "
    "and identifies unusual transactions."
)

st.success("SmartSpend dashboard is running successfully!")

# Load transactions
DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "sample_transactions.csv"
)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

st.divider()
st.header("Upload Transactions")

uploaded_file = st.file_uploader(
    "Upload a bank transaction CSV",
    type=["csv"]
)

# Load uploaded CSV or sample data
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except (pd.errors.EmptyDataError, pd.errors.ParserError,
            UnicodeDecodeError) as e:
        st.error(f"Unable to read CSV file: {e}")
        st.stop()

    st.success("CSV uploaded successfully!")
else:
    df = load_data()
    st.info("Showing sample transactions. Upload a CSV to use your own data.")

# Check for an empty dataset
if df.empty:
    st.error("The CSV contains no transactions.")
    st.stop()

# Standardize column names
df.columns = df.columns.str.strip().str.lower()

if "amt" in df.columns and "amount" not in df.columns:
    df = df.rename(columns={"amt": "amount"})

if "team_category" in df.columns:
    df["category"] = df["team_category"]


# Clean uploaded bank transactions
if "description" in df.columns:
    df = clean_transactions(
        df.copy(),
        create_sample=False
    )

    if df.empty:
        st.error("No valid transactions remain after cleaning.")
        st.stop()

    if "amount" not in df.columns:
        st.error("Missing required column: amount")
        st.stop()

    # Predict categories using the saved model
    df = predict_categories(df)

else:
    # Handle the existing Kaggle sample dataset
    if "amount" not in df.columns:
        st.error("Missing required column: amount")
        st.stop()

    df["amount"] = pd.to_numeric(
        df["amount"], errors="coerce"
    )

    df = df.dropna(subset=["amount"]).copy()

    if df.empty:
        st.error("No valid transactions remain.")
        st.stop()


# Show successful transaction count
st.caption(f"Loaded {len(df):,} valid transactions.")


# Check that categories are available
if "category" not in df.columns:
    st.error(
        "No spending categories are available. "
        "Please upload a CSV containing transaction descriptions."
    )
    st.stop()


st.divider()

st.header("Transaction Data")

st.write("Explore and filter SmartSpend transactions.")

# Category selection
categories = ["All Categories"] + sorted(
    df["category"].dropna().unique().tolist()
)

selected_category = st.selectbox(
    "Filter transactions by spending category:",
    categories
)

# Apply category filter
if selected_category == "All Categories":
    filtered_df = df.copy()
else:
    filtered_df = df[
        df["category"] == selected_category
    ].copy()

# Display filtered transactions
st.dataframe(
    filtered_df,
    use_container_width=True,
    hide_index=True
)

st.caption(
    f"Showing {len(filtered_df):,} of {len(df):,} transactions"
)

# Spending summary
st.divider()
st.header("Spending Overview")

total_spending = df["amount"].sum()
average_transaction = df["amount"].mean()
largest_transaction = df["amount"].max()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Spending", f"${total_spending:,.2f}")

with col2:
    st.metric("Average Transaction", f"${average_transaction:,.2f}")

with col3:
    st.metric("Largest Transaction", f"${largest_transaction:,.2f}")

# Spending by category
st.divider()
st.header("Spending by Category")

category_spending = (
    df.groupby("category")["amount"]
    .sum()
    .sort_values(ascending=False)
)

st.bar_chart(category_spending)

st.subheader("Category Spending Details")

category_table = category_spending.reset_index()
category_table.columns = ["Category", "Total Spending"]

st.dataframe(
    category_table,
    use_container_width=True,
    hide_index=True
)
# Transaction counts by category
st.divider()
st.header("Transactions by Category")

category_counts = df["category"].value_counts()

st.bar_chart(category_counts)

st.caption(
    "This chart shows the number of transactions "
    "in each spending category."
)

# Unusual Transaction Detection
st.divider()
st.header("🚨 Unusual Transaction Detection")

st.write(
    "SmartSpend uses Isolation Forest to identify "
    "transactions with unusual spending patterns."
)

# Prepare transactions for anomaly detection
anomaly_df = df.copy()


# Use the reusable function from anomaly.py
results_df = flag_unusual(anomaly_df)

# Count unusual transactions
unusual_count = int(results_df["unusual"].sum())

col1, col2 = st.columns(2)

with col1:
    st.metric("Total Transactions", len(results_df))

with col2:
    st.metric("Unusual Transactions", unusual_count)

# Display flagged transactions
st.subheader("Flagged Transactions")

flagged_df = results_df[results_df["unusual"]]

st.dataframe(
    flagged_df,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Flagged transactions are unusual, not necessarily fraudulent."
)
# Transactions Needing Review
st.divider()
st.header("🔎 Transactions Needing Review")

st.write(
    "These transactions have low-confidence category predictions "
    "and may need to be checked manually."
)

# Compare predictions with existing labels
if "actual_category" in df.columns:
    st.divider()
    st.header("📊 Categorization Accuracy Check")

    comparison_df = df.copy()

    comparison_df["correct"] = (
        comparison_df["category"]
        == comparison_df["actual_category"]
    )

    accuracy = comparison_df["correct"].mean()

    st.metric(
        "Accuracy on Uploaded Labeled CSV",
        f"{accuracy:.1%}"
    )

    st.subheader("Prediction Comparison")

    st.dataframe(
        comparison_df[
            [
                "description",
                "actual_category",
                "category",
                "confidence",
                "needs_review",
                "correct"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "This comparison uses the category labels provided "
        "in the uploaded CSV. It is separate from the "
        "unseen-brand evaluation."
    )


# Check whether the categorizer provided review flags
if "needs_review" in df.columns:

    # Keep only transactions marked for review
    review_df = df[df["needs_review"].fillna(False).astype(bool)]

    # Show number of transactions requiring review
    st.metric("Transactions Needing Review", len(review_df))

    if review_df.empty:
        st.success("No transactions currently need category review.")
    else:
        st.dataframe(
            review_df,
            use_container_width=True,
            hide_index=True
        )

else:
    st.info(
        "Category review results will appear here once "
        "the categorization model is connected."
    )
