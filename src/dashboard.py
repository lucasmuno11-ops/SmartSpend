import streamlit as st
import pandas as pd
from pathlib import Path
from anomaly import flag_unusual
from data_cleaning import clean_transactions

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


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("CSV uploaded successfully!")
else:
    df = load_data()
    st.info("Showing sample transactions. Upload a CSV to use your own data.")

# Standardize Kaggle column names
if "amt" in df.columns and "amount" not in df.columns:
    df = df.rename(columns={"amt": "amount"})

if "team_category" in df.columns:
    df["category"] = df["team_category"]

# Clean uploaded transaction descriptions
if "description" in df.columns:
    df = clean_transactions(df)

# Check the columns required for the next pipeline steps
if "amount" not in df.columns:
    st.error("The CSV must contain an amount column.")
    st.stop()

df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
df = df.dropna(subset=["amount"]).copy()

st.caption(f"Loaded {len(df):,} valid transactions.")

# Handle uploaded files without categories
if "category" not in df.columns:
    st.warning(
        "This bank CSV has not been categorized yet. "
        "Waiting for the categorization model to be connected."
    )
    st.dataframe(df, use_container_width=True, hide_index=True)
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
