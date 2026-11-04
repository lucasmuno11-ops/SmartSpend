import streamlit as st
import pandas as pd
from pathlib import Path

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

# Load sample transaction data
DATA_PATH = Path(__file__).resolve().parent / "data" / "sample_transactions.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

st.divider()

st.header("Transaction Data")

st.write("Explore and filter SmartSpend transactions.")

# Category selection
categories = ["All Categories"] + sorted(
    df["team_category"].dropna().unique().tolist()
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
        df["team_category"] == selected_category
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

total_spending = df["amt"].sum()
average_transaction = df["amt"].mean()
largest_transaction = df["amt"].max()

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
    df.groupby("team_category")["amt"]
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

category_counts = df["team_category"].value_counts()

st.bar_chart(category_counts)

st.caption(
    "This chart shows the number of transactions "
    "in each spending category."
)

# Unusual Transaction Detection
st.divider()
st.header("🚨 Unusual Transaction Detection")

st.write(
    "SmartSpend uses Isolation Forest to identify transactions "
    "with unusual amounts or spending patterns."
)

from sklearn.ensemble import IsolationForest

# Prepare transaction data
anomaly_data = df[["amt", "team_category"]].copy()

# Convert categories into numbers
anomaly_features = pd.get_dummies(
    anomaly_data,
    columns=["team_category"]
)

# Train Isolation Forest
anomaly_model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)

anomaly_model.fit(anomaly_features)

# Predict anomalies
anomaly_predictions = anomaly_model.predict(anomaly_features)

# Add flags to transaction data
results_df = df.copy()

results_df["Status"] = [
    "⚠️ Needs Review" if prediction == -1 else "✅ Normal"
    for prediction in anomaly_predictions
]

# Count unusual transactions
unusual_count = (results_df["Status"] == "⚠️ Needs Review").sum()

# Display summary
col1, col2 = st.columns(2)

with col1:
    st.metric("Total Transactions", len(results_df))

with col2:
    st.metric("Unusual Transactions", unusual_count)

# Display flagged transactions
st.subheader("Flagged Transactions")

flagged_df = results_df[
    results_df["Status"] == "⚠️ Needs Review"
]

st.dataframe(
    flagged_df,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Flagged transactions are unusual, not necessarily fraudulent."
)


