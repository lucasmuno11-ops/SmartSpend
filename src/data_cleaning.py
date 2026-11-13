"""
Load and clean student bank transaction data from a CSV export.

Handles data cleaning for student bank transactions:
- Standardizes column names (date, description, amount, category)
- Cleans and formats transaction descriptions to uppercase and removes extra whitespace
- Cleans and parses transaction amounts to positive floats (spending)
- Standardizes dates to YYYY-MM-DD format
- Drops rows with missing required values (description, amount)
- Filters out sensitive banking columns (account numbers, SSN, card numbers, balances)
- Removes duplicates and sorts transactions chronologically
- Creates and exports realistic sample student bank transactions matching merchants
  from real_merchants.csv for University of Florida (UF) students without sensitive data.
"""

from datetime import date, timedelta
from pathlib import Path
from typing import Optional, Union

import pandas as pd

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
REAL_MERCHANTS_PATH = DATA_DIR / "real_merchants.csv"
DEFAULT_SAMPLE_PATH = DATA_DIR / "sample_student_transactions.csv"
ALT_SAMPLE_PATH = DATA_DIR / "sample_bank_transactions.csv"

# 30 realistic transactions for a UF college student in Gainesville, FL.
# Every description is an exact match for a merchant entry in data/real_merchants.csv.
UF_STUDENT_TRANSACTIONS = [
    {"description": "THE STANDARD #9809", "amount": 875.00, "category": "Housing"},
    {"description": "PURCHASE GAINESVILLE REGIONAL UTILITIES", "amount": 68.45, "category": "Housing"},
    {"description": "SPECTRUM*JCP23HB5", "amount": 49.99, "category": "Housing"},
    {"description": "PUBLIX SUPERMARKET", "amount": 54.32, "category": "Groceries"},
    {"description": "POS PURCHASE STARBUCKS GAINESVILLE FL", "amount": 6.25, "category": "Dining"},
    {"description": "UF BOOKSTORE", "amount": 74.50, "category": "Education"},
    {"description": "Chick-Fil-A", "amount": 11.40, "category": "Dining"},
    {"description": "GAINESVILLE RTS", "amount": 3.00, "category": "Transportation"},
    {"description": "SPOTIFY P1234567", "amount": 5.99, "category": "Entertainment"},
    {"description": "CHIPOTLE", "amount": 13.75, "category": "Dining"},
    {"description": "TRADER JOE'S", "amount": 38.15, "category": "Groceries"},
    {"description": "UF STUDENT HEALTH CARE CENTER", "amount": 20.00, "category": "Healthcare"},
    {"description": "UBER", "amount": 18.25, "category": "Transportation"},
    {"description": "CHATGPT PLUS", "amount": 20.00, "category": "Education"},
    {"description": "GATOR DINING GAINESVILLE FL", "amount": 12.50, "category": "Dining"},
    {"description": "SQ *THE SWAMP RESTAURANT", "amount": 32.50, "category": "Nightlife"},
    {"description": "INSOMNIA COOKIES GAINESVILLE FL", "amount": 10.50, "category": "Dining"},
    {"description": "SHELL GAINESVILLE FL", "amount": 34.50, "category": "Transportation"},
    {"description": "ALDI #8062", "amount": 26.80, "category": "Groceries"},
    {"description": "QUIZLET*IIB25X1 9/23", "amount": 7.99, "category": "Education"},
    {"description": "SWEETGREEN GAINESVILLE FL", "amount": 14.80, "category": "Dining"},
    {"description": "UF Parking", "amount": 35.00, "category": "Transportation"},
    {"description": "DEBIT CARD PURCHASE NETFLIX", "amount": 15.49, "category": "Entertainment"},
    {"description": "PANDA EXPRESS GAINESVILLE FL", "amount": 11.25, "category": "Dining"},
    {"description": "GOPUFF STORE 8389", "amount": 16.40, "category": "Groceries"},
    {"description": "POS PURCHASE CHEGG 5/4", "amount": 15.95, "category": "Education"},
    {"description": "TACO BELL", "amount": 8.65, "category": "Dining"},
    {"description": "PURCHASE SUNPASS", "amount": 15.00, "category": "Transportation"},
    {"description": "SUBWAY", "amount": 9.95, "category": "Dining"},
    {"description": "DEBIT CARD PURCHASE UF ATHLETICS GAINESVILLE FL", "amount": 25.00, "category": "Entertainment"},
]


def load_transactions(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Load a transaction CSV into a DataFrame."""
    return pd.read_csv(csv_path)


def create_sample_bank_transactions(
    output_path: Optional[Union[str, Path]] = None,
    reference_date: Optional[Union[date, str]] = None,
) -> pd.DataFrame:
    """
    Create a sample bank transactions CSV with 30 realistic transactions in the last month
    for a UF college student, matching merchants from real_merchants.csv without sensitive data.

    Parameters:
    - output_path: File path to save the sample CSV. Defaults to data/sample_student_transactions.csv.
    - reference_date: The ending date for the last month of transactions (defaults to today).

    Returns:
    - pd.DataFrame containing the 30 generated transactions.
    """
    if reference_date is None:
        ref = date.today()
    elif isinstance(reference_date, str):
        ref = pd.to_datetime(reference_date).date()
    else:
        ref = reference_date

    target_path = Path(output_path) if output_path else DEFAULT_SAMPLE_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # 30 transactions distributed across the last 30 days
    records = []
    for i, item in enumerate(UF_STUDENT_TRANSACTIONS):
        txn_date = ref - timedelta(days=29 - i)
        records.append(
            {
                "date": txn_date.strftime("%Y-%m-%d"),
                "description": item["description"],
                "amount": item["amount"],
                "category": item["category"],
            }
        )

    df_sample = pd.DataFrame(records)
    df_sample.to_csv(target_path, index=False)

    # Also keep ALT_SAMPLE_PATH synchronized if different
    if target_path.resolve() != ALT_SAMPLE_PATH.resolve():
        try:
            df_sample.to_csv(ALT_SAMPLE_PATH, index=False)
        except Exception:
            pass

    return df_sample


def clean_transactions(
    data: Union[pd.DataFrame, str, Path, None] = None,
    create_sample: bool = True,
    sample_output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Clean bank transaction data from an inputted CSV path or DataFrame of student data.
    Also creates/ensures a sample bank transactions CSV with realistic UF student transactions.

    Cleaning pipeline:
    1. Standardizes column names (date, description, amount, category).
    2. Strips sensitive columns (account numbers, routing numbers, SSN, card numbers, balances).
    3. Normalizes descriptions: drops missing values, trims extra whitespace, converts to uppercase.
    4. Normalizes amounts: cleans currency symbols, handles negative debits/parentheses, converts to positive float.
    5. Normalizes dates: parses and standardizes to YYYY-MM-DD format.
    6. Normalizes category (if present).
    7. Deduplicates rows and sorts chronologically.

    Parameters:
    - data: A DataFrame, path to a CSV file, or None (defaults to sample transactions).
    - create_sample: If True, ensures the sample bank transactions CSV is created.
    - sample_output_path: Custom output path for sample transactions CSV.

    Returns:
    - pd.DataFrame: The cleaned transaction DataFrame.
    """
    sample_path = Path(sample_output_path) if sample_output_path else DEFAULT_SAMPLE_PATH

    # Create or update sample CSV if requested or if sample file does not exist
    if create_sample or data is None:
        if not sample_path.exists() or create_sample:
            create_sample_bank_transactions(sample_path)

    # Load data from CSV if a path was passed or if data is None
    if data is None:
        df = pd.read_csv(sample_path)
    elif isinstance(data, (str, Path)):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError(f"Unsupported data type for clean_transactions: {type(data)}")

    # 1. Normalize column names (lowercase, stripped)
    column_mapping = {}
    for col in df.columns:
        norm_col = str(col).strip().lower()
        if norm_col in {
            "date", "transaction_date", "trans_date", "posting_date",
            "posted_date", "time", "transaction date", "trans date", "post date"
        }:
            column_mapping[col] = "date"
        elif norm_col in {
            "description", "desc", "merchant", "merchant_name", "merchant name",
            "payee", "name", "memo", "details", "narrative",
            "transaction_description", "transaction description"
        }:
            column_mapping[col] = "description"
        elif norm_col in {
            "amount", "amt", "transaction_amount", "trans_amount",
            "transaction amount", "trans amount", "price", "cost"
        }:
            column_mapping[col] = "amount"
        elif norm_col in {
            "category", "team_category", "team category",
            "spending_category", "spending category", "label", "type"
        }:
            column_mapping[col] = "category"

    df = df.rename(columns=column_mapping)

    # Handle separate debit / credit columns if amount was not directly provided
    if "amount" not in df.columns:
        lower_cols = {str(c).strip().lower(): c for c in df.columns}
        if "debit" in lower_cols:
            debit_series = pd.to_numeric(
                df[lower_cols["debit"]].astype(str).str.replace(r"[^\d.-]", "", regex=True),
                errors="coerce"
            ).fillna(0)
            if "credit" in lower_cols:
                credit_series = pd.to_numeric(
                    df[lower_cols["credit"]].astype(str).str.replace(r"[^\d.-]", "", regex=True),
                    errors="coerce"
                ).fillna(0)
                df["amount"] = debit_series - credit_series
            else:
                df["amount"] = debit_series

    # 2. Drop sensitive personal / account data columns if present in student bank CSV exports
    sensitive_keywords = {
        "account", "acct", "routing", "ssn", "card", "pan", "cvv", "cvc", "pin",
        "balance", "first_name", "last_name", "customer_name", "phone", "email",
        "address", "zip", "zipcode", "postal"
    }
    cols_to_drop = [
        col for col in df.columns
        if any(keyword in str(col).strip().lower() for keyword in sensitive_keywords)
        and col not in {"date", "description", "amount", "category"}
    ]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # 3. Clean description
    if "description" in df.columns:
        df = df.dropna(subset=["description"])
        df["description"] = (
            df["description"]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .str.upper()
        )
        df = df[df["description"] != ""]

    # 4. Clean amount
    if "amount" in df.columns:
        if df["amount"].dtype == object:
            cleaned_amt = (
                df["amount"]
                .astype(str)
                .str.strip()
                .str.replace(r"^\((.*)\)$", r"-\1", regex=True)
                .str.replace(r"[\$,]", "", regex=True)
            )
            df["amount"] = pd.to_numeric(cleaned_amt, errors="coerce")
        else:
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

        df = df.dropna(subset=["amount"])
        # Standardize spending to positive numbers, rounded to 2 decimal places
        df["amount"] = df["amount"].abs().round(2)
        df = df[df["amount"] > 0]

    # 5. Clean date
    if "date" in df.columns:
        parsed_dates = pd.to_datetime(df["date"], errors="coerce")
        valid_dates = parsed_dates.notna()
        df = df[valid_dates].copy()
        df["date"] = parsed_dates[valid_dates].dt.strftime("%Y-%m-%d")
        df = df.sort_values(by="date")

    # 6. Clean category if present
    if "category" in df.columns:
        df["category"] = df["category"].astype(str).str.strip()
        df["category"] = df["category"].replace({"nan": "Other", "": "Other"}).fillna("Other")

    # 7. Deduplicate and reorder columns
    primary_cols = [c for c in ["date", "description", "amount"] if c in df.columns]
    if primary_cols:
        df = df.drop_duplicates(subset=primary_cols)
    else:
        df = df.drop_duplicates()

    ordered_cols = [c for c in ["date", "description", "amount", "category"] if c in df.columns]
    other_cols = [c for c in df.columns if c not in ordered_cols]
    df = df[ordered_cols + other_cols].reset_index(drop=True)

    return df


if __name__ == "__main__":
    # Create sample student transactions CSV and run cleaning
    sample_file = DEFAULT_SAMPLE_PATH
    create_sample_bank_transactions(sample_file)
    print(f"Created sample bank transactions CSV at: {sample_file}")

    cleaned_df = clean_transactions(sample_file)
    print(f"\nCleaned {len(cleaned_df)} transactions successfully:")
    print(cleaned_df.to_string())
