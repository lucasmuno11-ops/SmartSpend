"""
Keyword rules for SmartSpend.

If a description clearly says what it is (RENT, ZELLE, PHARMACY), we trust the
keyword instead of the model. Rows that match a rule get confidence 1.0.
To add a rule, add one line to RULES: (words to look for, category).
"""

import re

import pandas as pd

# Each rule is (regular expression, category). The first rule that matches wins.
RULES = [
    (r"\bZELLE\b", "Transfers"),
    (r"\bVENMO\b", "Transfers"),
    (r"\bCASH ?APP\b", "Transfers"),
    (r"\bAPPLE CASH\b", "Transfers"),
    (r"\bRENT\b(?!-A)(?!.*\bCAR\b)", "Housing"),
    (r"\bUTILIT(Y|IES)\b", "Housing"),
    (r"\bPHARMACY\b", "Healthcare"),
    (r"\bURGENT CARE\b", "Healthcare"),
    (r"\bDENTAL\b", "Healthcare"),
]
COMPILED = [(re.compile(pattern), category) for pattern, category in RULES]


def rule_category(description):
    """Return the category for a description, or None if no rule matches."""
    text = str(description).upper()
    for pattern, category in COMPILED:
        if pattern.search(text):
            return category
    return None


def apply_rules(df):
    """Overwrite category and confidence for rows that match a rule."""
    df = df.copy()
    matched = df["description"].map(rule_category)
    has_rule = matched.notna()
    df.loc[has_rule, "category"] = matched[has_rule]
    df.loc[has_rule, "confidence"] = 1.0
    df.loc[has_rule, "needs_review"] = False
    return df
