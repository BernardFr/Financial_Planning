"""
Helper functions for market_value.py.

This file contains a subset of the functions in BenPlan/src/utils.py.
"""

import os
import re
import sys

import pandas as pd


def get_prog_name() -> str:
    """Get the program name from the path."""
    basename = os.path.basename(sys.argv[0])
    return re.sub("\\.py$", "", basename)


def to_dollar_str(z) -> str:
    """
    Convert a floating number to a string showing a dollar value.
    Prepend with '$', add commas, 2 decimal points, i.e. $xx,xxx.xx
    """
    return z if isinstance(z, str) else f'${float(z):,.2f}'


def clean_text(value):
    """
    Remove non-printable and non-ASCII characters (e.g. Excel's non-breaking spaces, smart quotes)
    from a string, and collapse whitespace. Non-string values pass through unchanged.
    """
    if not isinstance(value, str):
        return value
    value = re.sub(r'[^\x20-\x7E]', ' ', value)
    return re.sub(r'\s+', ' ', value).strip()


def clean_excel_text(df: pd.DataFrame) -> pd.DataFrame:
    """Apply clean_text to the column labels, the index and each string cell of a DataFrame read from Excel."""
    df = df.rename(columns=clean_text)
    df.index = df.index.map(clean_text)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(clean_text)
    return df
