#! /usr/local/bin/python3

"""
Computes the same "Risk Statistics" Yahoo Finance shows on a fund/ETF's
/risk page (Alpha, Beta, Mean Annual Return, R-squared, Standard Deviation,
Sharpe Ratio, Treynor Ratio) -- but over the *longest* history available
for the ticker, rather than Yahoo's fixed 3/5/10-year windows.

Uses monthly returns (the convention Yahoo/Morningstar use for these
stats) regressed against a benchmark to derive alpha/beta/R-squared.

Usage:
    python risk_stats.py VUG
    python risk_stats.py VUG --benchmark ^GSPC
    python risk_stats.py VUG --benchmark ^GSPC --rf 0.03
    python risk_stats.py VUG BND VNQ GLD --benchmark ^GSPC --output risk_stats.csv

Dependencies:
    pip install yfinance pandas numpy scipy
"""

import argparse
import sys
from typing import cast

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats as sp_stats


# --------------------------------------------------------------------------
# Data fetching
# --------------------------------------------------------------------------

def fetch_monthly_prices(ticker: str) -> pd.Series:
    """Full available history, resampled to month-end adjusted close."""
    data = yf.download(ticker, period="max", interval="1mo",
                        auto_adjust=True, progress=False)
    if data is None or data.empty:
        raise ValueError(f"No data returned for {ticker}")
    close = data.get("Close")
    if close is None:
        raise ValueError(f"'Close' column missing for {ticker}")

    if isinstance(close, pd.DataFrame):  # yfinance sometimes nests columns
        first_col = close.columns[0]
        close_series = close[ticker] if ticker in close.columns else close[first_col]
    else:
        close_series = close

    return close_series.dropna()


def fetch_risk_free_rate(rf_arg) -> float:
    """
    Annual risk-free rate as a decimal (e.g. 0.03 for 3%).
    If rf_arg is a number, use it directly. If 'auto', pull the latest
    13-week T-bill yield (^IRX) as a proxy. Falls back to 0.0 on failure.
    """
    if rf_arg is not None:
        try:
            return float(rf_arg)
        except ValueError:
            pass  # fall through to 'auto' handling below

    try:
        irx_data = yf.download(
            "^IRX", period="5d", interval="1d", auto_adjust=True, progress=False
        )
        if irx_data is None or irx_data.empty:
            raise ValueError("No ^IRX data returned")

        irx_close = irx_data.get("Close")
        if irx_close is None:
            raise ValueError("'Close' column missing in ^IRX data")

        if isinstance(irx_close, pd.DataFrame):
            first_col = irx_close.columns[0]
            irx_close = irx_close["^IRX"] if "^IRX" in irx_close.columns else irx_close[first_col]

        latest = float(irx_close.dropna().iloc[-1])
        return latest / 100.0  # ^IRX quoted as a percentage
    except Exception:
        print("Warning: could not fetch ^IRX risk-free rate, using 0.0",
              file=sys.stderr)
        return 0.0


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def compute_risk_stats(ticker: str, benchmark: str, rf_annual: float) -> dict:
    fund_px = fetch_monthly_prices(ticker)
    bench_px = fetch_monthly_prices(benchmark)

    fund_ret = fund_px.pct_change().dropna()
    bench_ret = bench_px.pct_change().dropna()

    df = pd.concat([fund_ret, bench_ret], axis=1, join="inner")
    df.columns = ["fund", "bench"]
    if len(df) < 6:
        raise ValueError(
            f"Not enough overlapping history between {ticker} and {benchmark} "
            f"({len(df)} months) to compute meaningful statistics"
        )

    rf_monthly = (1 + rf_annual) ** (1 / 12) - 1

    fund_excess = df["fund"] - rf_monthly
    bench_excess = df["bench"] - rf_monthly

    # Linear regression: fund_excess = alpha_monthly + beta * bench_excess
    regression = sp_stats.linregress(bench_excess.values, fund_excess.values)
    beta, alpha_monthly, r_value, _, _ = cast(
        tuple[float, float, float, float, float], regression
    )
    r_squared = r_value ** 2
    alpha_annual = alpha_monthly * 12  # simple annualization, matches Yahoo convention

    mean_annual_return = df["fund"].mean() * 12
    std_annual = df["fund"].std(ddof=1) * np.sqrt(12)

    sharpe = (mean_annual_return - rf_annual) / std_annual if std_annual != 0 else np.nan
    treynor = (mean_annual_return - rf_annual) / beta if beta != 0 else np.nan

    return {
        "ticker": ticker,
        "benchmark": benchmark,
        "start": df.index[0].strftime("%Y-%m-%d"),
        "end": df.index[-1].strftime("%Y-%m-%d"),
        "months": len(df),
        "years": round(len(df) / 12, 1),
        "risk_free_rate_%": rf_annual * 100,
        "alpha_%": alpha_annual * 100,
        "beta": beta,
        "mean_annual_return_%": mean_annual_return * 100,
        "r_squared": r_squared * 100,
        "std_dev_%": std_annual * 100,
        "sharpe_ratio": sharpe,
        "treynor_ratio_%": treynor * 100,
    }


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def print_stats(stats: dict):
    print(f"\n{stats['ticker']} vs {stats['benchmark']}  "
          f"({stats['start']} to {stats['end']}, "
          f"{stats['months']} months / ~{stats['years']} years)")
    print(f"{'Risk-free rate:':28}{stats['risk_free_rate_%']:.2f}%")
    print(f"{'Alpha (annualized):':28}{stats['alpha_%']:.2f}%")
    print(f"{'Beta:':28}{stats['beta']:.3f}")
    print(f"{'Mean Annual Return:':28}{stats['mean_annual_return_%']:.2f}%")
    print(f"{'R-squared:':28}{stats['r_squared']:.2f}")
    print(f"{'Standard Deviation:':28}{stats['std_dev_%']:.2f}%")
    print(f"{'Sharpe Ratio:':28}{stats['sharpe_ratio']:.3f}")
    print(f"{'Treynor Ratio:':28}{stats['treynor_ratio_%']:.2f}%")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tickers", nargs="+", help="One or more ETF/fund tickers")
    parser.add_argument("--benchmark", default="^GSPC",
                         help="Benchmark ticker (default: ^GSPC, the S&P 500 index)")
    parser.add_argument("--rf", default=None,
                         help="Annual risk-free rate as a decimal (e.g. 0.03). "
                              "If omitted, fetches latest 13-week T-bill yield (^IRX).")
    parser.add_argument("--output", default=None,
                         help="Optional path to write all results as CSV")
    args = parser.parse_args()

    rf_annual = fetch_risk_free_rate(args.rf)

    rows = []
    for ticker in args.tickers:
        try:
            stats = compute_risk_stats(ticker.upper(), args.benchmark.upper(), rf_annual)
            print_stats(stats)
            rows.append(stats)
        except Exception as e:
            print(f"\n{ticker}: failed -- {e}", file=sys.stderr)

    if args.output and rows:
        pd.DataFrame(rows).to_csv(args.output, index=False)
        print(f"\nFull results written to {args.output}")


if __name__ == "__main__":
    main()
