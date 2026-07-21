#!/usr/local/bin/python3
"""
risk_parity_allocator.py

Takes a list of ETF holdings (ticker + shares), pulls historical prices,
and computes a risk-parity (equal risk contribution) reallocation --
i.e. a new set of weights where every holding contributes the same
amount of portfolio volatility, rather than the same amount of dollars.

This generally lowers overall portfolio risk vs. a cap-weighted or
dollar-weighted starting point, because it shifts money away from the
most volatile holdings.

Input .xlsx format (holdings_risk_parity.xlsx):
    ticker | shares
    VTI | 120
    BND | 300
    VNQ | 50
    GLD | 40

Usage:
    python3 risk_parity_allocator.py --holdings holdings_risk_parity.xlsx
    python3 risk_parity_allocator.py --holdings holdings_risk_parity.xlsx --lookback-days 504 --output rebalance.xlsx
    python3 risk_parity_allocator.py --holdings holdings_risk_parity.xlsx --method inverse-vol   # simpler approximation

"""

"""
ToDo:
- Add totals row to report for relevant columns (current_value, target_value, shares_to_trade)
- Save as 2 decimals
- Add Shares_to_trade_%
- Bold column names 
- Make input file reading more flexible: i.e. allow, and ignore, extra columns - save the results in same file - different sheet
- BACKTEST
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import yfinance as yf
from scipy.optimize import minimize

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utilities import clean_excel_text

LOOKBACK_DAYS_DEFAULT = 252  # ~1 year of trading days

# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

@dataclass
class Holding:
    ticker: str
    shares: float


def load_holdings(path: str) -> list[Holding]:
    df = clean_excel_text(pd.read_excel(path))
    df.columns = [c.strip().lower() for c in df.columns]
    if "ticker" not in df.columns or "shares" not in df.columns:
        raise ValueError("Excel must have 'ticker' and 'shares' columns")
    holdings = [
        Holding(ticker=str(row.ticker).strip().upper(), shares=float(row.shares)) # type: ignore
        for row in df.itertuples()
    ]
    if not holdings:
        raise ValueError(f"No holdings found in Excel: {path}")
    return holdings


def fetch_prices(tickers: list[str], lookback_days: int) -> pd.DataFrame:
    """Fetch adjusted close prices for all tickers over the lookback window."""
    period_days = max(lookback_days + 30, 60)  # pad for weekends/holidays
    data = yf.download(
        tickers,
        period=f"{period_days}d",
        auto_adjust=True,
        progress=False,
    )

    if data is not None:
        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Close"]
        else:
            # single ticker case: yfinance returns flat columns
            prices = data[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.dropna(how="all").tail(lookback_days) # type: ignore
    missing = [t for t in tickers if t not in prices.columns or prices[t].isna().all()]
    if missing:
        raise ValueError(f"No price data returned for: {', '.join(missing)}")

    prices = prices.ffill().dropna()
    return prices # type: ignore


# --------------------------------------------------------------------------
# Risk calculations
# --------------------------------------------------------------------------

def annualized_cov(prices: pd.DataFrame, trading_days: int = LOOKBACK_DAYS_DEFAULT) -> pd.DataFrame:
    returns = prices.pct_change().dropna()
    return returns.cov() * trading_days


def portfolio_vol(weights: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(weights @ cov @ weights))


def risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Each asset's contribution to total portfolio volatility (sums to portfolio_vol)."""
    port_vol = portfolio_vol(weights, cov)
    if port_vol == 0:
        return np.zeros_like(weights)
    marginal = cov @ weights / port_vol
    return weights * marginal


def inverse_vol_weights(cov: np.ndarray) -> np.ndarray:
    """Fast approximation: weight inversely proportional to each asset's own volatility.
    Ignores correlations -- a reasonable starting point or fallback if scipy isn't available."""
    vols = np.sqrt(np.diag(cov))
    inv = 1.0 / vols
    return inv / inv.sum()


def erc_weights(cov: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """True equal-risk-contribution (risk parity) weights via constrained optimization.
    Every asset ends up contributing the same share of total portfolio volatility."""
    n = cov.shape[0]
    x0 = inverse_vol_weights(cov)

    def objective(w):
        rc = risk_contributions(w, cov)
        target = rc.sum() / n
        return np.sum((rc - target) ** 2)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(1e-6, 1.0) for _ in range(n)]

    result = minimize(
        objective, x0, method="SLSQP", bounds=bounds,
        constraints=constraints, options={"maxiter": 1000, "ftol": tol},
    )
    if not result.success:
        print(f"Warning: optimizer did not fully converge ({result.message}); "
              f"falling back to inverse-vol weights", file=sys.stderr)
        return x0
    w = result.x
    return w / w.sum()


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def build_report(holdings: list[Holding], prices: pd.DataFrame, method: str) -> pd.DataFrame:
    tickers = [h.ticker for h in holdings]
    prices = prices[tickers]  # enforce consistent order
    last_prices = prices.iloc[-1]

    current_values = np.array([h.shares * last_prices[h.ticker] for h in holdings])
    total_value = current_values.sum()
    current_weights = current_values / total_value

    cov = annualized_cov(prices).loc[tickers, tickers].values

    if method == "inverse-vol" :
        new_weights = inverse_vol_weights(cov)
    else:
        new_weights = erc_weights(cov)

    current_shares = np.array([h.shares for h in holdings])
    target_values = new_weights * total_value
    target_shares = target_values / last_prices[tickers].values
    share_deltas = target_shares - np.array([h.shares for h in holdings])
    shares_to_trade_pct = share_deltas / current_shares

    current_rc = risk_contributions(current_weights, cov)
    new_rc = risk_contributions(new_weights, cov)

    df = pd.DataFrame({
        "ticker": tickers,
        "price": last_prices[tickers].values,
        "current_shares": current_shares,
        "current_value": current_values,
        "current_weight_%": current_weights * 100,
        "current_risk_share_%": (current_rc / current_rc.sum()) * 100,
        "target_weight_%": new_weights * 100,
        "target_risk_share_%": (new_rc / new_rc.sum()) * 100,
        "target_value": target_values,
        "target_shares": target_shares,
        "shares_to_trade": share_deltas,
        "shares_to_trade_%": shares_to_trade_pct * 100,
        "action": np.where(share_deltas > 0.005, "BUY",
                   np.where(share_deltas < -0.005, "SELL", "HOLD")),
    })

    port_vol_before = portfolio_vol(current_weights, cov)
    port_vol_after = portfolio_vol(new_weights, cov)

    df.attrs["total_value"] = total_value
    df.attrs["vol_before"] = port_vol_before
    df.attrs["vol_after"] = port_vol_after

    return df


def print_report(df: pd.DataFrame, output_path: str | None = None):
    total_value = df.attrs["total_value"]
    vol_before = df.attrs["vol_before"]
    vol_after = df.attrs["vol_after"]

    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    print(f"\nPortfolio value: ${total_value:,.2f}")
    print(f"Estimated annualized volatility  -  current: {vol_before*100:.2f}%   "
          f"risk-parity: {vol_after*100:.2f}%   "
          f"(Δ {(vol_after-vol_before)*100:+.2f} pts)\n")

    cols = ["ticker", "price", "current_shares", "current_weight_%",
            "current_risk_share_%", "target_weight_%", "target_risk_share_%",
            "target_value", "target_shares", "shares_to_trade", "shares_to_trade_%", "action"]
    cols_with_totals = [ "price", "current_weight_%", "current_risk_share_%", "target_weight_%", "target_risk_share_%",
            "target_value"]
    # Add totals row for columsn in cols_with_totals
    totals_row = {col: df[col].sum() for col in cols_with_totals}
    totals_row["ticker"] = "TOTAL"
    # Add the totals row to the DataFrame
    df = pd.concat([df, pd.DataFrame([totals_row])], ignore_index=True)
    print(df[cols].to_string(index=False))

    # save to file
    if output_path:
        df = df.round(2) # 2 decimal places
        df.to_excel(output_path, index=False, sheet_name="risk_parity_rebalance")
        print(f"Full report written to {output_path}")

    print()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--holdings", help="Excel file with columns: ticker, shares", default="holdings_risk_parity.xlsx")
    parser.add_argument("--lookback-days", type=int, default=LOOKBACK_DAYS_DEFAULT,
                         help="Trading days of price history to use (default: LOOKBACK_DAYS_DEFAULT, ~1yr)")
    parser.add_argument("--method", choices=["erc", "inverse-vol"], default="erc",
                         help="'erc' = true equal-risk-contribution; "
                              "'inverse-vol' = faster, ignores correlations (default: erc)")
    parser.add_argument("--output", default=None,
                         help="Optional path to write the full rebalance report as Excel")
    args = parser.parse_args()
    print(f"args: {args}")

    holdings = load_holdings(args.holdings)
    tickers = [h.ticker for h in holdings]

    print(f"Fetching {args.lookback_days} trading days of price history for: "
          f"{', '.join(tickers)} ...")
    prices = fetch_prices(tickers, args.lookback_days)

    df = build_report(holdings, prices, args.method)
    print_report(df, args.output)


if __name__ == "__main__":
    main()
