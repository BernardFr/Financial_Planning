# Holdings Cross-Correlation

## Objective: 

Compute the mean returns and standard deviation, as well as the cross-correlation, of a series of assets - read from a file

## Input file

 has the format: Positions_YYYY_MM_DD.xlsx - where YYYY_MM_DD represent the creation date of the file. We need the most recent file in the ./Data folder

The asset tickers are in the first column "Ticker"

## Output file

has the format Portfolio_Stats_YYYY_MM_DD.xlsx. Contains 2 tabs: "Stats" has the returns and standard Deviation, by Ticker. "Corr" has the cross-correlation matrix - rows and columns are Tickers

## Computation

For each ticker, the daily Adjusted Close ('Adj Close') prices will be dowloaded from Yahoo Finance for the past 10 years-

Compute the daily log return and save date, adjusted close and daily log return in a JSON file in the Cache folder. Each JSON file will have the format "ticker_Creation_Date". 

The program reads from the Cache. The Cache is only refreshed if the computation is run 1 month or more later than the month of "Creation_Date"

The mean return is the monthly CAGR over the 10 years

The standard deviation is computed on the monthly close prices

The cross correlation is also computed on monthly close prices



## Config File

The configuration file is a TOML file  whose name has the same root as the script. The `.toml` file contains: 

* Prefix string for the Positions file: "Positions_"
* Location of input folder: "./Data"
* Date format for both input and output file: "%Y_%m\_%d"
* Location of cache folder: "./Data/Cross_Corr_Cache"





------

# From Claude

## what is the best way to compute the average return of a given stock (equity) using 10 years of daily close data

Use **adjusted close** prices and work with **log returns**. Which average you want depends on what you'll use it for. The two common ones differ noticeably for a volatile stock, and the gap matters for Monte Carlo inputs.

**1. Use adjusted close, not raw close**

Raw closes ignore dividends and splits. They understate total return by the dividend yield, about 1–2%/yr for many stocks, and splits show up as fake crashes. Use `Adj Close` from yfinance, or a total-return series.

**2. Pick the average that fits the purpose**

- **Geometric (CAGR)** is the actual compound growth rate you'd have earned. Use it to report historical performance or to project a buy-and-hold outcome. It only depends on the first and last prices: `(P_end / P_start)^(1/years) − 1`.
- **Arithmetic annual mean** is the expected one-year return. Use it as the `mean` input when you draw random annual returns, as in your `montecarlo_simulation_class`. Compounding the draws in the simulation automatically produces the lower geometric outcome. If you feed it the CAGR instead, you double-count volatility drag and the simulation comes out too pessimistic.

They're linked by roughly **geometric ≈ arithmetic − σ²/2**. For a stock with 30% volatility, that's a gap of about 4.5%/yr.

**3. How to compute it**

```python
import numpy as np, pandas as pd

px = df["Adj Close"].dropna()              # ~2,520 trading days
logr = np.log(px / px.shift(1)).dropna()   # daily log returns

TD = 252
mu_log  = logr.mean() * TD                 # annualized log (continuously compounded) return
sigma   = logr.std(ddof=1) * np.sqrt(TD)   # annualized volatility

years   = (px.index[-1] - px.index[0]).days / 365.25
cagr    = (px.iloc[-1] / px.iloc[0]) ** (1 / years) - 1   # geometric; ≈ exp(mu_log) - 1
arith   = np.exp(mu_log + sigma**2 / 2) - 1               # expected 1-yr simple return (lognormal)

# Empirical cross-check: mean of calendar-year returns
annual  = px.resample("YE").last().pct_change().dropna()
arith_empirical = annual.mean()
```

Some notes on this:

- **Log returns** add up over time, and annualizing them by ×252 is exact. Annualizing simple daily returns as `(1 + mean)^252 − 1` has a bias, so avoid it.
- **Annualize volatility with √252**, which assumes daily returns are roughly independent.
- **Don't trust the mean too much.** With 10 years of data, the standard error of the annual mean is σ/√10 ≈ 0.32σ. For a stock with 30% volatility, your estimate is ±~9.5%/yr at one standard error. Daily data pins down volatility and correlation very well, but the mean barely at all, because more frequent sampling doesn't help with the mean. That's why planning tools usually take expected returns from forward-looking capital market assumptions (JPM, Vanguard, Morningstar) and use historical data only for volatility and correlations.
- **Cross-asset correlation:** compute it on daily or weekly log returns of the aligned price series, using `logr_df.corr()`. Weekly returns help avoid timing mismatches between assets with different closing times.

**Recommendation for your simulation:** feed it the arithmetic annual mean (or the forward-looking estimate) plus annualized σ from daily log returns. Report CAGR as the historical performance number.



## what is the best way to compute the variance of a given stock (equity) using 10 years of daily close data

Use daily log returns from adjusted close, take the sample variance, and multiply by 252 to annualize. Unlike the mean, 10 years of daily data estimates variance very precisely. The real questions are whether the full-period number is the right one and whether it matches the return definition your simulation uses.

**1. Core computation**

```python
import numpy as np

px   = df["Adj Close"].dropna()
logr = np.log(px / px.shift(1)).dropna()

TD = 252
var_daily = logr.var(ddof=1)          # sample variance (ddof=1, not pandas/numpy default mismatch)
var_ann   = var_daily * TD            # annualized variance of log returns
sigma_ann = np.sqrt(var_ann)          # annualized volatility
```

- **Use adjusted close.** Splits in raw closes create huge fake returns that dominate the variance.
- **Use `ddof=1`.** pandas `.var()` already defaults to it, but NumPy's `np.var` defaults to `ddof=0`. With about 2,500 observations the difference is negligible, but being explicit avoids mixing the two.
- **Annualize with ×252.** This assumes daily returns are uncorrelated. Weekends and holidays are already handled because you're counting trading days.

**2. Precision: this is where daily data pays off**

The standard error of σ is roughly σ/√(2n). With n ≈ 2,520, that's about ±1.4% of σ, so a 25% volatility estimate is good to about ±0.35 percentage points. Compare that with the mean, which was uncertain by ±8 points. Sampling more often sharpens variance but not the mean. That's why historical data is a good source for σ and correlations, and a poor one for expected return.

**3. Check the ×252 assumption**

If daily returns are correlated (mean reversion or momentum, or stale prices in thinly traded names), annualizing daily variance gives a different answer than measuring at longer horizons. Compare the results:

```python
def ann_var(px, rule, periods):
    r = np.log(px.resample(rule).last()).diff().dropna()
    return r.var(ddof=1) * periods

ratio_weekly  = ann_var(px, "W-FRI", 52) / var_ann
ratio_monthly = ann_var(px, "ME", 12)    / var_ann
```

Ratios near 1.0 mean ×252 is fine. A ratio well below 1 means daily returns overstate long-horizon risk. For a planning model with annual steps, monthly or weekly variance is arguably the more honest input. Monthly data gives only about 120 observations, so expect noise of about ±6% of σ.

**4. Full-period vs. recent volatility**

Volatility clusters: 2020 alone can inflate a 10-year σ. Choose based on the horizon:

- **Long-horizon planning (your Monte Carlo):** use the full 10-year unconditional variance. It averages calm and stressed regimes, and that's what you want for decades-long paths.

- **Current risk** (position sizing, near-term VaR): use EWMA, the RiskMetrics method with λ = 0.94 for daily data:

  ```python
  ewma_var_ann = (logr**2).ewm(alpha=1-0.94, adjust=False).mean().iloc[-1] * TD
  ```

- **Rolling 1-year σ** is useful as a diagnostic plot (`logr.rolling(252).std() * np.sqrt(TD)`) to see how stable the number really is.

**5. Match the variance to what you simulate**

This is the same issue as arithmetic vs. geometric means. σ² from log returns is the variance of continuously compounded returns. If your `ArrayRandGen` draws simple annual returns from a normal distribution, convert to the simple-return moments:

```python
mu_log   = logr.mean() * TD
m_simple = np.exp(mu_log + var_ann/2) - 1                          # arithmetic mean
v_simple = (np.exp(var_ann) - 1) * np.exp(2*mu_log + var_ann)      # variance of simple returns
s_simple = np.sqrt(v_simple)
```

At 20% volatility the two σ's differ by a point or two, and the gap widens for volatile stocks. Pairing the arithmetic mean with the log σ is a common, quiet bug. The cleaner approach is to simulate in log space, drawing N(μ_log, σ_log) and applying `exp(x) − 1`. That's internally consistent and can never produce a return below −100%. It would also replace the ±40% floor and ceiling in `test_ror.py`.

**6. Fat tails**

Daily equity returns have excess kurtosis, often 5–10. Variance captures their spread but not their crash risk. If tail outcomes matter (ruin probability in retirement planning), check `logr.kurt()`. You can also draw from a Student-t (`scipy.stats.t`, df ≈ 4–6, scaled to match σ) instead of a normal.

**For multiple assets:** build the covariance matrix from the same aligned daily log returns, `logr_df.cov() * 252`. For many holdings, Ledoit-Wolf shrinkage (`sklearn.covariance.LedoitWolf`) gives a better-conditioned matrix that's safer to Cholesky-decompose for correlated draws.



## what is the best way to compute the cross-correlation of a given set of stocks (equity) using 10 years of daily close data

Compute Pearson correlation on daily log returns from adjusted close, after aligning all the stocks to common trading dates. Then check two things: whether daily data distorts the correlations (asynchronous closes), and whether the resulting matrix is valid for Cholesky in your Monte Carlo.

**1. Core computation**

```python
import numpy as np, pandas as pd

# prices: DataFrame, one column per ticker, index = dates, Adj Close values
prices = prices.sort_index()
logr = np.log(prices / prices.shift(1))

common = logr.dropna(how="any")          # only dates where every stock traded
corr   = common.corr(method="pearson")
cov    = common.cov() * 252              # annualized covariance, if you need it
```

- **Correlate returns, not prices.** Price levels are trending series, and two unrelated stocks that both went up over 10 years will show a price correlation near 0.9. That's a spurious result.
- **Use adjusted close**, for the same reason as before: a split shows up as a fake −50% day and can swamp a correlation.
- **Align dates with `dropna(how="any")`**, which keeps only dates where every stock has a return. Don't forward-fill missing prices: a filled day is a fake 0% return, which pushes correlations toward zero.

**2. Unequal history lengths: common window vs. pairwise**

If one holding IPO'd in 2021, the common window shrinks to about 5 years for every pair. The alternative is `logr.corr()` without the `dropna`, which uses all available overlapping data for each pair separately. That keeps more data, but each entry comes from a different period, so **the matrix can stop being positive semi-definite** and Cholesky fails. If you use pairwise estimates, repair the matrix afterwards (see step 5).

**3. Asynchronous closes: the main pitfall with daily data**

If your holdings include international ETFs (Europe, Japan, EM), their closes happen hours before the US close. News that moves the US at 2 pm moves Tokyo the next day, so daily correlations come out **too low**, sometimes dramatically: US vs. Japan can drop from about 0.6 to about 0.3. The same downward bias affects thinly traded small caps and bond funds (the Epps effect).

The fix is to measure correlation at a lower frequency:

```python
wk  = np.log(prices.resample("W-FRI").last()).diff().dropna()
corr_weekly = wk.corr()

print((corr_weekly - corr).round(2))   # large positive gaps = asynchronicity
```

With about 520 weekly observations, precision is still good. For a mixed US/international portfolio feeding an annual-step simulation, **weekly correlation is usually the better input.** Monthly data (about 120 points) works too but is noisier. If you have to stay daily, you can add the lead-lag terms to the correlation (the Dimson or Newey-West style adjustment), but switching to weekly is simpler.

**4. How precise and how stable the estimates are**

The sampling error is small: SE(ρ) ≈ (1 − ρ²)/√n, which is about ±0.015 for daily data and ±0.04 for weekly. The bigger problem is that **correlations aren't stable over time**. They rise sharply in crises (2008, March 2020), exactly when diversification matters most. Check this:

```python
roll = logr["SPY"].rolling(252).corr(logr["EFA"])   # 1-yr rolling correlation for one pair
```

For long-horizon planning, the full-period correlation is a reasonable default. If you want to stress-test, run the simulation a second time with correlations computed only on high-volatility days, or with all off-diagonal values pushed upward.

**5. Make the matrix usable for the simulation**

For correlated draws you need a correlation matrix that is positive definite and not too noisy. With N assets you estimate N(N−1)/2 pairs, so noise grows quickly with portfolio size.

```python
from sklearn.covariance import LedoitWolf

lw = LedoitWolf().fit(common.values)                 # shrinks toward a structured target
cov_lw  = pd.DataFrame(lw.covariance_ * 252, index=common.columns, columns=common.columns)
d       = np.sqrt(np.diag(cov_lw))
corr_lw = cov_lw / np.outer(d, d)

# Repair a matrix that isn't PSD (e.g., after pairwise estimation) by clipping eigenvalues
def nearest_psd_corr(C, eps=1e-8):
    w, V = np.linalg.eigh(C)
    C2 = V @ np.diag(np.clip(w, eps, None)) @ V.T
    d = np.sqrt(np.diag(C2))
    return C2 / np.outer(d, d)

L = np.linalg.cholesky(corr_lw)                      # fails loudly if not PD
```

- **Ledoit-Wolf shrinkage** is worth using once you have more than about 10 assets. It reduces extreme pairwise values that are just noise, and it always returns a well-conditioned matrix.
- **Clip eigenvalues** if Cholesky fails. This is the simple fix; Higham's algorithm is the rigorous version.
- **Cholesky:** use `L` to correlate the draws. Draw independent standard normals `Z`, then `X = Z @ L.T`, then scale each column by its σ and add its μ. Do this in log space, as in the variance answer.

**6. Rank correlation as a robustness check**

Pearson correlation is sensitive to a few extreme days. Compare it with Spearman:

```python
corr_spearman = common.corr(method="spearman")
```

If a pair differs a lot between the two, a handful of crash days drives that relationship. That's real information about joint tail risk, and a single normal-copula correlation can't represent it.

**For your setup:** use weekly log returns of adjusted close on the common date window, apply Ledoit-Wolf shrinkage, and run Cholesky. That combination should also help with your "seed initialization / very high std" issues in `cross_correlated_rvs`. A near-singular, noisy correlation matrix tends to produce unstable, high-variance draws. Your RNG seeding is a separate issue.

