"""
VaR & Expected Shortfall backtest: historical vs. normal vs. EWMA vs. GARCH-t.

Question: does assuming normally distributed returns underestimate risk,
especially in stress periods (2020, 2022)? And does a model that reacts faster
to volatility (EWMA), or one that also has fat tails (GARCH-t), fix the problem?

Run:
    python var_backtest.py          # real data (needs internet + yfinance)
    python var_backtest.py --demo   # synthetic fat-tailed data, works offline
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
TICKER = "SPY"          # S&P 500 ETF: liquid, long history, standard equity benchmark
START = "2010-01-01"
END = "2025-12-31"
CONFIDENCE = 0.99       # 99% one-day VaR (the Basel market-risk standard)
WINDOW = 250            # ~1 trading year of history for each forecast
LAMBDA = 0.94           # EWMA decay (RiskMetrics standard): weight on yesterday's variance
GARCH_REFIT = 21        # re-estimate GARCH parameters every ~1 month (daily refits are slow)
GARCH_MAX_HISTORY = 1000  # GARCH needs more data than 250 days to estimate 4 parameters well


MODELS = ("hist", "norm", "ewma", "garch")


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
def load_returns(ticker: str, start: str, end: str) -> pd.Series:
    """Adjusted close prices -> daily simple returns."""
    import yfinance as yf  # imported here so --demo works without yfinance

    prices = yf.download(ticker, start=start, end=end,
                         auto_adjust=True, progress=False)["Close"]
    if isinstance(prices, pd.DataFrame):          # newer yfinance returns a DataFrame
        prices = prices.iloc[:, 0]
    return prices.pct_change().dropna()


def demo_returns(n_days: int = 4000, seed: int = 42) -> pd.Series:
    """Synthetic returns with fat tails (Student-t, 3 d.o.f.) and a calm/stress regime switch."""
    rng = np.random.default_rng(seed)
    vol = np.where((np.arange(n_days) // 500) % 4 == 3, 0.025, 0.009)  # every 4th block is 'stress'
    t_draws = rng.standard_t(df=3, size=n_days) / np.sqrt(3)
    dates = pd.bdate_range("2010-01-01", periods=n_days)
    return pd.Series(vol * t_draws, index=dates, name="demo")


# ---------------------------------------------------------------------------
# 2. Risk models. VaR and ES are reported as positive numbers (loss size)
# ---------------------------------------------------------------------------
def historical_var_es(past: np.ndarray, alpha: float) -> tuple[float, float]:
    """Historical simulation: empirical quantile, no distributional assumption."""
    cutoff = np.quantile(past, 1 - alpha)
    var = -cutoff
    es = -past[past <= cutoff].mean()
    return var, es


def normal_var_es(past: np.ndarray, alpha: float) -> tuple[float, float]:
    """Normal VaR/ES, closed form."""
    mu, sigma = past.mean(), past.std(ddof=1)
    z = stats.norm.ppf(1 - alpha)
    var = -(mu + z * sigma)
    es = -(mu - sigma * stats.norm.pdf(z) / (1 - alpha))
    return var, es


def ewma_var_es(returns: np.ndarray, window: int, alpha: float, lam: float):
    """
    EWMA (RiskMetrics), zero mean:
        sigma2_t = lam * sigma2_{t-1} + (1 - lam) * r_{t-1}^2
    Output aligned with rolling_forecasts.
    """
    z = stats.norm.ppf(1 - alpha)
    sigma2 = returns[:window].var(ddof=1)        # initialise with first window
    vars_, ess = [], []
    for t in range(window, len(returns)):
        sigma2 = lam * sigma2 + (1 - lam) * returns[t - 1] ** 2
        sigma = np.sqrt(sigma2)
        vars_.append(-z * sigma)
        ess.append(sigma * stats.norm.pdf(z) / (1 - alpha))
    return np.array(vars_), np.array(ess)


def garch_t_var_es(returns: np.ndarray, window: int, alpha: float):
    """
    GARCH(1,1) with Student-t errors:
        sigma2_t = omega + a * r_{t-1}^2 + b * sigma2_{t-1}
    (EWMA = special case omega = 0, a = 1 - lambda, b = lambda.)
    Parameters refitted every GARCH_REFIT days on past data only.
    """
    from arch import arch_model

    r100 = returns * 100                          # arch optimiser is more stable in %
    vars_, ess = [], []
    params = None
    sigma2 = None
    for i, t in enumerate(range(window, len(returns))):
        if i % GARCH_REFIT == 0:
            past = r100[max(0, t - GARCH_MAX_HISTORY):t]
            res = arch_model(past, mean="Zero", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
            params = res.params                   # omega, alpha[1], beta[1], nu
            sigma2 = res.conditional_volatility[-1] ** 2   # variance on day t-1
        omega, a, b, nu = params["omega"], params["alpha[1]"], params["beta[1]"], params["nu"]
        sigma2 = omega + a * r100[t - 1] ** 2 + b * sigma2  # forecast for day t
        sigma = np.sqrt(sigma2) / 100

        # standardised t (unit variance)
        scale = np.sqrt((nu - 2) / nu)
        q = stats.t.ppf(1 - alpha, nu)
        vars_.append(-sigma * q * scale)
        es_std = stats.t.pdf(q, nu) / (1 - alpha) * (nu + q ** 2) / (nu - 1)
        ess.append(sigma * es_std * scale)
    return np.array(vars_), np.array(ess)


def rolling_forecasts(returns: pd.Series, window: int, alpha: float) -> pd.DataFrame:
    """VaR/ES for day t from days t-window ... t-1 (no look-ahead)."""
    values = returns.to_numpy()
    rows = []
    for t in range(window, len(values)):
        past = values[t - window:t]
        h_var, h_es = historical_var_es(past, alpha)
        n_var, n_es = normal_var_es(past, alpha)
        rows.append((returns.index[t], values[t], h_var, h_es, n_var, n_es))
    cols = ["date", "return", "hist_var", "hist_es", "norm_var", "norm_es"]
    fc = pd.DataFrame(rows, columns=cols).set_index("date")
    fc["ewma_var"], fc["ewma_es"] = ewma_var_es(values, window, alpha, LAMBDA)
    fc["garch_var"], fc["garch_es"] = garch_t_var_es(values, window, alpha)
    return fc


# ---------------------------------------------------------------------------
# 3. Backtest
# ---------------------------------------------------------------------------
def kupiec_test(n_obs: int, n_exceptions: int, alpha: float) -> float:
    """
    Kupiec POF test. H0: exception rate = 1 - alpha. Returns p-value.
    """
    p = 1 - alpha
    x, n = n_exceptions, n_obs
    p_hat = x / n
    ll_null = (n - x) * np.log(1 - p) + x * np.log(p)
    ll_alt = 0.0 if x in (0, n) else (n - x) * np.log(1 - p_hat) + x * np.log(p_hat)
    lr = -2 * (ll_null - ll_alt)                  # ~ chi-squared with 1 d.o.f. under H0
    return 1 - stats.chi2.cdf(lr, df=1)


def _xlogy(x: float, y: float) -> float:
    """x * log(y) with 0 * log(0) = 0."""
    return 0.0 if x == 0 else x * np.log(y)


def christoffersen_test(breach: np.ndarray) -> float:
    """
    Christoffersen independence test.
    H0: P(breach | breach yesterday) = P(breach | no breach yesterday). Returns p-value.
    """
    prev, curr = breach[:-1], breach[1:]
    n00 = np.sum(~prev & ~curr)
    n01 = np.sum(~prev & curr)
    n10 = np.sum(prev & ~curr)
    n11 = np.sum(prev & curr)
    pi0 = n01 / max(n00 + n01, 1)
    pi1 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll_null = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    ll_alt = (_xlogy(n00, 1 - pi0) + _xlogy(n01, pi0)
              + _xlogy(n10, 1 - pi1) + _xlogy(n11, pi1))
    lr = -2 * (ll_null - ll_alt)
    return 1 - stats.chi2.cdf(lr, df=1)


def summarise(fc: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Exception = loss > VaR forecast."""
    out = {}
    for model in MODELS:
        breach = -fc["return"] > fc[f"{model}_var"]
        n, x = len(fc), int(breach.sum())
        out[model] = {
            "expected exceptions": round(n * (1 - alpha), 1),
            "actual exceptions": x,
            "exception rate %": round(100 * x / n, 2),
            "Kupiec p-value": round(kupiec_test(n, x, alpha), 4),
            "Christoffersen p-value": round(christoffersen_test(breach.to_numpy()), 4),
            # ES check: realised loss vs ES forecast on breach days
            "avg loss on breach %": round(-100 * fc.loc[breach, "return"].mean(), 2),
            "avg ES on breach %": round(100 * fc.loc[breach, f"{model}_es"].mean(), 2),
        }
    return pd.DataFrame(out)


def exceptions_by_year(fc: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        model: (-fc["return"] > fc[f"{model}_var"]).groupby(fc.index.year).sum()
        for model in MODELS
    })


# ---------------------------------------------------------------------------
# 4. Plot
# ---------------------------------------------------------------------------
def plot(fc: pd.DataFrame, title: str, path: str = "var_backtest.png") -> None:
    """Full period + zoom on the COVID crash."""
    colors = {"hist": "#1f77b4", "norm": "#ff7f0e", "ewma": "#2ca02c", "garch": "#9467bd"}
    labels = {"hist": "historical", "norm": "normal", "ewma": "EWMA", "garch": "GARCH-t"}
    loss = (-fc["return"] * 100).clip(lower=0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[1, 1.2])

    # --- top: full period ---
    ax1.bar(fc.index, loss, width=1.5, color="lightgrey", label="daily loss")
    for m in MODELS:
        ax1.plot(fc.index, fc[f"{m}_var"] * 100, lw=0.9, color=colors[m], label=f"{labels[m]} 99% VaR")
    ax1.set_title(title)
    ax1.set_ylabel("loss, %")
    ax1.legend(loc="upper left", ncol=5, fontsize=8)

    # --- bottom: zoom on Feb-Jun 2020 ---
    z = fc.loc["2020-02-01":"2020-06-30"]
    if len(z):
        zloss = (-z["return"] * 100).clip(lower=0)
        ax2.bar(z.index, zloss, width=0.8, color="lightgrey", label="daily loss")
        for m in MODELS:
            ax2.plot(z.index, z[f"{m}_var"] * 100, lw=1.5, color=colors[m], label=labels[m])
        for m, marker in (("norm", "x"), ("garch", "o")):
            hit = -z["return"] > z[f"{m}_var"]
            ax2.scatter(z.index[hit], zloss[hit], marker=marker, s=40, color=colors[m],
                        zorder=3, label=f"breach of {labels[m]}")
        ax2.set_title("Zoom: COVID crash, Feb-Jun 2020")
        ax2.set_ylabel("loss, %")
        ax2.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"\nChart saved to {path}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo = "--demo" in sys.argv
    returns = demo_returns() if demo else load_returns(TICKER, START, END)
    name = "synthetic fat-tailed data" if demo else TICKER

    forecasts = rolling_forecasts(returns, WINDOW, CONFIDENCE)

    print(f"\n=== 99% one-day VaR backtest: {name}, {len(forecasts)} days ===\n")
    print(summarise(forecasts, CONFIDENCE).to_string())
    print("\n=== Exceptions per year ===\n")
    print(exceptions_by_year(forecasts).to_string())

    plot(forecasts, f"99% VaR backtest: {name}")