import numpy as np
import pandas as pd
import yfinance as yf

def get_annual_vol(ticker="TSLA", period="1y") -> float:
    df = yf.download(ticker, period=period, interval="1d", progress=False)

    # Forzar columna correcta
    if isinstance(df, pd.DataFrame):
        close = df["Close"]
    else:
        close = df

    close = close.dropna()
    rets = close.pct_change().dropna()

    vol = rets.std().values[0] * np.sqrt(252)
    return float(vol)

def montecarlo_bankruptcy_proxy(
    price_now: float,
    debt_total_musd: float,
    years: float = 2.0,
    n_sims: int = 20000,
    steps_per_year: int = 252,
    vol_annual: float = None,
):
    if vol_annual is None:
        raise ValueError("vol_annual requerido")

    steps = int(years * steps_per_year)
    dt = 1.0 / steps_per_year
    mu = 0.0

    debt_b = debt_total_musd / 1000.0
    threshold_frac = min(0.60, 0.15 + 0.03 * debt_b)
    barrier = price_now * threshold_frac

    Z = np.random.normal(size=(n_sims, steps))
    increments = (mu - 0.5 * vol_annual**2) * dt + vol_annual * np.sqrt(dt) * Z
    log_paths = np.cumsum(increments, axis=1)
    paths = price_now * np.exp(log_paths)

    bankrupt = (paths.min(axis=1) <= barrier)
    prob = float(bankrupt.mean())

    return {
        "price_now": float(price_now),
        "vol_annual": float(vol_annual),
        "debt_total_musd": float(debt_total_musd),
        "years": float(years),
        "n_sims": int(n_sims),
        "barrier_price": float(barrier),
        "barrier_frac": float(threshold_frac),
        "prob_bankruptcy_proxy": prob,
    }

if __name__ == "__main__":
    ticker = "TSLA"
    price_now = float(yf.Ticker(ticker).fast_info["lastPrice"])
    vol = get_annual_vol(ticker, period="1y")

    debt_total_musd = 12161.0  # del PDF

    out = montecarlo_bankruptcy_proxy(
        price_now=price_now,
        debt_total_musd=debt_total_musd,
        years=2.0,
        n_sims=20000,
        vol_annual=vol,
    )
    print(out)
