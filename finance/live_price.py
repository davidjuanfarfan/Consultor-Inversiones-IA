import yfinance as yf

def get_price(ticker: str = "TSLA") -> dict:
    t = yf.Ticker(ticker)
    info = t.fast_info  # rápido
    return {
        "ticker": ticker,
        "price": float(info["lastPrice"]),
        "currency": info.get("currency", "USD"),
    }

if __name__ == "__main__":
    print(get_price("TSLA"))
