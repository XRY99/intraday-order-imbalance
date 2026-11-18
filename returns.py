"""Next-day close-to-close returns (the regression target)."""

import pandas as pd
import yfinance as yf

SYMBOLS = ["AAPL", "AMZN", "JNJ", "JPM", "KO", "MSFT", "NVDA", "PG", "WMT", "XOM"]


def get_next_day_returns(symbols, start, end):
    """Return a long table: date, symbol, ret_next.

    ret_next on day t = close(t+1) / close(t) - 1, using adjusted closes.
    yfinance's end date is exclusive, so end must be one trading day
    after the last return you need.
    """
    prices = yf.download(symbols, start=start, end=end,
                         auto_adjust=True, progress=False)["Close"]
    # Return from t to t+1, stored on row t (this is the target, not a feature)
    ret_next = prices.pct_change().shift(-1)
    out = ret_next.stack().rename("ret_next").reset_index()
    out.columns = ["date", "symbol", "ret_next"]
    out["date"] = pd.to_datetime(out["date"]).dt.date
    return out


if __name__ == "__main__":
    rets = get_next_day_returns(SYMBOLS, start="2023-09-29", end="2024-02-02")
    print(rets.head(12))
    print(f"Rows: {len(rets)}, symbols: {rets['symbol'].nunique()}")

    # Sanity check: AAPL row on 2024-01-02 should hold the 01-02 -> 01-03 return
    px = yf.download("AAPL", start="2024-01-02", end="2024-01-04",
                     auto_adjust=True, progress=False)["Close"].squeeze()
    manual = px.iloc[1] / px.iloc[0] - 1
    row = rets[(rets["symbol"] == "AAPL") &
               (rets["date"] == pd.Timestamp("2024-01-02").date())]
    print(f"AAPL 2024-01-02 ret_next: {row['ret_next'].iloc[0]:.6f}, manual: {manual:.6f}")

    rets.to_parquet("data/next_day_returns.parquet", index=False)
    print("Saved data/next_day_returns.parquet")
