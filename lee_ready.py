import pandas as pd
import numpy as np

EXCLUDE = {2, 7, 8, 10, 12, 13, 15, 16, 17, 18, 19, 20, 22, 25, 28, 30, 32, 33, 35, 38, 52, 53}


def load_and_merge(trades_path, quotes_path, symbol):
    """Load raw trades/quotes and attach the quote as of 1 ms before each trade."""
    trades = pd.read_parquet(trades_path)
    quotes = pd.read_parquet(quotes_path)

    # Integer nanoseconds -> timestamps (participant = exchange clock)
    trades["ts_event"] = pd.to_datetime(trades["participant_timestamp"], unit="ns", utc=True)
    quotes["ts_quote"] = pd.to_datetime(quotes["participant_timestamp"], unit="ns", utc=True)

    trades["lookup_time"] = trades["ts_event"] - pd.Timedelta("1ms")
    trades = trades.sort_values("lookup_time")
    quotes = quotes.sort_values("ts_quote")

    df = pd.merge_asof(
        trades,
        quotes[["ts_quote", "bid_price", "ask_price"]],
        left_on="lookup_time",
        right_on="ts_quote",
        direction="backward",
        allow_exact_matches=False,
    )

    df = df.rename(columns={"bid_price": "bid_px_00", "ask_price": "ask_px_00"})
    df["symbol"] = symbol
    df = df[["ts_event", "symbol", "price", "size", "bid_px_00", "ask_px_00", "conditions"]]
    return df


def parse_codes(s):
    """Turn "12,37" into {12, 37}; empty or missing -> empty set."""
    if s is None or pd.isna(s) or s == "":
        return set()
    return {int(x) for x in str(s).split(",")}


def filter_trades(df):
    """Keep regular trading hours and drop non-regular condition codes."""
    df["ts_event"] = df["ts_event"].dt.tz_convert("America/New_York")
    df = df.set_index("ts_event")
    df = df.between_time("09:30", "16:00", inclusive="left")
    df = df.reset_index()

    bad = df["conditions"].apply(lambda s: len(parse_codes(s) & EXCLUDE) > 0)
    n_before = len(df)
    df = df[~bad].copy()
    print(f"Trades: {n_before:,} -> {len(df):,} (removed {n_before - len(df):,})")

    df = df.sort_values(["symbol", "ts_event"])
    return df


def classify_lee_ready(df):
    """Quote rule first; tick test for trades at the midpoint."""
    df["date"] = df["ts_event"].dt.date

    df["mid"] = (df["bid_px_00"] + df["ask_px_00"]) / 2

    df["direction"] = np.nan
    df.loc[df["price"] > df["mid"], "direction"] = 1
    df.loc[df["price"] < df["mid"], "direction"] = -1

    # Group by symbol AND date so the tick test never carries across days
    df["price_change"] = df.groupby(["symbol", "date"])["price"].diff()
    df["tick"] = np.sign(df["price_change"]).replace(0, np.nan)
    df["tick"] = df.groupby(["symbol", "date"])["tick"].ffill()

    df["direction"] = df["direction"].fillna(df["tick"])
    return df


def make_bars(df):
    """Aggregate signed volume into 30-minute order imbalance bars."""
    df["size"] = df["size"].astype(float)
    df["buy_vol"] = df["size"].where(df["direction"] == 1, 0)
    df["sell_vol"] = df["size"].where(df["direction"] == -1, 0)

    df = df.set_index("ts_event")
    bars = df.groupby("symbol")[["buy_vol", "sell_vol"]].resample("30min").sum()
    bars["imbalance"] = (bars["buy_vol"] - bars["sell_vol"]) / (bars["buy_vol"] + bars["sell_vol"])
    bars = bars.dropna(subset=["imbalance"])
    return bars


def main():
    df = load_and_merge(
        "data/massive_AAPL_2024-01-03_trades.parquet",
        "data/massive_AAPL_2024-01-03_quotes.parquet",
        "AAPL",
    )
    df = filter_trades(df)
    df = classify_lee_ready(df)
    print(df["direction"].value_counts(dropna=False))

    bars = make_bars(df)
    print(bars)
    print("Bars per symbol:")
    print(bars.groupby("symbol").size())


if __name__ == "__main__":
    main()