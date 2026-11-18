import pandas as pd
import numpy as np
from pathlib import Path

from returns import SYMBOLS

EXCLUDE = {2, 7, 8, 10, 12, 13, 15, 16, 17, 18, 19, 20, 22, 25, 28, 30, 32, 33, 35, 38, 52, 53}


def load_asof(data_dir, symbols=None):
    """Load pre-matched trade files (quote as of 1 ms before each trade).

    Each file holds one symbol-day: massive_{SYM}_{DATE}_asof_1ms.parquet
    """
    files = sorted(Path(data_dir).glob("massive_*_asof_1ms.parquet"))
    if symbols is not None:
        files = [f for f in files if f.name.split("_")[1] in symbols]
    if not files:
        raise FileNotFoundError(f"No asof files found in {data_dir}")

    frames = []
    for f in files:
        frames.append(pd.read_parquet(f))
    print(f"Loaded {len(files)} files")

    df = pd.concat(frames, ignore_index=True)
    return df


def restrict_to_file_date(df, path):
    """Keep only trades whose New York date matches the date in the filename.

    A few January 2024 AAPL asof files spill into the next morning's first
    half hour (09:30-10:00 ET). Loading those together with the next day's
    file would double-count that bar.
    """
    file_date = pd.Timestamp(Path(path).name.split("_")[2]).date()
    et_dates = df["ts_event"].dt.tz_convert("America/New_York").dt.date
    keep = et_dates == file_date
    n_drop = int((~keep).sum())
    if n_drop:
        print(f"  {Path(path).name}: dropped {n_drop:,} trades not on {file_date}")
    return df.loc[keep].copy()


def list_asof_files(data_dir, symbols=None):
    """Return asof parquet paths, optionally filtered by symbol."""
    files = sorted(Path(data_dir).glob("massive_*_asof_1ms.parquet"))
    if symbols is not None:
        files = [f for f in files if f.name.split("_")[1] in symbols]
    if not files:
        raise FileNotFoundError(f"No asof files found in {data_dir}")
    return files


def parse_codes(s):
    """Turn "12,37" into {12, 37}; empty or missing -> empty set."""
    if s is None or pd.isna(s) or s == "":
        return set()
    return {int(x) for x in str(s).split(",")}


def filter_trades(df, verbose=True):
    """Keep regular trading hours and drop non-regular condition codes."""
    df["ts_event"] = df["ts_event"].dt.tz_convert("America/New_York")
    df = df.set_index("ts_event")
    df = df.between_time("09:30", "16:00", inclusive="left")
    df = df.reset_index()

    bad = df["conditions"].apply(lambda s: len(parse_codes(s) & EXCLUDE) > 0)
    n_before = len(df)
    df = df[~bad].copy()
    if verbose:
        print(f"Trades: {n_before:,} -> {len(df):,} (removed {n_before - len(df):,})")

    df = df.sort_values(["symbol", "ts_event"])
    return df, n_before, len(df)


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
    # Process one symbol-day at a time so peak memory stays well below
    # the machine limit (loading all 84 days of AAPL at once was killed).
    all_bars = []
    for sym in SYMBOLS:
        files = list_asof_files("data", symbols=[sym])
        n_before_total = 0
        n_after_total = 0
        n_unclassified = 0
        buy_total = 0.0
        sell_total = 0.0
        n_days = 0
        for f in files:
            df = pd.read_parquet(f)
            df = restrict_to_file_date(df, f)
            df, n_before, n_after = filter_trades(df, verbose=False)
            n_before_total += n_before
            n_after_total += n_after
            df = classify_lee_ready(df)
            n_unclassified += int(df["direction"].isna().sum())
            bars = make_bars(df)
            buy_total += float(bars["buy_vol"].sum())
            sell_total += float(bars["sell_vol"].sum())
            all_bars.append(bars)
            n_days += 1
            del df

        print(f"{sym}: {n_days} days  "
              f"trades {n_before_total:,} -> {n_after_total:,} "
              f"(removed {n_before_total - n_after_total:,}); "
              f"unclassified {n_unclassified:,}; "
              f"buy_vol {buy_total:,.0f}  sell_vol {sell_total:,.0f}")

    bars = pd.concat(all_bars)
    days = bars.index.get_level_values("ts_event").date
    per_day = bars.groupby([bars.index.get_level_values("symbol"), days]).size()
    print(f"\nTotal bars: {len(bars):,}")
    print(f"Symbol-days: {len(per_day)}")
    print("Bars per symbol-day:")
    print(per_day.value_counts())
    print(f"Total buy_vol:  {bars['buy_vol'].sum():,.0f}")
    print(f"Total sell_vol: {bars['sell_vol'].sum():,.0f}")

    bars.to_parquet("data/bars_30min.parquet")
    print("Saved data/bars_30min.parquet")


if __name__ == "__main__":
    main()
