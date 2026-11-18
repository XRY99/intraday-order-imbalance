import pandas as pd


def load_bars(path):
    """Load 30-min bars and add date and time columns."""
    bars = pd.read_parquet(path)          # 1. read the file
    bars = bars.reset_index()                    # 2. move symbol, ts_event out of the index
    bars["date"] = bars["ts_event"].dt.date                 # 3. date part only
    bars["time"] = bars["ts_event"].dt.strftime("%H%M")     # 4. e.g. "0930"
    return bars

def pivot_imbalance(bars):
    """One row per (date, symbol), one column per 30-min bar."""
    wide = bars.pivot_table(index=["date", "symbol"],
                            columns="time",
                            values="imbalance")
    wide.columns = ["ib_" + c for c in wide.columns]   # "0930" -> "ib_0930"
    return wide

def add_am_pm(wide):
    """Morning minus afternoon average imbalance."""
    am_cols = list(wide.columns[:6])    # first 6 columns
    pm_cols = list(wide.columns[6:])    # last 7 columns
    wide["am_pm"] = wide[am_cols].mean(axis=1) - wide[pm_cols].mean(axis=1)
    return wide


def zscore_by_date(wide):
    """Cross-sectional z-score: compare stocks within the same day."""
    cols = list(wide.columns)          # 13 ib_ columns + am_pm
    grouped = wide.groupby(level="date")[cols]
    z = (wide[cols] - grouped.transform("mean")) / grouped.transform("std")
    z.columns = [c + "_z" for c in cols]
    return wide.join(z)

def merge_target(wide, returns_path):
    """Attach next-day return to each (date, symbol) row."""
    rets = pd.read_parquet(returns_path)
    rets = rets.set_index(["date", "symbol"])
    df = wide.join(rets, how="inner")
    return df

if __name__ == "__main__":
    bars = load_bars("data/bars_30min.parquet")
    print(bars.head())
    print(bars.dtypes)

    wide = pivot_imbalance(bars)
    print(wide.shape)
    print(wide.head())

    wide = add_am_pm(wide)
    print(wide[["am_pm"]].describe())

    wide = zscore_by_date(wide)
    print(wide.shape)

    


    # Checks: within each date, z-scores should have mean ~0 and std ~1
    by_date = wide.groupby(level="date")["ib_1530_z"]
    print("max |mean|:", by_date.mean().abs().max())
    print("std range:", by_date.std().min(), by_date.std().max())

    df = merge_target(wide, "data/next_day_returns.parquet")
    print(df.shape)
    print("Missing targets:", df["ret_next"].isna().sum())
    df.to_parquet("data/features.parquet")
    print("Saved data/features.parquet")