"""Regress next-day returns on intraday order imbalance features."""

import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

TARGET = "ret_next"


def load_and_split(path, n_train_days):
    """Split by date: first n_train_days for training, the rest for testing."""
    df = pd.read_parquet(path)
    dates = sorted(df.index.get_level_values("date").unique())
    train_dates = dates[:n_train_days]
    test_dates = dates[n_train_days:]
    train = df[df.index.get_level_values("date").isin(train_dates)]
    test = df[df.index.get_level_values("date").isin(test_dates)]
    return train, test


def fit_ols(train, features):
    """OLS with standard errors clustered by date.

    Stocks on the same day share market moves, so their errors are correlated;
    clustering by date keeps the t-stats from being overstated.
    """
    X = sm.add_constant(train[features])
    y = train[TARGET]
    groups = pd.factorize(train.index.get_level_values("date"))[0]
    return sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})


def univariate_table(train, features):
    """One regression per feature: coefficient and t-stat."""
    rows = []
    for f in features:
        res = fit_ols(train, [f])
        rows.append({"feature": f, "coef": res.params[f], "t": res.tvalues[f]})
    return pd.DataFrame(rows).set_index("feature")


def daily_ic(test, pred):
    """Spearman rank correlation between prediction and return, per day."""
    df = test[[TARGET]].copy()
    df["pred"] = pred
    ic = df.groupby(level="date").apply(
        lambda g: spearmanr(g["pred"], g[TARGET])[0])
    return ic


def report_ic(name, ic):
    icir = ic.mean() / ic.std()
    print(f"{name:<22} mean IC {ic.mean():+.3f}  std {ic.std():.3f}  "
          f"ICIR {icir:+.2f}  days {len(ic)}  positive {(ic > 0).mean():.0%}")


if __name__ == "__main__":
    train, test = load_and_split("data/features.parquet", n_train_days=15)
    print("Train:", train.shape, "Test:", test.shape)

    ib_z = [c for c in train.columns if c.startswith("ib_") and c.endswith("_z")]
    features = ib_z + ["am_pm_z"]

    # 1. Which time bar matters? One feature at a time (train only)
    print("\nUnivariate regressions on train (t-stats clustered by date):")
    print(univariate_table(train, features).round(4))

    # 2. Pre-specified single-feature model: last 30 minutes (from the mentor's notes)
    res_last = fit_ols(train, ["ib_1530_z"])
    pred_last = res_last.predict(sm.add_constant(test[["ib_1530_z"]]))

    # 3. Full model with all 14 features (likely overfit: 15 params, 150 rows)
    res_all = fit_ols(train, features)
    print("\nFull model on train:")
    print(f"R^2 {res_all.rsquared:.3f}, adj R^2 {res_all.rsquared_adj:.3f}")
    pred_all = res_all.predict(sm.add_constant(test[features]))

    # 4. Out-of-sample: daily cross-sectional IC on the test dates
    print("\nOut-of-sample IC on test dates:")
    report_ic("last 30 min only", daily_ic(test, pred_last))
    report_ic("all 14 features", daily_ic(test, pred_all))
