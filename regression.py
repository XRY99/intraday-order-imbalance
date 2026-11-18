"""Regress next-day returns on intraday order imbalance features."""

from datetime import date
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

TARGET = "ret_next"
# Train on 2023 Q4; test on January 2024
TRAIN_END = date(2023, 12, 29)
TEST_START = date(2024, 1, 2)


def load_and_split(path):
    """Split by date: 2023 Q4 for training, January 2024 for testing."""
    df = pd.read_parquet(path)
    dates = df.index.get_level_values("date")
    train = df[dates <= TRAIN_END].copy()
    test = df[dates >= TEST_START].copy()
    return train, test


def fit_ols(train, features):
    """OLS with standard errors clustered by date.

    Stocks on the same day share market moves, so their errors are correlated;
    clustering by date keeps the t-stats from being overstated.
    """
    data = train[list(features) + [TARGET]].dropna()
    X = sm.add_constant(data[features])
    y = data[TARGET]
    groups = pd.factorize(data.index.get_level_values("date"))[0]
    return sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups}), len(data)


def univariate_table(train, features):
    """One regression per feature: coefficient and t-stat."""
    rows = []
    for f in features:
        res, n = fit_ols(train, [f])
        rows.append({"feature": f, "coef": res.params[f], "t": res.tvalues[f], "n": n})
    return pd.DataFrame(rows).set_index("feature")


def daily_ic(test, pred):
    """Spearman rank correlation between prediction and return, per day."""
    df = test[[TARGET]].copy()
    df["pred"] = pred
    ic = df.groupby(level="date").apply(
        lambda g: spearmanr(g["pred"], g[TARGET])[0],
        include_groups=False,
    )
    return ic


def report_ic(name, ic):
    icir = ic.mean() / ic.std()
    line = (f"{name:<28} mean IC {ic.mean():+.4f}  std {ic.std():.4f}  "
            f"ICIR {icir:+.3f}  days {len(ic)}  "
            f"positive {(ic > 0).mean():.1%}")
    print(line)
    return line


def predict_ols(res, test, features):
    return res.predict(sm.add_constant(test[features]))


if __name__ == "__main__":
    train, test = load_and_split("data/features.parquet")
    print("Train:", train.shape, "Test:", test.shape)
    train_dates = sorted(train.index.get_level_values("date").unique())
    test_dates = sorted(test.index.get_level_values("date").unique())
    print(f"Train dates: {len(train_dates)}  {train_dates[0]} -> {train_dates[-1]}")
    print(f"Test dates:  {len(test_dates)}  {test_dates[0]} -> {test_dates[-1]}")

    ib_z = [c for c in train.columns if c.startswith("ib_") and c.endswith("_z")]
    features = ib_z + ["am_pm_z"]

    print("\nUnivariate regressions on train (t-stats clustered by date):")
    uni = univariate_table(train, features)
    print(uni.round(4))

    # Pre-specified single-feature model: last 30 minutes (mentor's notes)
    res_last, n_last = fit_ols(train, ["ib_1530_z"])
    pred_last = predict_ols(res_last, test, ["ib_1530_z"])

    # Full model with all 14 features
    res_all, n_all = fit_ols(train, features)
    print("\nFull model on train:")
    print(f"R^2 {res_all.rsquared:.4f}, adj R^2 {res_all.rsquared_adj:.4f}  n={n_all}")
    print(f"Last-30min model n={n_last}")
    pred_all = predict_ols(res_all, test, features)

    # Single-feature models chosen after looking at January 2024 results.
    # These are potentially biased and are reported separately.
    extra_models = {}
    for f in ["ib_1130_z", "am_pm_z"]:
        res_f, _n_f = fit_ols(train, [f])
        extra_models[f] = predict_ols(res_f, test, [f])

    print("\nOut-of-sample IC on test dates:")
    lines = []
    lines.append(report_ic("last 30 min only", daily_ic(test, pred_last)))
    lines.append(report_ic("all 14 features", daily_ic(test, pred_all)))
    for f, pred in extra_models.items():
        lines.append(report_ic(f"{f} only (post-hoc)", daily_ic(test, pred)))

    out = Path("data/regression_results.txt")
    with out.open("w") as fh:
        fh.write("Train: {}  Test: {}\n".format(train.shape, test.shape))
        fh.write(f"Train dates: {len(train_dates)}  {train_dates[0]} -> {train_dates[-1]}\n")
        fh.write(f"Test dates:  {len(test_dates)}  {test_dates[0]} -> {test_dates[-1]}\n\n")
        fh.write("Univariate regressions on train (t-stats clustered by date):\n")
        fh.write(uni.round(6).to_string())
        fh.write("\n\nFull model on train:\n")
        fh.write(f"R^2 {res_all.rsquared:.6f}, adj R^2 {res_all.rsquared_adj:.6f}  n={n_all}\n")
        fh.write(f"Last-30min model n={n_last}\n")
        fh.write("\nOut-of-sample IC on test dates:\n")
        for line in lines:
            fh.write(line + "\n")
        fh.write("\nNote: ib_1130_z and am_pm_z single-feature models were "
                 "chosen after looking at the January 2024 results "
                 "and are potentially biased.\n")
    print(f"\nSaved {out}")
