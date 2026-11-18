# Intraday Order Imbalance

Does intraday buying or selling pressure in a stock predict its next-day return?
This project classifies every trade as buyer- or seller-initiated, aggregates the
result into 13 half-hour order imbalance bars per stock-day, and tests whether
those bars predict the next day's close-to-close return across stocks.

> Status: pipeline complete on 2023 Q4 + January 2024 (10 stocks, 84 days).
> Result so far: no reliable predictive signal; see Results.

## Pipeline

| Step | File | What it does |
|---|---|---|
| 1 | `lee_ready.py` | Load trades matched to the NBBO quote 1 ms before each trade, filter, classify with Lee-Ready, build 30-min bars |
| 2 | `returns.py` | Next-day close-to-close returns from adjusted closes (yfinance) |
| 3 | `features.py` | 13 imbalance features + morning-minus-afternoon feature, cross-sectional z-scores, merge with target |
| 4 | `regression.py` | OLS with a date-based train/test split, date-clustered t-stats, daily Spearman IC |

Run in order:

```
python lee_ready.py
python returns.py
python features.py
python regression.py
```

## Data

- Source: Massive (formerly Polygon.io) consolidated trades and NBBO quotes
  (flat files for 2023 Q4, REST API for January 2024).
- Universe: AAPL, AMZN, JNJ, JPM, KO, MSFT, NVDA, PG, WMT, XOM.
- Period: 2023-10-02 to 2024-01-31, 84 trading days, 840 stock-days,
  about 218 million regular-hours trades after filtering.
- 2023-11-24 was a half day (close at 13:00 ET), so it has only 7-8 bars and
  its afternoon features are missing; models using afternoon features drop those 10 rows.
- Data files are not included in this repo (`data/` is gitignored).

## Method

**Trade classification (Lee-Ready).** Each trade is compared with the prevailing
NBBO midpoint: above the mid is a buy, below is a sell. Trades at the mid use the
tick test (sign of the last non-zero price change), computed within each
stock-day so it never carries across days.

- *Quote timing.* Trades are matched to the last quote at least 1 ms before the
  trade (participant timestamps, `merge_asof`). On a validation sample (AAPL,
  2024-01-03, 21,985 trades matched to Databento's exchange-reported side),
  agreement rose from 76% with no lag to about 91% with the 1 ms lag.
- *Filtering.* Regular hours only (09:30-16:00). Trades with non-regular
  condition codes are dropped (auction and official open/close prints,
  Form T / out-of-hours, average-price, derivatively priced, and similar).
  The opening and closing auction prints would otherwise dominate the first and
  last bars.
- *One file = one trading day.* Each daily file is processed on its own and
  only trades whose New York date matches the file's date are kept (see Bug fix).

**Order imbalance.** For each 30-minute bar:
`imbalance = (buy volume - sell volume) / (buy volume + sell volume)`,
giving 13 bars per stock-day (09:30 to 15:30 bar starts).

**Features.**
- `ib_0930` ... `ib_1530`: the 13 bar imbalances.
- `am_pm`: mean of the first 6 bars (09:30-12:30) minus mean of the last 7 (12:30-16:00).
- All features are z-scored cross-sectionally within each date, so each value
  measures a stock's buying pressure relative to the other stocks that day and
  uses only same-day information (no look-ahead).

**Target.** `ret_next` on day t = adjusted close(t+1) / adjusted close(t) - 1.

**Evaluation.**
- Split by time, not randomly: train on 2023 Q4 (63 days, 630 rows),
  test on January 2024 (21 days, 210 rows).
- OLS standard errors are clustered by date, since stocks on the same day share market moves.
- Out of sample: Spearman rank correlation between prediction and next-day return,
  computed each test day (IC), then averaged.

## Results

### Main result: train 2023 Q4, test January 2024

Univariate regressions on the 63 training days:

| Feature | Coef | t (clustered by date) |
|---|---|---|
| ib_0930_z | -0.0001 | -0.14 |
| ib_1000_z | -0.0001 | -0.15 |
| ib_1030_z | -0.0004 | -0.55 |
| ib_1100_z | -0.0005 | -0.76 |
| ib_1130_z | 0.0003 | 0.74 |
| ib_1200_z | -0.0003 | -0.47 |
| ib_1230_z | 0.0004 | 0.65 |
| ib_1300_z | 0.0007 | 1.03 |
| ib_1330_z | 0.0006 | 1.15 |
| ib_1400_z | -0.0000 | -0.07 |
| ib_1430_z | -0.0005 | -0.92 |
| ib_1500_z | -0.0005 | -0.88 |
| ib_1530_z | 0.0009 | 1.48 |
| am_pm_z | -0.0007 | -0.95 |

No feature reaches |t| = 1.96. The all-14 model has R² 0.018 (adjusted -0.005).

Out-of-sample daily IC on the 21 test days:

| Model | Mean IC | IC std | ICIR | Positive days |
|---|---|---|---|---|
| Last 30 min only (`ib_1530_z`, pre-specified) | +0.070 | 0.342 | +0.20 | 52% |
| All 14 features | -0.096 | 0.363 | -0.26 | 38% |

### Earlier pilot: January 2024 only (15 train days, 6 test days)

With only one month, the 10:30-12:00 bars looked strongly significant in training
(t = 3.44, 3.25, 4.58; `am_pm_z` t = 2.65), and the last-30-minute model had a
test IC of +0.164 on 6 days (all-14 model: -0.051).

### What changed

The mid-morning pattern from the one-month pilot does not appear in 2023 Q4
(all |t| < 1), so it was most likely noise from a small sample. The
last-30-minute bar, which the original idea expected to matter most, is the
strongest feature in Q4 but still not significant, and its out-of-sample IC is
weak. The full model overfits in both experiments.

Takeaway: with 10 stocks and a few months, there is no reliable evidence yet that
intraday imbalance predicts next-day returns in this universe.

## Bug fix

Three January 2024 AAPL daily files (01-09, 01-16, 01-22) also contained trades
from the next morning's 09:30-10:00 bar. When all files were loaded together,
those trades were counted twice in the next day's 09:30 bar. The fix processes
each file separately and keeps only trades dated on the file's own day
(`restrict_to_file_date` in `lee_ready.py`). This changed 3 of 2,730 January bars;
in the pilot it moved `ib_0930_z` t from 0.56 to 0.77, `am_pm_z` t from 2.39 to
2.65, and the all-14 test IC from -0.089 to -0.051. Other reported numbers were
unchanged.

## Limitations

- Still small: 10 large-cap stocks, so each daily IC is a rank correlation over
  only 10 names and is very noisy.
- Lee-Ready is about 91% accurate on the validation sample, not 100%.
- The target is raw next-day return; market and factor exposures are not yet removed.
- No transaction costs or tradability check (for example, whether the last bar
  can be traded on before the close).

## Next steps

- Residualize returns against the market and common factors.
- Widen the universe (for example, 50-100 liquid stocks) so daily ICs are less noisy.
- Rolling or walk-forward evaluation instead of a single split.
- Turn the signal into a long-short portfolio and include costs.
