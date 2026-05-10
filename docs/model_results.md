# Model Results

Test window: 2024-01-01 → 2024-08-22 (hourly), single chronological split.
Target: next-hour BTC close (`target_nexthour`).

## Headline table

| Category | Model | MAE | MSE |
|----------|-------|-----|-----|
| Baseline | **LightGBM** | **254.37** | 258,918.80 |
| Baseline | XGBoost | 258.53 | 271,929.61 |
| Baseline | ARIMA | 1,602.19 | 4,847,913.23 |
| Proposed | **SOFTS** | 304.23 | **183,679.47** |
| Proposed | iTransformer | 1,948.00 | 8,123,500.00 |
| Proposed | CNN-LSTM | 1,941.32 | 6,730,000.00 |
| Proposed | SE-GRN | 2,377.06 | 883,273.00 |
| Proposed | Times-FM | 2,672.28 | 3,600,000.00 |

## Interpretation

- **LightGBM achieved the lowest MAE, while SOFTS achieved the lowest MSE.** This is the safest one-line summary; it explicitly avoids declaring a universal winner.
- The MAE-vs-MSE split says: SOFTS produces fewer extreme errors (lower MSE) but its typical-case error is larger than LightGBM's (higher MAE). LightGBM's typical predictions are closest to truth, but it is more vulnerable to occasional large misses.
- **Picking a "best" model depends on the loss function the downstream user cares about** — average error vs. tail error containment.
- **Tree-based baselines beat several deep models.** XGBoost and LightGBM, with hand-engineered features, beat SE-GRN, Times-FM, iTransformer, and CNN-LSTM on both metrics. Useful evidence against the "deep learning always wins on time series" prior.
- **Capacity is not a substitute for fit.** Times-FM has the highest MAE; CNN-LSTM has very high MSE. On this dataset, larger does not mean better.

## Caveats on these numbers

1. Single test window — generalization across regimes is unverified.
2. Hyperparameters were tuned reasonably but not exhaustively for every model.
3. Some deep-model MSE values are reported with low precision in the source results table; treat ordering as more reliable than exact values.
4. Errors are reported in price-space; large-error episodes correspond to volatility spikes and are not weighted by economic significance.

## What's NOT measured here

- Profit & loss after transaction costs.
- Hit-rate / directional accuracy.
- Sharpe, drawdown, or other strategy metrics.
- Calibration of predictive intervals.

These are intentional gaps — addressed conceptually in the Limitations section of the README and pursued in the companion [Digital Asset Market Behavior Intelligence Platform](https://github.com/bobaoxu2001/Digital-Asset-Market-Behavior-Intelligence-Platform).
