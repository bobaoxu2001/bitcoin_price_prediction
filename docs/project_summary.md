# Project Summary

**Title:** Bitcoin Price Dynamics — A Forecasting & Model-Comparison Study
**Context:** NYU Capstone, Group 36 (Sam Lai, Zexuan Yang, Yichao Yang, Ao Xu), 2024.

## In one paragraph

We study hourly Bitcoin price forecasting on ~68k observations (Nov 2016 – Aug 2024) by combining BTC price with macro features (Nasdaq, Gold, VIX) and lagged social-media sentiment from Twitter/X, Reddit, and Bitcointalk. Eight models are compared end-to-end: ARIMA, XGBoost, LightGBM, SE-GRN, iTransformer, Times-FM, SOFTS, and CNN-LSTM. The split is chronological. Results show **LightGBM with the lowest MAE** and **SOFTS with the lowest MSE** — no single winner across both metrics. Larger architectures (Times-FM, CNN-LSTM) underperformed tuned gradient boosting, illustrating that capacity does not substitute for inductive bias on noisy hourly crypto data.

## What this project shows

- Practical multi-source data engineering (hourly BTC + daily macro forward-filled + multi-platform sentiment with multi-lag features).
- Disciplined benchmarking — same split, same target, comparable metrics.
- Honest framing: this is forecasting accuracy, not a trading strategy, and pure price prediction has clear limits.

## What this project does *not* claim

- It does not claim to predict Bitcoin price reliably.
- It does not claim a tradable edge or compute strategy P&L.
- It does not include execution costs, on-chain data, or regime classification.

## Companion project

The newer [Digital Asset Market Behavior Intelligence Platform](https://github.com/bobaoxu2001/Digital-Asset-Market-Behavior-Intelligence-Platform) addresses what this project intentionally lacks: on-chain activity, DeFi liquidity, event studies, regime classification, and behavior-driven strategy framing.
