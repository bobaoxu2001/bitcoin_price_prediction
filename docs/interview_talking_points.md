# Talking Points

Project-facing notes — the kind of things a viewer of the repo might want to ask about. Not interview prep scripts.

## 60-second summary

> "This was an NYU capstone where we benchmarked eight models — from ARIMA up through SOFTS and Times-FM — on hourly Bitcoin forecasting using macro features (Nasdaq, Gold, VIX) and multi-platform social sentiment with lagged features. The headline finding is that **LightGBM had the lowest MAE and SOFTS had the lowest MSE** — no single winner. More interestingly, the largest deep models underperformed tuned gradient boosting, which I take as a useful prior against architecture-chasing on noisy hourly crypto data. The biggest takeaway is the limit of pure price prediction: even the best model only narrows error, and none of these numbers translate into a trading edge once you add execution costs and regime risk. That's why my newer Digital Asset Market Behavior Intelligence Platform shifts the framing from prediction to behavior, regimes, and event studies."

## Things worth pointing at in the repo

- `feature_selection.py` — gain-based feature importance, which surfaces Nasdaq features and short MAs as top predictors and confirms the macro-tech-equity link.
- `eda_analysis.py` — time-varying NASDAQ–BTC correlation showing the COVID-era break; useful for any "do you understand regime risk" question.
- `softs_model.py` and `lightgbm_model.py` — the two models that win on different metrics; concrete examples of the MAE/MSE trade-off.

## Likely questions and honest answers

**Q: Which model is "best"?**
A: It depends on the metric. LightGBM wins on MAE; SOFTS wins on MSE. SOFTS controls large errors better; LightGBM has a smaller typical error.

**Q: Would you trade this?**
A: No. These are point-forecast accuracy numbers, not strategy P&L. There are no costs, no slippage, no exchange flow data, and the test window is one regime.

**Q: Why didn't the big models win?**
A: With ~68k hourly samples and strong hand-engineered features, gradient boosting is a hard baseline. Foundation-style models like Times-FM are designed to transfer broad time-series priors and don't necessarily benefit a feature-rich, well-engineered task with this much volatility.

**Q: What would you do differently now?**
A: Replace pure price targets with regime / volatility-bucket classification, add on-chain and exchange-flow data, evaluate on multiple test windows, and report strategy metrics (Sharpe, drawdown) under realistic costs. That is exactly what the newer Digital Asset Market Behavior Intelligence Platform does.

**Q: How is this different from your newer platform?**
A: This project predicts BTC price. The newer platform analyzes market *behavior*: regime classification, event studies, on-chain flows, DeFi liquidity. Different question, different answer.
