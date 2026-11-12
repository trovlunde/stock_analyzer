# Strategy improvement plan

Assessment of the investing strategies in this repo (November 2026) and a phased plan for improving them and adding modern approaches. Companion to [stack-improvement-plan.md](./stack-improvement-plan.md), which covers tooling; this document covers *what we trade and how we know it works*.

**Bottom line:** the execution side (paper broker, next-open fills, kill switch, cost model) is in better shape than the research side. The strategies themselves are single-ticker technical signals with weak or leaky evidence behind them. The highest-value work is not a fancier model — it is (1) fixing the validity problems below, (2) building one honest backtester that shares code with the daily job, and (3) moving from "predict tomorrow's direction for one ticker" to cross-sectional ranking and portfolio-level risk sizing.

---

## 1. What we have today

| Strategy / tool | Where | Status | Wired into daily job? |
|---|---|---|---|
| MA crossover (20/50 SMA golden/death cross) | `trading/signals/ma_crossover.py`, `ai/technical_analysis/ma_signals.py` | Live in paper portfolio | Yes |
| ML movement classifier (3-class next-day / next-week, RandomForest per ticker) | `ai/technical_analysis/movement_classification.py`, `trading/signals/ml_movement.py` | Live; only AAPL has a model | Yes |
| Signal combination (consensus / priority / any) | `trading/signals/combine.py` | Live | Yes |
| RSI overbought/oversold | `ai/technical_analysis/rsi_signals.py` | Research only | No |
| Dual daily+weekly agreement signal | `movement_classification.full_analysis` | Research only | No |
| QVM sector-relative screener | `research/qvm.py` | Snapshot screen, no backtest | No |
| Stock finder (dividend / undervalued / growth) | `research/stock_finder.py` | Snapshot screen | No |
| Valuation models (DCF, EPV, Graham, reverse DCF, multiples) | `ai/fundamental_analysis/valuation_methods.py` | Per-ticker calculator | No |
| Financial-statement classifier | `ai/fundamental_analysis/fin_statement_classifier.py` | Experiment | No |
| Finviz signal classifier | `ai/finviz_classifier/` | Experiment on scraped data | No |
| Mean-variance optimiser (max Sharpe) | `research/analyze_stock_data.py::optimize_portfolio` | Interactive CLI only | No |
| Significant-day analysis (PCA, regression), intraday volatility | `ai/technical_analysis/*sig_days.py`, `intraday_volatility_analysis.py` | Exploratory | No |
| Leveraged-product cost analysis (mini futures, bull/bear) | `research/trading_cost_analysis.py` | Used by research backtester | Optional cost model |

Paper portfolio: 3 tickers (AAPL, MSFT, GOOGL), equal 10 % sizing, long-only, max 10 positions, portfolio kill switch at −5 % day / −20 % drawdown.

---

## 2. Findings

### 2.1 Validity problems (fix before trusting any number)

1. **Weekly target leaks today's return.** `prepare_classification_data.py:16` computes the weekly label as `pct_change(periods=5).shift(-4)`, i.e. `Close[t+4] / Close[t-1] − 1`. That window includes the move from `t-1` to `t`, which is exactly the `prev_day_return` feature. The weekly model can "predict" part of its label from a feature. It should be `shift(-5)` (`Close[t+5] / Close[t] − 1`). Same pattern in `nn_predictor.py:18`. Every weekly accuracy number, and the daily+weekly agreement signal, is inflated by this.
2. **Shuffled split in the financial-statement classifier.** `fin_statement_classifier.py:229` uses a random stratified `train_test_split` over pooled tickers and dates, so test rows sit between training rows in time and share dates across tickers.
3. **CV without purge or embargo.** `classifier_comparison.py` uses `TimeSeriesSplit(n_splits=5)` with no gap. With 5-day overlapping labels, the last training rows of each fold share return days with the first test rows. The scaler is also fitted on the full training set before CV rather than inside each fold.
4. **Holdout reused for model selection.** `compare_variants` ranks variants by holdout accuracy and reports the winner's holdout accuracy — a selection-biased estimate. There is no untouched final test period.
5. **Deployed model ignores the most recent 20 % of data.** The `/api/v1/train` path calls `train_classifier_single_stock` with the default `eval_split=0.2`, so the saved model is fitted on the oldest 80 % only. Nothing ever retrains it afterwards.
6. **Survivorship bias in pooled training.** S&P 500 mode trains on today's constituents (Wikipedia scrape) over 10–20 years of history.
7. **`bfill` on features** (`prepare_classification_data.py:62`) back-fills the warm-up rows from the future. Small, but it should be `dropna`.
8. **Research backtester is not comparable to paper trading.** `portfolio_evaluating.analyze_trading_strategies` enters at the signal day's close (the same close used to compute the features), assumes 10× leveraged mini futures, and uses different cost logic from `trading/cost_models.py`. `simulate_trading` sums per-trade returns with no costs or compounding. Neither reports a buy-and-hold benchmark alongside.

### 2.2 Strategy design weaknesses

- **The prediction target is close to noise.** Next-day direction of a single large-cap from 3–11 price-derived features has very little signal. Models are scored on accuracy/F1, not on what we care about (return after costs, Sharpe, drawdown, turnover).
- **One model per ticker.** ~2 500 rows per model, no cross-sectional information, and a separate training step per watchlist name (only AAPL exists today, so `ml_movement` is a no-op for MSFT/GOOGL).
- **Features are thin.** No market/sector context, no cross-asset features (VIX, rates, index trend), no relative strength, no earnings-calendar awareness. Volatility annualisation is wrong (`sqrt(252/5)` on a daily std) — harmless after scaling but confusing.
- **MA crossover is event-based, not state-based.** It emits BUY only on the cross day. If that run is skipped, the portfolio is full, or the name was added to the watchlist mid-trend, the entry is missed until the next cross — possibly months. 20/50 on single stocks also whipsaws.
- **Consensus mode is really "any non-HOLD".** HOLDs are ignored, so a single ML BUY at 55 % confidence opens a 10 % position. 55 % on an uncalibrated 3-class RandomForest is not a meaningful bar.
- **Two different RSIs.** `helpers.RSI` (Wilder/EWM) and `rsi_signals` (simple rolling mean) disagree; the RSI strategy fires every day the level is beyond the threshold rather than on a cross, and has no trend filter.
- **No portfolio construction.** Fixed 10 % notional per name regardless of volatility or correlation; three highly correlated mega-cap tech names; no cash/benchmark comparison in the daily report. `optimize_portfolio` uses mean historical returns + sample covariance → max Sharpe, which is known to produce unstable, concentrated weights.
- **Fundamental work is disconnected from trading.** QVM, the stock finder and the valuation models produce snapshots from `yfinance.info` (not point-in-time, cannot be backtested) and never feed the watchlist. QVM momentum is raw 52-week change (no skip-month); value is P/E and P/B only; negative forward P/E ranks as "cheapest".

---

## 3. Plan

Each phase is sized so its items can become individual issues / Ralph stories.

### Phase 0 — Make the numbers trustworthy

- [x] Fix weekly label to `shift(-5)`; regression tests assert the label window and that today's return cannot move it. Same fix in `nn_predictor.py`.
- [x] Replace `ffill().bfill()` with dropping rows whose features cannot be computed. `predict_latest` now refuses to predict when the latest bar is one of them.
- [x] Time-based split in `fin_statement_classifier` (`split_by_report_date`).
- [x] Purged walk-forward CV (`purged_time_series_cv`, gap = label horizon); scaler inside an sklearn `Pipeline` so it is fitted per fold. Used by `compare_classifiers`. Holdout splits take `label_horizon` and purge train rows whose labels reach into the holdout.
- [x] Three-way temporal split (`temporal_three_way_split`): `compare_variants` ranks on validation and scores only the winner on the final test period (`--validation-months`).
- [x] `/api/v1/train` evaluates on models that never saw the holdout, then saves models fitted on all labelled data; metadata records `trained_through`, `training_samples` and the holdout metrics.
- [x] Remove the mutable default `classifier=RandomForestClassifier(...)` arguments.
- [x] `compute_holdout_metrics` pairs daily and weekly predictions by date (they were paired by position, which misaligned them at the end of the series).
- [x] Re-run `stock-analysis-evaluate variants` — results below.
- [x] Saved per-ticker models (`models/AAPL_*`, February 2025, old pipeline) are no longer used: `ml_movement` was taken out of `config/portfolio.yaml` (D1). Retrain with `POST /api/v1/train/<ticker>` before putting it back.
- [x] Survivorship bias: `--universe sp500` uses historical index members rebuilt from the Wikipedia change log (`market_data/index_membership.py`). Yahoo has no prices for many delisted names, so part of the bias remains; the report shows the coverage per year.

**Before / after** — `stock-analysis-evaluate variants --ticker ^GSPC --holdout-months 12`, winning variant `threshold_1pct_extra`, same test period (from 2025-11-06, 252 days):

| Metric | Before | After |
|---|---|---|
| Daily accuracy | 77.38 % | 77.78 % |
| Weekly accuracy | 48.81 % | 40.64 % |
| Combined accuracy | 78.57 % | 78.63 % |
| Combined signal rate | 0.79 % | 0.81 % |

Reading the numbers: weekly accuracy fell by about eight points once the label stopped containing a feature. The ~78 % daily and combined accuracy is very unlikely to be skill — at a 1 % threshold most index days are "neutral" and the combined signal fires on under 1 % of days, which is what a model that rarely leaves the majority class looks like (the majority-class baseline was not computed here; add it to the report in Phase 1). Ranking variants by accuracy therefore rewards the widest threshold; Phase 1 replaces it with return-based metrics against a benchmark. One ticker and one test year — indicative only.

### Phase 1 — One honest backtester

The paper-trading loop already has the right shape (decide at close → fill at next open → costs). Reuse it instead of maintaining a second simulator.

- [x] `stock_analysis/backtest/` engine that replays history day by day through the existing `SignalStrategy.evaluate` → `combine_signals` → `decide` → `EquityCostModel` path with an in-memory portfolio state. Do not change the `DecisionResult` / `FillResult` contracts.
- [x] Standard report for every strategy: CAGR, volatility, Sharpe, Sortino, max drawdown, turnover, cost drag, exposure, hit rate — always next to buy-and-hold of the same universe and SPY.
- [x] Walk-forward mode for ML strategies (`ml_walk_forward`: retrain every N sessions on an expanding window, predict only the sessions that follow).
- [x] Guard against over-fitting to the backtest: every configuration is logged to `data/backtests/runs.jsonl`; deflated Sharpe ratio once two or more configurations exist for the same universe and period; block-bootstrap interval on Sharpe and on the comparison with buy-and-hold; probability of backtest overfitting (combinatorially symmetric cross-validation over ten blocks) from the daily returns saved with each logged run.
- [x] `stock-analysis-evaluate backtest --strategies … --tickers …` CLI; the leveraged mini-future simulation is opt-in (`train --leveraged-sim`).
- [x] Benchmark-relative return in the daily paper report (`defaults.benchmark`, SPY by default).
- [x] Majority-class baseline next to classifier accuracy in `variants` output and in `compute_holdout_metrics`.
- [x] RSI (`rsi`) and the ensemble classifier (`ensemble_walk_forward`) run through `backtest`; the legacy `strategies` command is removed.

**Phase 1 results — first baseline (2026-11-10).** AAPL, MSFT, GOOGL from `config/portfolio.yaml`, 2016-11-10 to 2026-11-09 (2512 sessions), $100,000, fills at next Open, costs from the config.

| | MA crossover 20/50 | ML walk-forward (retrain every 63 sessions) | Buy & hold (universe) | Buy & hold (SPY) |
|---|---:|---:|---:|---:|
| CAGR | 4.00 % | 4.95 % | 26.84 % | 15.11 % |
| Sharpe | 0.77 | 0.88 | 1.08 | 0.87 |
| Max drawdown | −12.00 % | −9.55 % | −34.33 % | −33.72 % |
| Cost drag (share of initial cash, whole period) | 1.12 % | 4.31 % | — | — |
| Resamples where Sharpe beats universe buy & hold | 5 % | 8 % | — | — |

Reading the numbers: neither strategy beats holding its own universe on risk-adjusted return, so by the success criterion in section 4 neither would be promoted today. Both are invested only about 20 % of the time on average (the ML run: 19.73 %), which explains the low return and the shallow drawdowns more than any timing skill does: position sizing is 10 % per ticker over three tickers, so most cash sits idle. That makes sizing and universe width (Phase 2) the first thing to fix before judging the signals themselves. The universe is three stocks picked with hindsight, so the buy-and-hold column is flattered; SPY is the fairer yardstick, and against it the strategies have a similar Sharpe at a third of the return. These two runs were made with `--no-log` and are not counted as trials.

### Phase 2 — Improve what we have

**Trend following**
- [x] State-based trend signal (`trend_state`): BUY for as long as the uptrend holds, SELL for as long as it is broken. `decide()` skips a BUY when already long and a SELL when flat, so a missed run, a full portfolio or a ticker added mid-trend is corrected on the next run.
- [x] Time-series momentum (average of 3/6/12-month returns) with the price above its own 200-day average for entry, plus an optional market-regime filter (`regime_filter`: no new BUYs while SPY is below its 200-day average). Evaluated on 20 ETFs — results below.
- [x] Trailing stops: `portfolio.stop_type: atr_trailing | pct_trailing`; the stop level also blocks re-entry. A no-trade band for `trend_state` (`trend_entry_threshold` / `trend_exit_threshold`).

**Phase 2 results — trend following on the ETF universe (2026-11-10).** `config/portfolio-etf-trend.yaml`: 20 ETFs (US equity and sectors, international, bonds, gold, commodities, property), 2016-11-10 to 2026-11-09, $100,000, fills at next Open, $1 commission + 0.05 % slippage. All four runs are logged trials.

| | A: MA cross 20/50, fixed 5 % | B: `trend_state`, fixed 5 % | C: `trend_state`, vol-target | D: C + regime filter | Buy & hold (universe) | Buy & hold (SPY) |
|---|---:|---:|---:|---:|---:|---:|
| CAGR | 5.29 % | 6.62 % | 7.11 % | 6.84 % | 11.59 % | 15.11 % |
| Volatility | 7.71 % | 8.50 % | 8.20 % | 8.16 % | 14.30 % | 18.03 % |
| Sharpe | 0.71 | 0.80 | 0.88 | 0.85 | 0.84 | 0.87 |
| Max drawdown | −14.81 % | −13.43 % | −12.71 % | −12.67 % | −28.45 % | −33.72 % |
| Avg exposure | 64 % | 76 % | 81 % | 79 % | 100 % | 100 % |
| Trades | 1070 | 1130 | 930 | 870 | — | — |
| Resamples where Sharpe beats universe buy & hold | 25 % | 36 % | 55 % | 50 % | — | — |

Reading the numbers:
- Against the Phase 1 baseline the portfolio is now actually invested (about 80 % against 20 %) and CAGR went from 4–5 % to about 7 %.
- The best variant (C) has the same risk-adjusted return as buying and holding the same ETFs (Sharpe 0.88 against 0.84) with less than half the drawdown and about 60 % of the return. The bootstrap puts it ahead of buy-and-hold in 55 % of resamples — a coin flip. This is what trend following usually delivers: a smoother ride, not a higher return, in a decade when almost everything went up.
- The state-based signal beats the 20/50 cross (0.80 against 0.71 at the same sizing), and volatility sizing adds a little more. The regime filter did not help here: each ETF's own trend rule already keeps it out of falling markets.
- With four configurations the probability of backtest overfitting is 49 %, so the order between B, C and D is not reliable. Do not tune further on this period.
- By the success criterion in section 4 this is **not yet a promotion**: C is ahead of its universe only within noise, on a period that was also used to compare the variants. It is a reasonable candidate for a second paper portfolio run side by side with the existing one (it uses its own portfolio name), judged on what it does from here on.
- `trend_state` wins only 31 % of its round trips and trades each ETF about every four months — many small losses around the point where momentum crosses zero. A wider no-trade band is the obvious next experiment, but it has to be tested on a period not used here.

**Found on the way:** several BUYs decided in the same run each saw the full cash balance and all free slots, because orders fill the next day. With three tickers at 10 % this never mattered; with twenty it let the portfolio queue more than its cash and more than `max_positions`. Both the daily job and the backtester now count orders already queued in the run (`with_queued_order`).

**Phase 2 results — second round (2026-11-12).** Test protocol (D3): variants are compared on 2007-03-01 to 2016-11-09; only the best one is then run once on 2016-11-10 to 2026-11-09, which was already used in the first round and is not used for choosing any more.

ETF trend portfolio (`config/portfolio-etf-trend.yaml`), tuning period 2007-03-01 to 2016-11-09 (2,444 sessions, includes 2008):

| | V0: config as is | V1: no-trade band ±2 % | V2: ATR trailing stop | V3: HRP sizing | V4: band + ATR stop | Buy & hold (universe) | Buy & hold (SPY) |
|---|---:|---:|---:|---:|---:|---:|---:|
| CAGR | 4.71 % | 4.93 % | 2.12 % | 2.86 % | 2.14 % | 6.20 % | 6.76 % |
| Volatility | 7.70 % | 7.69 % | 6.81 % | 3.33 % | 6.78 % | 14.92 % | 21.08 % |
| Sharpe | 0.64 | 0.66 | 0.34 | 0.86 | 0.35 | 0.48 | 0.42 |
| Max drawdown | −11.68 % | −11.41 % | −15.12 % | −6.08 % | −14.33 % | −41.00 % | −55.19 % |
| Trades | 714 | 350 | 2,698 | 812 | 2,512 | — | — |
| Cost drag | 3.13 % | 1.43 % | 12.80 % | 2.28 % | 12.09 % | — | — |
| Resamples where Sharpe beats universe buy & hold | 70 % | 73 % | 32 % | 86 % | 32 % | — | — |

Probability of backtest overfitting over the five: 1 %, so the order is stable within this period. Final check of the winner (V3, HRP) on 2016-11-10 to 2026-11-09: CAGR 3.76 %, volatility 5.12 %, Sharpe 0.75, max drawdown −11.33 %, against Sharpe 0.84 for buy-and-hold of the same ETFs and 0.85 for the config as is (run D above). It beats buy-and-hold in 36 % of resamples.

Reading the numbers:
- **The config as is (V0) passed a real out-of-sample test.** It was chosen on 2016–2026; on 2007–2016, which it never saw, it beat buy-and-hold of its universe (Sharpe 0.64 against 0.48, in 70 % of resamples) with a drawdown of −12 % against −41 %. Together with 2016–2026 (a tie on Sharpe, half the drawdown) this is the best evidence in this document. The paper portfolio keeps this config.
- **HRP did not carry over.** On 2007–2016 it won by holding mostly bonds during a bond bull market (3 % volatility). On 2016–2026 it lost to buy-and-hold. Not adopted.
- **The no-trade band halves trading and cost** for about the same result. A cheap improvement, but too small to separate from noise; it is not switched on in the paper config.
- **The trailing stops as built hurt badly** (cost drag 12–13 %). The stop level trails the highest close of the last 63 sessions regardless of entry, and re-entry is allowed as soon as the price is back above it, so a state-based BUY buys straight back and is stopped out again (open task O1). Stops are opt-in; nothing live uses them.

Pooled model (`ml_pooled`, `config/research-pooled.yaml`), one untuned run with default parameters: 150 tickers drawn at random from all 884 S&P 500 members since 1976, of which 105 have Yahoo prices, traded only while they were index members; 2016-11-10 to 2026-11-09.

| | `ml_pooled` | Buy & hold (members, equal weight) | Buy & hold (SPY) |
|---|---:|---:|---:|
| CAGR | 7.66 % | 13.72 % | 15.11 % |
| Volatility | 16.20 % | 18.19 % | 18.03 % |
| Sharpe | 0.54 | 0.80 | 0.87 |
| Max drawdown | −37.86 % | −39.20 % | −33.72 % |
| Turnover / cost drag | 9.2×/yr / 8.39 % | — | — |
| Avg exposure | 50 % | 100 % | 100 % |

- **It fails.** The model ranks a little better than chance (mean daily rank IC 0.013; the top 10 names beat the universe by 0.25 % per 10 sessions before costs; 2025 was the only strong year), but after costs and with half the money idle it trails buy-and-hold in 94 % of resamples. Calibration works (the predicted probabilities match observed frequencies by decile), so the model knows how little it knows.
- Volatility and beta features carry the most weight; momentum features matter less than expected.
- This run used `--no-kill-switch`. The first run with the config's risk limits tripped the kill switch on 2020-03-09 (a 6.7 % day) and bought nothing for the next six years; both runs are logged trials.
- Survivorship bias is reduced, not gone: Yahoo has no prices for 45 of the 150 sampled names (mostly acquired or delisted companies), and coverage of members is 61–71 % before 2017.

QVM (`qvm`) has not been run on real data: it needs `EDGAR_IDENTITY` set (open task O2).

**ML movement model**
- [x] One pooled cross-sectional LightGBM model for the whole universe (`ai/cross_sectional/`, `ml_pooled` in the backtester), retrained walk-forward.
- [x] Target: top fifth of the 10-session forward return among that date's members (`labels.py`). Triple-barrier not done.
- [x] Features: momentum and reversal over several horizons, volatility and volume regime, distance from the 52-week high, beta-adjusted market and sector relative strength, ranked per date; VIX, 10-year yield and index trend as levels. Days since earnings is supported but off by default (free earnings history is too short).
- [x] Isotonic calibration on a held-out validation year; the entry threshold is the one that earned most net of costs there, or none.
- [x] `retrain-pooled`: champion/challenger retraining, meant for cron / Task Scheduler. Nothing in the daily job reads the saved model yet.
- [x] Gain share and SHAP contribution per feature, rank IC per year and a walk-forward calibration table in the backtest report.

**Signal combination**
- [x] `signal_mode: quorum` (N of M agree, none disagree) and `score` (weighted blend of per-strategy scores, with smaller positions for weaker blends). Existing modes unchanged.

**QVM / fundamentals**
- [x] QVM factors: 12-1 momentum, value as a composite of yields, quality with gross profitability, accruals and earnings stability; winsorised and ranked within sector (`factors/`).
- [x] Point-in-time fundamentals from EDGAR (`ai/fundamentals/point_in_time.py`): known from filing date plus a lag, trailing twelve months from quarterlies, restatements do not leak backwards. Needs `EDGAR_IDENTITY`.
- [x] Monthly-rebalanced QVM strategy in the backtester (`qvm`) and `qvm-watchlist` for a positions snippet.
- [x] EPV / reverse-DCF margin of safety with sector discount rates, as an optional value input (`qvm_margin_of_safety`).

**Portfolio construction and risk**
- [x] Volatility-targeted position sizing per position (`portfolio.sizing: vol_target`: weight = `risk_per_position` / annualised volatility, capped at `max_position_pct`). Rebalancing and a portfolio-level volatility target were added later (next two items).
- [x] `portfolio.sizing: min_variance | hrp` on a Ledoit-Wolf-shrunk covariance (PyPortfolioOpt); `target_volatility` for the portfolio as a whole.
- [x] `max_sector_pct`, a correlation cap on new positions, `rebalance_band` with a minimum trade size, and a 1 % cash buffer. No explicit turnover budget: the band and the minimum trade size are what limit turnover.
- [x] Wider universe as a separate config (`config/portfolio-etf-trend.yaml`, 20 liquid ETFs, portfolio name `paper-etf-trend`). The `trader` service runs it next to `config/portfolio.yaml` (D2).

### Phase 3 — Modern approaches worth exploring

Ordered by expected value for a hobby-scale, daily-bar, free-data setup. Each is a spike: build it in the Phase 1 backtester, compare against the Phase 2 baseline, adopt only if it survives costs out of sample.

| Approach | What it is | Why it fits here | Caveat |
|---|---|---|---|
| **Cross-sectional learning-to-rank** | LightGBM `lambdarank` ranking stocks per date; trade top-k | Natural next step from the pooled model; directly optimises the thing we trade | Needs a universe of 100+ names |
| **Meta-labeling** | Keep a simple primary signal (trend/QVM); a second model predicts whether *that signal's* trade will pay off and sizes accordingly | Reuses existing signals; easier problem than raw direction | Needs enough primary-signal trades to train on |
| **Regime detection** | Hidden Markov model or simple vol/trend states to scale exposure | statsmodels already present in `diagnostics/`; improves drawdown more than return | Keep it to 2–3 states; easy to over-fit |
| **Post-earnings drift and revisions** | Earnings surprise, guidance and estimate-revision signals | Well-documented anomaly; EDGAR 8-K dates are free | Consensus-estimate history is limited on free sources |
| **LLM text signals** | Score 10-K/10-Q/8-K changes, risk-factor diffs and earnings-call tone with an LLM; use as features | EDGAR adapter exists; filings are free and timestamped | Use only filings dated after the model's training cutoff when backtesting, otherwise the model "knows" the outcome |
| **Additional factor sleeves** | Low-volatility / betting-against-beta, short-term reversal, 52-week-high | Cheap to add once the ranking backtester exists; diversify momentum | Reversal is turnover-heavy; cost-sensitive |
| **Pairs / statistical arbitrage** | Cointegration-based spread trading | Stationarity helpers already exist | Needs shorting; current portfolio is long-only |
| **Conformal prediction** | Distribution-free prediction intervals; trade only when the interval excludes zero | Principled replacement for the hand-set confidence threshold | Coverage guarantees weaken under regime shift |
| **Time-series foundation models** | Pretrained forecasters (Chronos, TimesFM, etc.) zero-shot on returns or volatility | Low effort to benchmark | Evidence for return forecasting is weak; volatility forecasting is the more promising use |
| **Oslo Børs factor strategies** | Run QVM/momentum on Norwegian equities | Titlon provides academic-quality Norwegian price and factor data; `oslo_tickers.py` exists | Liquidity and spreads are much worse than US large-cap |

### Explicitly not recommended

| Idea | Reason |
|---|---|
| Deep sequence models (LSTM/Transformer) trained from scratch on daily bars | Too little data per asset; tree models on good features win at this scale |
| Reinforcement-learning trading agents | Sample-inefficient, very hard to validate, no edge over supervised + rules here |
| Intraday / high-frequency strategies | Data, latency and cost model are all daily-bar; yfinance intraday history is short |
| Leveraged mini-future / bull-bear execution of ML signals | Financing and spread dominate; the cost analysis in this repo already shows the decay |
| More classifier types in the grid search | Model class is not the bottleneck; labels, features and validation are |

---

## 4. Suggested order

1. Phase 0 in full (small, mechanical, changes every later conclusion).
2. Phase 1 backtest engine + benchmark report.
3. Re-evaluate the current MA and ML strategies in the new engine — this sets the baseline and may show the current ML signal has no edge.
4. Phase 2: state-based trend + vol-targeted sizing + wider universe (largest practical improvement to the paper portfolio).
5. Phase 2: pooled cross-sectional model and point-in-time QVM.
6. Phase 3 spikes, one at a time, starting with learning-to-rank and meta-labeling.

Success criterion throughout: a strategy is promoted to the paper portfolio only if it beats buy-and-hold of its own universe on risk-adjusted return, net of costs, on a test period that was not used to choose it.

---

## 5. Open tasks

Status on 2026-11-12: Phases 0–2 are done; every checkbox in section 3 is ticked. Results are in the Phase 1 and Phase 2 sections. What is left is listed here in the order it should be picked up. Tooling work (visualisation clean-up, Polars spike) is tracked separately in [issues/README.md](./issues/README.md).

### Decisions taken in Phase 2

| # | Decision | Outcome |
|---|---|---|
| D1 | Keep `ml_movement` in `config/portfolio.yaml`? | Taken out: no edge in the walk-forward backtest, and the only saved models predate the label fix. |
| D2 | Run the ETF trend portfolio in paper trading? | Yes: the `trader` service runs `config/portfolio.yaml` and `config/portfolio-etf-trend.yaml` side by side. |
| D3 | Which period is the untouched test period? | Variants are compared on 2007-03-01 to 2016-11-09; only the winner is run once on 2016-11-10 to 2026-11-09. Both periods have now been used; the next clean test is paper trading from November 2026 on. |

### Next tasks, in order

| # | Task | Size | Notes |
|---|---|---|---|
| O1 | Fix re-entry after a trailing stop. Today the stop level trails the 63-day high whatever the entry was, and re-entry is allowed as soon as the price is back above it, so a state-based signal churns (12 % cost drag). Options: a cool-down after a stop-out, or re-entry only on a new N-day high. Test on 2007–2016 only. | Small | Stops are opt-in; nothing live uses them |
| O2 | Run `qvm` in the backtester on a historical S&P 500 sample, with `EDGAR_IDENTITY="Name email"` set. Same protocol: compare variants on 2007–2016, final check on 2016–2026. | Medium (mostly download time) | Needs the owner's SEC contact identity |
| O3 | Review the two paper portfolios after six and twelve months against SPY and their own universes, using the benchmark section of the daily report. This is the only test period nothing has been tuned on. | Small, recurring | From 2026-11 |
| O4 | Decide whether to keep the no-trade band (`trend_entry_threshold: 0.02`, `trend_exit_threshold: -0.02`) in the ETF config: half the trades and cost for the same result on 2007–2016. Not yet run on 2016–2026. | Small | |
| O5 | Pooled model follow-ups, only if Phase 3 learning-to-rank does not replace it: a longer label horizon (20–60 sessions), a larger universe (the full index, not a 150-name sample), triple-barrier labels. The model is calibrated but its ranking edge (rank IC 0.013) does not survive costs. | Medium | `retrain-pooled` exists, but nothing reads the saved model |
| O6 | Price history for delisted companies. Yahoo has none for about a third of past S&P 500 members, so survivorship bias remains before 2017 (61–71 % coverage). Needs another data source. | Medium | Data source to be found |

After these: the Phase 3 spikes, starting with learning-to-rank (it builds directly on the pooled model's features, labels and historical universe) and meta-labeling on top of `trend_state`.

### Small fixes

| # | Fix | Where |
|---|---|---|
| F5 | `pd.Timedelta(days=…)` warns about the NumPy "generic" unit under Python 3.14 (also inside `pandas_market_calendars` and `exchange_calendars`). Recheck after the next pandas / NumPy upgrade. | `stock_analysis/research/analyze_stock_data.py:75-76` |
