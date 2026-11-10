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
- [ ] Existing saved models (`models/AAPL_*`, dated February 2025) were trained with the old pipeline; retrain before relying on `ml_movement`. Still not done on 2026-11-10 — see decision D1 in section 5.
- [ ] Not addressed: survivorship bias in S&P 500 pooled training (needs historical constituents).

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
- [ ] `stock-analysis-evaluate strategies` (ensemble vs RSI vs MA) still runs on the old research simulator and is labelled legacy. To compare those signals fairly, register RSI and the ensemble as `SignalStrategy` implementations and run them through `backtest`; then delete the command.

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
- [ ] ATR-based stops / trailing exits instead of fixed-percentage stop from average cost.

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

**ML movement model**
- [ ] One pooled cross-sectional model for the whole universe instead of one model per ticker; LightGBM as default (already a dependency).
- [ ] Change the target: 5–20 day forward return *relative to the universe* (rank or top-quintile label) rather than absolute next-day direction. Alternative label: triple-barrier (profit target / stop / time-out).
- [ ] Features: multi-horizon momentum and reversal, volatility and volume regime, distance from 52-week high, sector and market relative strength, VIX and rates, days to/from earnings; cross-sectionally ranked per date.
- [ ] Probability calibration (isotonic / Platt on the validation fold) so `ml_confidence_threshold` means something; choose the threshold by net return on validation, not by hand.
- [ ] Scheduled retraining job with a champion/challenger check before a new model replaces the old one.
- [ ] Feature importance / SHAP and per-year performance breakdown in the evaluation report to catch regime dependence.

**Signal combination**
- [ ] Add a true consensus mode (N of M must be actionable and agree) and a score-based mode where each strategy returns a continuous score that is blended and then sized. Keep the existing modes for backward compatibility.

**QVM / fundamentals**
- [ ] Momentum = 12-1 month total return; value = composite of earnings yield, FCF yield, EV/EBITDA, book-to-price (yields, so negatives rank correctly); quality adds gross profitability, accruals, earnings stability; winsorise before ranking.
- [ ] Point-in-time fundamentals from EDGAR (filing dates are available through the existing `edgar_adapter`) so the screen can be backtested with realistic reporting lag.
- [ ] Monthly-rebalanced QVM strategy in the backtester (top decile, sector-neutral, equal or inverse-vol weight), then as a watchlist generator for the paper portfolio.
- [ ] Use the valuation models as a margin-of-safety feature (price vs. EPV / reverse-DCF implied growth), with sector-appropriate discount rates instead of a flat 10 %.

**Portfolio construction and risk**
- [x] Volatility-targeted position sizing per position (`portfolio.sizing: vol_target`: weight = `risk_per_position` / annualised volatility, capped at `max_position_pct`). Sized once at entry; no rebalancing and no portfolio-level volatility target yet.
- [ ] Replace max-Sharpe on sample estimates with Ledoit-Wolf shrinkage + minimum-variance or Hierarchical Risk Parity (both in PyPortfolioOpt, already a dependency).
- [ ] Sector / single-name / correlation caps; turnover budget; rebalance bands.
- [x] Wider universe as a separate config (`config/portfolio-etf-trend.yaml`, 20 liquid ETFs, portfolio name `paper-etf-trend`). `config/portfolio.yaml` is unchanged; running the new config in the daily job is a decision for the owner.

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

Status on 2026-11-10: Phase 0 and Phase 1 are done apart from the items listed here; Phase 2 has its first slice (trend signal, volatility sizing, regime filter, ETF universe). The checkboxes in section 3 are the full record; this is the same open work in the order it should be picked up. Tooling work (visualisation clean-up, Polars spike) is tracked separately in [issues/README.md](./issues/README.md).

### Decisions for the owner

These need a choice, not code.

| # | Decision | Background |
|---|---|---|
| D1 | Keep `ml_movement` in `config/portfolio.yaml`, or take it out? | The only saved models are `models/AAPL_*` from February 2025, trained with the old leaky pipeline. MSFT and GOOGL have none, so the strategy does nothing for them. The walk-forward backtest showed no edge. Keeping it means retraining all three with `POST /api/v1/train/<ticker>`; the pooled model (T1) will replace it either way. |
| D2 | Run the ETF trend portfolio in paper trading next to the existing one? | `config/portfolio-etf-trend.yaml` has its own portfolio name and state. It matched buy-and-hold on risk-adjusted return with half the drawdown, but is not a proven win. Running it costs a second scheduler entry and gives real out-of-sample evidence. |
| D3 | Which period is the untouched test period from now on? | 2016-11-10 to 2026-11-09 has been used to compare four trend variants and must not be used for more tuning. Most of the 20 ETFs have history back to 2007 or earlier, so one option is to tune on data up to 2016-11-09 and keep the last ten years for a single final check. T3 and T4 depend on this. |

### Next tasks, in order

| # | Task | Size | Depends on | Section 3 items |
|---|---|---|---|---|
| T1 | Pooled cross-sectional ML model: one LightGBM model for the whole universe, target = 5–20 day forward return relative to the universe, features ranked per date. Model, target and features are one piece of work — changing one without the others is not worth evaluating. Run it through `backtest` as a walk-forward strategy on a universe of 100+ stocks. | Large | A stock universe without survivorship bias (T9), or accept and state the bias | ML movement model, items 1–3 |
| T2 | Probability calibration on the validation fold, and choose the confidence threshold by net return on validation instead of by hand. | Medium | T1 | ML movement model, item 4 |
| T3 | Wider no-trade band for `trend_state`. It wins 31 % of round trips and trades each ETF about every four months; most losses are small ones around zero momentum. | Small | D3 | Trend following (new) |
| T4 | ATR-based or trailing stops in place of the fixed-percentage stop from average cost. | Medium | D3 | Trend following, item 3 |
| T5 | Signal combination: a true consensus mode (N of M actionable and in agreement) and a score-based mode. Keep the existing modes. | Medium | — | Signal combination |
| T6 | Point-in-time fundamentals from EDGAR filing dates, so the QVM screen can be backtested. | Large | — | QVM, item 2 |
| T7 | QVM factor definitions (12-1 momentum, composite value from yields, quality additions, winsorising), then a monthly-rebalanced QVM strategy in the backtester. | Large | T6 | QVM, items 1, 3, 4 |
| T8 | Portfolio construction: Ledoit-Wolf + minimum-variance or HRP; sector, single-name and correlation caps; rebalance bands; a portfolio-level volatility target (sizing is per position and set once at entry today). | Large | — | Portfolio construction, items 2–3 |
| T9 | Historical index constituents for pooled training, to remove survivorship bias. | Medium | A data source | Phase 0, last item |
| T10 | Register RSI and the ensemble classifier as `SignalStrategy` implementations, compare them through `backtest`, then delete the legacy `strategies` command. | Medium | — | Phase 1, last item |
| T11 | Scheduled retraining with a champion/challenger check; feature importance and per-year breakdown in the evaluation report. | Medium | T1 | ML movement model, items 5–6 |

After these: the Phase 3 spikes, starting with learning-to-rank and meta-labeling.

### Small fixes

| # | Fix | Where |
|---|---|---|
| F1 | No test covers the `/api/v1/train` endpoint (there is no `tests/web/`). It was changed in Phase 0 to evaluate on holdout-blind models and then refit on all data. | `stock_analysis/web/app.py` |
| F2 | Several BUYs in one run can still leave cash slightly negative: each order is sized against cash at the signal day's close, without commission, slippage or a higher Open the next day. Reserve a small buffer or cap the last order. | `stock_analysis/trading/decision.py` |
| F3 | Check whether the live notional ceiling counts BUY orders queued earlier in the same run. It adds only the current order to the portfolio value. | `stock_analysis/jobs/daily.py` |
| F4 | Add `.claude/worktrees/` to `.gitignore` so agent worktrees do not show up as untracked files. | `.gitignore` |
