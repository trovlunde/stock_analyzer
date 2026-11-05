# Trader container readiness

Coding tasks remaining before the `trader` compose service runs unattended.
Manual/ops steps (host, `.env`, model training, 20-day paper soak, live cutover) are out of scope here.

## 1. Add `.dockerignore` — done

`COPY . .` in the Dockerfile copied `.env` (secrets) into an image layer, plus the
host's Windows `.venv`, `.cache`, `data/` (SQLite DB) and `models/`.

- [x] `.dockerignore` excludes secrets, venvs, caches, runtime data, VCS/agent dirs, tests
- [x] `data/` and `models/` still arrive via compose volume mounts

## 2. Mark failed runs as failed, not orphaned — done

Any exception between `store.start_run()` and `store.finish_run()` in
`stock_analysis/jobs/daily.py` (yfinance outage, missing Alpaca keys, Alpaca API
error, `LiveTradingBlocked`) leaves the run row `running`. The next run treats it as
an unclean shutdown and trips the kill switch — one data hiccup blocks BUYs until a
manual clear and resets the 20-clean-run live-gate streak.

- [x] Wrap the post-`start_run` body so a handled exception calls `finish_run(run_id, "failed", ...)` and re-raises
- [x] A failed run does not trip the kill switch on the next run
- [x] A failed run still breaks the consecutive-clean-run count for the live gate
- [x] Real crashes (process killed mid-run) still trip the switch via orphan detection
- [x] Tests in `tests/jobs/test_daily.py`

## 3. Handle stale / missing ML models

Only AAPL models exist (Feb 2025), predating the `pandas-ta` feature migration.
MSFT/GOOGL have none.

- [ ] Detect feature-set mismatch between a saved model and current `predict_latest` features → HOLD with a clear reason (not silent mis-prediction)
- [ ] Log/alert once per run which configured tickers have no usable model

## 4. Pin uv image in Dockerfile

- [ ] Replace `ghcr.io/astral-sh/uv:latest` with a pinned version

## 5. Trader-appropriate healthcheck

The Dockerfile's `HEALTHCHECK` curls :5000; only overridden for `trader` in compose.

- [ ] Move the web healthcheck to compose (or make it service-agnostic) so `trader` run outside compose isn't marked unhealthy

## 6. Late-wake cut-off in scheduler (nice to have)

If the host sleeps past 16:05 ET, `run_forever` fires as soon as it wakes, at any hour.

- [ ] Skip (and alert) a missed slot if waking after the next session's open
