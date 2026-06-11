from datetime import date

import pandas as pd
import pytest

from stock_analysis.storage.portfolio_store import PortfolioStore
from stock_analysis.storage.schema import create_engine_for_url, init_db
from stock_analysis.trading.cost_models import EquityCostModel
from stock_analysis.trading.decision import OrderIntent
from stock_analysis.trading.paper_broker import PaperBroker, open_price_for_date


def test_open_price_for_date():
    index = pd.to_datetime(["2024-06-03", "2024-06-04"])
    df = pd.DataFrame({"Open": [100.0, 101.0], "Close": [100.5, 101.5]}, index=index)
    assert open_price_for_date(df, date(2024, 6, 4)) == 101.0


@pytest.fixture
def broker_setup(tmp_path):
    engine = create_engine_for_url(f"sqlite:///{tmp_path / 'broker.db'}")
    init_db(engine)
    store = PortfolioStore(engine)
    state = store.get_or_create_portfolio("paper-us", 10_000)
    broker = PaperBroker(store, EquityCostModel())
    return broker, store, state


def test_fill_pending_buy_order(broker_setup):
    broker, store, state = broker_setup
    run_id = store.start_run(state.id, dry_run=False)
    store.create_pending_order(
        state.id, run_id, "AAPL", "BUY", 10, "test", fill_date=date(2024, 6, 3)
    )

    index = pd.to_datetime(["2024-06-03"])
    price_data = {
        "AAPL": pd.DataFrame({"Open": [100.0], "Close": [101.0]}, index=index),
    }
    fills = broker.fill_pending_orders(
        state, price_data, session_date=date(2024, 6, 3), dry_run=False
    )
    assert len(fills) == 1
    assert state.cash < 10_000
    assert len(state.positions) == 1
