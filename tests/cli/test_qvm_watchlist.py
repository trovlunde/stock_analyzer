"""`qvm-watchlist` with fake prices, filings and sectors — no network."""

import argparse

import pandas as pd
import pytest
import yaml

from stock_analysis.ai.fundamentals import point_in_time as pit
from stock_analysis.ai.fundamentals.point_in_time import (
    EdgarIdentityMissing,
    PointInTimeFundamentals,
)
from stock_analysis.cli.qvm import add_qvm_parser, run_qvm_watchlist_command
from tests.factors.synthetic import SECTORS, TICKERS, make_fact_frames, make_prices


class FakeProvider:
    def __init__(self, unknown=()):
        self.unknown = set(unknown)
        self.requested: list[tuple[str, str]] = []

    def get_history(self, ticker, period="1y", start=None, end=None):
        self.requested.append((ticker, period))
        if ticker in self.unknown or ticker not in TICKERS:
            return pd.DataFrame()
        frame = make_prices(TICKERS.index(ticker))
        frame.index = frame.index.tz_localize("America/New_York")
        return frame


def _parse(*argv):
    parser = argparse.ArgumentParser()
    add_qvm_parser(parser.add_subparsers(dest="command", required=True))
    return parser.parse_args(["qvm-watchlist", *argv])


def _run(args, provider=None):
    return run_qvm_watchlist_command(
        args,
        provider=provider or FakeProvider(),
        fundamentals=PointInTimeFundamentals.from_frames(make_fact_frames()),
        sector_lookup=lambda tickers: {t: SECTORS[t] for t in tickers},
    )


def test_prints_a_positions_snippet_that_parses_as_yaml(tmp_path):
    output = tmp_path / "out" / "watchlist.yaml"
    provider = FakeProvider()

    text = _run(_parse("--tickers", *[t.lower() for t in TICKERS], "--top", "3", "--output", str(output)), provider)

    assert text.startswith("# QVM watchlist as of 2023-08-11 (top 3 of 8 ranked")
    parsed = yaml.safe_load(text)
    assert list(parsed) == ["positions"]
    assert len(parsed["positions"]) == 3
    assert all(set(entry) == {"ticker"} and entry["ticker"] in TICKERS for entry in parsed["positions"])
    assert "QVM " in text and "(Q " in text
    assert output.read_text(encoding="utf-8") == text
    assert {period for _, period in provider.requested} == {"2y"}


def test_watchlist_order_matches_the_scores(tmp_path):
    from stock_analysis.backtest.benchmarks import load_price_data
    from stock_analysis.factors.qvm_strategy import QvmScoreProvider, qvm_watchlist

    text = _run(_parse("--tickers", *TICKERS, "--top", "8"))

    price_data = load_price_data(TICKERS, FakeProvider(), period="2y")
    expected = qvm_watchlist(
        QvmScoreProvider(price_data, PointInTimeFundamentals.from_frames(make_fact_frames()), SECTORS),
        top=8,
    )
    listed = [entry["ticker"] for entry in yaml.safe_load(text)["positions"]]
    assert listed == list(expected["ticker"])


def test_tickers_without_prices_are_left_out_and_small_sectors_unranked():
    text = _run(_parse("--tickers", *TICKERS, "NOPE"), FakeProvider(unknown={"E3"}))

    listed = [entry["ticker"] for entry in yaml.safe_load(text)["positions"]]
    # Energy is down to three names, below the default minimum sector size of four
    assert sorted(listed) == ["T0", "T1", "T2", "T3"]

    relaxed = _run(_parse("--tickers", *TICKERS, "--min-sector-size", "3"), FakeProvider(unknown={"E3"}))
    assert len(yaml.safe_load(relaxed)["positions"]) == 7


def test_nothing_rankable_gives_an_empty_list():
    text = _run(_parse("--tickers", "T0", "T1"))

    assert yaml.safe_load(text) == {"positions": []}


def test_no_price_data_at_all_is_an_error():
    with pytest.raises(SystemExit, match="No price data"):
        _run(_parse("--tickers", "NOPE"))


def test_universe_and_tickers_are_exclusive_and_one_is_required():
    with pytest.raises(SystemExit):
        _parse()
    with pytest.raises(SystemExit):
        _parse("--tickers", "T0", "--universe", "sp500")
    assert _parse("--universe", "sp500").universe == "sp500"


def test_default_fundamentals_need_an_edgar_identity(monkeypatch):
    monkeypatch.delenv("EDGAR_IDENTITY", raising=False)
    monkeypatch.setattr(pit, "load_dotenv", lambda: None)
    provider = FakeProvider()

    with pytest.raises(EdgarIdentityMissing):
        run_qvm_watchlist_command(_parse("--tickers", "T0"), provider=provider)

    assert provider.requested == []  # failed before downloading anything
