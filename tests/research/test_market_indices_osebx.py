"""Tests for Oslo ticker resolution (mocked; no live Yahoo required)."""

from unittest.mock import MagicMock, patch

import pytest

from stock_analysis.research.market_indices import MarketIndices
from stock_analysis.research.oslo_tickers import (
    candidates_for,
    resolve_oslo_yahoo_tickers,
    to_yahoo_ol_ticker,
)


_FAKE_OSLO_HTML = """
<html><body>
<table class="wikitable">
<tr><th>*</th><td>Denotes equity certificate</td></tr>
</table>
<table class="wikitable">
<tr><th>Company</th><th>Ticker</th><th>First day of listing</th><th>GICS Industry</th></tr>
<tr><td>Equinor</td><td>OSE:EQNR</td><td>2001</td><td>Energy</td></tr>
<tr><td>Archer</td><td>OSE:ARCHER</td><td>2010</td><td>Energy</td></tr>
<tr><td>Akastor</td><td>OSE:AKA</td><td>2014</td><td>Energy</td></tr>
<tr><td>Golden Ocean</td><td>OSE:GOGL</td><td>2004</td><td>Marine</td></tr>
<tr><td>DeadCo</td><td>OSE:DEAD</td><td>2000</td><td>x</td></tr>
</table>
</body></html>
"""


def test_to_yahoo_ol_ticker_normalizes_ose_prefix():
    assert to_yahoo_ol_ticker("OSE:EQNR") == "EQNR.OL"
    assert to_yahoo_ol_ticker("eqnr.ol") == "EQNR.OL"
    assert to_yahoo_ol_ticker("OSE:SB1NO") == "SB1NO.OL"
    assert to_yahoo_ol_ticker("") is None
    assert to_yahoo_ol_ticker("OSE:") is None
    assert MarketIndices._to_yahoo_ol_ticker("OSE:DNB") == "DNB.OL"


def test_candidates_for_applies_remap_and_dual_list():
    assert candidates_for("ARCHER.OL") == ["ARCH.OL", "ARCHER.OL"]
    assert candidates_for("AKA.OL") == ["AKAST.OL", "AKA.OL"]
    assert candidates_for("GOGL.OL") == ["GOGL", "GOGL.OL"]
    assert candidates_for("EQNR.OL") == ["EQNR.OL"]


def test_resolve_picks_alive_remap_and_drops_dead():
    entries = [
        ("Equinor", "OSE:EQNR"),
        ("Archer", "OSE:ARCHER"),
        ("Akastor", "OSE:AKA"),
        ("Golden Ocean", "OSE:GOGL"),
        ("DeadCo", "OSE:DEAD"),
    ]
    alive = {"EQNR.OL", "ARCH.OL", "AKAST.OL", "GOGL"}

    with patch(
        "stock_analysis.research.oslo_tickers.validate_yahoo_symbols",
        return_value=alive,
    ):
        tickers, remaps = resolve_oslo_yahoo_tickers(
            entries, validate=True, use_search_fallback=False
        )

    assert tickers == ["EQNR.OL", "ARCH.OL", "AKAST.OL", "GOGL"]
    assert remaps["ARCHER.OL"] == "ARCH.OL"
    assert remaps["AKA.OL"] == "AKAST.OL"
    assert remaps["GOGL.OL"] == "GOGL"
    assert "DEAD.OL" not in tickers


def test_resolve_search_fallback_for_unresolved():
    entries = [("Some Oslo Co", "OSE:MISSING")]
    with (
        patch(
            "stock_analysis.research.oslo_tickers.validate_yahoo_symbols",
            side_effect=[set(), {"FOUND.OL"}],
        ),
        patch(
            "stock_analysis.research.oslo_tickers.search_osl_symbol",
            return_value="FOUND.OL",
        ),
    ):
        tickers, remaps = resolve_oslo_yahoo_tickers(
            entries, validate=True, use_search_fallback=True
        )
    assert tickers == ["FOUND.OL"]
    assert remaps["MISSING.OL"] == "FOUND.OL"


@patch("stock_analysis.research.market_indices.requests.get")
def test_get_osebx_tickers_uses_resolver(mock_get):
    response = MagicMock()
    response.content = _FAKE_OSLO_HTML.encode("utf-8")
    response.raise_for_status = MagicMock()
    mock_get.return_value = response

    with patch(
        "stock_analysis.research.oslo_tickers.resolve_oslo_yahoo_tickers",
        return_value=(["EQNR.OL", "ARCH.OL"], {"ARCHER.OL": "ARCH.OL"}),
    ) as mock_resolve:
        tickers = MarketIndices.get_osebx_tickers(validate=True)

    assert tickers == ["EQNR.OL", "ARCH.OL"]
    entries = mock_resolve.call_args.args[0]
    assert ("Equinor", "OSE:EQNR") in entries
    assert ("Archer", "OSE:ARCHER") in entries
    assert mock_resolve.call_args.kwargs["validate"] is True


@patch("stock_analysis.research.market_indices.MarketIndices.get_osebx_tickers")
def test_get_market_tickers_osebx_aliases(mock_osebx):
    mock_osebx.return_value = ["EQNR.OL"]
    assert MarketIndices.get_market_tickers("osebx") == ["EQNR.OL"]
    assert MarketIndices.get_market_tickers("OSLO") == ["EQNR.OL"]
    assert MarketIndices.get_market_tickers("oslo_bors") == ["EQNR.OL"]


def test_get_market_tickers_unknown_raises():
    with pytest.raises(ValueError, match="not supported"):
        MarketIndices.get_market_tickers("not-a-market")
