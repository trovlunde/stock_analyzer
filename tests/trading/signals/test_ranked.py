import pandas as pd

from stock_analysis.trading.signals.ranked import RankedUniverseStrategy

SESSION = pd.Timestamp("2024-03-01")


class _FixedScores:
    def __init__(self, scores):
        self.scores = scores
        self.calls = 0

    def scores_on(self, session):
        self.calls += 1
        return self.scores


def _frame():
    return pd.DataFrame({"Close": [1.0]}, index=[SESSION])


def _action(strategy, ticker, params=None):
    return strategy.evaluate(_frame(), ticker, params or {}).action


def test_top_ranked_are_bought_and_the_rest_sold():
    provider = _FixedScores(pd.Series({"A": 3.0, "B": 2.0, "C": 1.0, "D": 0.0}))
    strategy = RankedUniverseStrategy(provider, top_k=2)

    assert [_action(strategy, t) for t in "ABCD"] == ["BUY", "BUY", "SELL", "SELL"]


def test_holding_between_top_k_and_exit_rank_is_kept():
    provider = _FixedScores(pd.Series({"A": 3.0, "B": 2.0, "C": 1.0, "D": 0.0}))
    strategy = RankedUniverseStrategy(provider, top_k=1, exit_rank=3)

    assert [_action(strategy, t) for t in "ABCD"] == ["BUY", "HOLD", "HOLD", "SELL"]


def test_params_override_constructor():
    provider = _FixedScores(pd.Series({"A": 3.0, "B": 2.0, "C": 1.0}))
    strategy = RankedUniverseStrategy(provider, top_k=1)

    assert _action(strategy, "B", {"rank_top_k": 2}) == "BUY"


def test_unscored_ticker_is_sold_once_a_ranking_exists():
    provider = _FixedScores(pd.Series({"A": 1.0, "B": float("nan")}))
    strategy = RankedUniverseStrategy(provider, top_k=1)

    assert _action(strategy, "B") == "SELL"
    assert _action(strategy, "MISSING") == "SELL"


def test_no_ranking_yet_is_a_hold():
    assert _action(RankedUniverseStrategy(_FixedScores(None), top_k=1), "A") == "HOLD"
    empty = _FixedScores(pd.Series({"A": float("nan")}))
    assert _action(RankedUniverseStrategy(empty, top_k=1), "A") == "HOLD"


def test_scores_are_fetched_once_per_session():
    provider = _FixedScores(pd.Series({"A": 3.0, "B": 2.0}))
    strategy = RankedUniverseStrategy(provider, top_k=1)

    for ticker in "AB":
        _action(strategy, ticker)

    assert provider.calls == 1


def test_top_ranked_below_min_score_is_held_not_bought():
    provider = _FixedScores(pd.Series({"A": 0.3, "B": -0.1, "C": -0.2, "D": -0.5}))
    strategy = RankedUniverseStrategy(provider, top_k=2, exit_rank=3, min_score=0.0)

    # B is in the top two but under the minimum; exits are unchanged (D is sold)
    assert [_action(strategy, t) for t in "ABCD"] == ["BUY", "HOLD", "HOLD", "SELL"]
    result = strategy.evaluate(_frame(), "B", {})
    assert "below the minimum" in result.reason
    assert result.metadata["score"] == -0.1


def test_score_equal_to_min_score_is_bought():
    provider = _FixedScores(pd.Series({"A": 0.0, "B": -1.0}))

    assert _action(RankedUniverseStrategy(provider, top_k=1, min_score=0.0), "A") == "BUY"


def test_min_score_param_overrides_constructor_and_default_is_off():
    provider = _FixedScores(pd.Series({"A": 0.3, "B": 0.1, "C": 0.0}))
    strategy = RankedUniverseStrategy(provider, top_k=2)

    assert _action(strategy, "B") == "BUY"
    assert _action(strategy, "B", {"rank_min_score": 0.2}) == "HOLD"
    assert _action(strategy, "A", {"rank_min_score": 0.2}) == "BUY"
