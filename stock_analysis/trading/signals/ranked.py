"""Cross-sectional strategies: hold the best-ranked names of a universe.

A `SignalStrategy` is asked about one ticker at a time. Strategies that compare
tickers with each other (a pooled ML model, a factor screen) therefore keep the
comparison in a `ScoreProvider` and let `RankedUniverseStrategy` turn the rank of
each ticker into BUY / HOLD / SELL.
"""

from typing import Protocol

import pandas as pd

from .base import SignalResult


class ScoreProvider(Protocol):
    def scores_on(self, session: pd.Timestamp) -> pd.Series | None:
        """Score per ticker (higher is better) using only what was known at the
        close of `session`. Tickers that cannot be ranked are left out or NaN.
        None while there is no ranking yet (for example before the first model
        has been trained). How often the scores change — daily, monthly — is up
        to the provider; between rebalances it returns the same scores."""
        ...


class RankedUniverseStrategy:
    """BUY the `top_k` best-scored tickers, SELL once a holding falls below `exit_rank`.

    The gap between `top_k` and `exit_rank` keeps a holding that slips from, say,
    rank 10 to rank 12 instead of trading it back and forth. A ticker with no
    score on a session that has a ranking (it left the universe, or its data is
    missing) is a SELL.

    With `min_score`, a ticker in the top `top_k` whose score is below it is a
    HOLD instead of a BUY: being the best of a bad lot is not a reason to enter.
    Exits are unchanged, so a holding is still kept until it falls below
    `exit_rank`.

    `rank_top_k`, `rank_exit` and `rank_min_score` in the strategy params
    override the constructor.
    """

    def __init__(
        self,
        provider: ScoreProvider,
        *,
        top_k: int = 10,
        exit_rank: int | None = None,
        min_score: float | None = None,
    ):
        self.provider = provider
        self.top_k = top_k
        self.exit_rank = exit_rank
        self.min_score = min_score
        self._cached: tuple[pd.Timestamp, pd.Series | None, pd.Series | None] | None = None

    def _ranks(self, session: pd.Timestamp) -> pd.Series | None:
        if self._cached is None or self._cached[0] != session:
            scores = self.provider.scores_on(session)
            ranks = None
            if scores is not None:
                scores = scores.dropna()
                if not scores.empty:
                    # 1 = best; ties broken by position so the result is deterministic
                    ranks = scores.rank(ascending=False, method="first").astype(int)
            self._cached = (session, ranks, scores)
        return self._cached[1]

    def evaluate(self, df: pd.DataFrame, ticker: str, params: dict) -> SignalResult:
        top_k = int(params.get("rank_top_k", self.top_k))
        exit_default = self.exit_rank if self.exit_rank is not None else top_k
        exit_rank = max(top_k, int(params.get("rank_exit", exit_default)))

        ranks = self._ranks(df.index[-1])
        if ranks is None:
            return SignalResult(action="HOLD", reason="no ranking available yet", metadata={})
        if ticker not in ranks.index:
            return SignalResult(
                action="SELL", reason="not in the ranked universe", metadata={"universe": len(ranks)}
            )

        rank = int(ranks[ticker])
        metadata = {"rank": rank, "universe": len(ranks), "top_k": top_k, "exit_rank": exit_rank}
        min_score = params.get("rank_min_score", self.min_score)
        if rank <= top_k and min_score is not None:
            score = float(self._cached[2][ticker])
            metadata["score"] = score
            if score < float(min_score):
                return SignalResult(
                    action="HOLD",
                    reason=f"rank {rank} of {len(ranks)} but score {score:.4f} is below "
                    f"the minimum {float(min_score):.4f}",
                    metadata=metadata,
                )
        if rank <= top_k:
            return SignalResult(
                action="BUY", reason=f"rank {rank} of {len(ranks)} (top {top_k})", metadata=metadata
            )
        if rank > exit_rank:
            return SignalResult(
                action="SELL",
                reason=f"rank {rank} of {len(ranks)} (below exit rank {exit_rank})",
                metadata=metadata,
            )
        return SignalResult(
            action="HOLD",
            reason=f"rank {rank} of {len(ranks)} (between top {top_k} and exit {exit_rank})",
            metadata=metadata,
        )
