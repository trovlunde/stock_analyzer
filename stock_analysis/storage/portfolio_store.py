import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

from sqlalchemy import delete, insert, select, update
from sqlalchemy.engine import Engine

from .schema import (
    paper_orders,
    paper_portfolios,
    paper_positions,
    paper_runs,
    paper_signals,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Position:
    ticker: str
    shares: float
    avg_cost: float
    opened_at: datetime


@dataclass
class PendingOrder:
    id: int
    ticker: str
    side: str
    shares: float
    signal_reason: str | None
    fill_date: date | None


@dataclass
class PortfolioState:
    id: int
    name: str
    cash: float
    positions: list[Position]


class PortfolioStore:
    def __init__(self, engine: Engine):
        self.engine = engine

    def get_or_create_portfolio(self, name: str, initial_cash: float) -> PortfolioState:
        existing = self.get_portfolio_by_name(name)
        if existing is not None:
            return existing

        now = _utcnow()
        with self.engine.begin() as conn:
            result = conn.execute(
                insert(paper_portfolios).values(
                    name=name,
                    cash=initial_cash,
                    created_at=now,
                    updated_at=now,
                )
            )
            portfolio_id = result.inserted_primary_key[0]

        return PortfolioState(id=portfolio_id, name=name, cash=initial_cash, positions=[])

    def get_portfolio_by_name(self, name: str) -> PortfolioState | None:
        with self.engine.connect() as conn:
            row = conn.execute(
                select(paper_portfolios).where(paper_portfolios.c.name == name)
            ).mappings().first()
            if row is None:
                return None
            positions = self._load_positions(conn, row["id"])
            return PortfolioState(
                id=row["id"],
                name=row["name"],
                cash=row["cash"],
                positions=positions,
            )

    def _load_positions(self, conn, portfolio_id: int) -> list[Position]:
        rows = conn.execute(
            select(paper_positions).where(paper_positions.c.portfolio_id == portfolio_id)
        ).mappings().all()
        return [
            Position(
                ticker=row["ticker"],
                shares=row["shares"],
                avg_cost=row["avg_cost"],
                opened_at=row["opened_at"],
            )
            for row in rows
        ]

    def update_cash(self, portfolio_id: int, cash: float) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                update(paper_portfolios)
                .where(paper_portfolios.c.id == portfolio_id)
                .values(cash=cash, updated_at=_utcnow())
            )

    def upsert_position(
        self,
        portfolio_id: int,
        ticker: str,
        shares: float,
        avg_cost: float,
        opened_at: datetime | None = None,
    ) -> None:
        opened = opened_at or _utcnow()
        with self.engine.begin() as conn:
            existing = conn.execute(
                select(paper_positions).where(
                    paper_positions.c.portfolio_id == portfolio_id,
                    paper_positions.c.ticker == ticker,
                )
            ).mappings().first()
            if existing is None:
                conn.execute(
                    insert(paper_positions).values(
                        portfolio_id=portfolio_id,
                        ticker=ticker,
                        shares=shares,
                        avg_cost=avg_cost,
                        opened_at=opened,
                    )
                )
            else:
                conn.execute(
                    update(paper_positions)
                    .where(paper_positions.c.id == existing["id"])
                    .values(shares=shares, avg_cost=avg_cost)
                )

    def remove_position(self, portfolio_id: int, ticker: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                delete(paper_positions).where(
                    paper_positions.c.portfolio_id == portfolio_id,
                    paper_positions.c.ticker == ticker,
                )
            )

    def start_run(self, portfolio_id: int, dry_run: bool) -> int:
        with self.engine.begin() as conn:
            result = conn.execute(
                insert(paper_runs).values(
                    portfolio_id=portfolio_id,
                    started_at=_utcnow(),
                    status="running",
                    dry_run=dry_run,
                )
            )
            return result.inserted_primary_key[0]

    def finish_run(self, run_id: int, status: str, summary: dict[str, Any]) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                update(paper_runs)
                .where(paper_runs.c.id == run_id)
                .values(
                    finished_at=_utcnow(),
                    status=status,
                    summary_json=json.dumps(summary),
                )
            )

    def record_signal(
        self,
        run_id: int,
        ticker: str,
        action: str,
        reason: str,
        metadata: dict,
    ) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                insert(paper_signals).values(
                    run_id=run_id,
                    ticker=ticker,
                    action=action,
                    reason=reason,
                    metadata_json=json.dumps(metadata),
                )
            )

    def create_pending_order(
        self,
        portfolio_id: int,
        run_id: int,
        ticker: str,
        side: str,
        shares: float,
        signal_reason: str,
        fill_date: date,
    ) -> int:
        with self.engine.begin() as conn:
            result = conn.execute(
                insert(paper_orders).values(
                    portfolio_id=portfolio_id,
                    run_id=run_id,
                    ticker=ticker,
                    side=side,
                    shares=shares,
                    status="PENDING",
                    signal_reason=signal_reason,
                    created_at=_utcnow(),
                    fill_date=fill_date,
                )
            )
            return result.inserted_primary_key[0]

    def get_pending_orders(self, portfolio_id: int) -> list[PendingOrder]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(paper_orders).where(
                    paper_orders.c.portfolio_id == portfolio_id,
                    paper_orders.c.status == "PENDING",
                )
            ).mappings().all()
            return [
                PendingOrder(
                    id=row["id"],
                    ticker=row["ticker"],
                    side=row["side"],
                    shares=row["shares"],
                    signal_reason=row["signal_reason"],
                    fill_date=row["fill_date"],
                )
                for row in rows
            ]

    def fill_order(
        self,
        order_id: int,
        fill_price: float,
        commission: float,
        slippage: float,
    ) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                update(paper_orders)
                .where(paper_orders.c.id == order_id)
                .values(
                    status="FILLED",
                    fill_price=fill_price,
                    commission=commission,
                    slippage=slippage,
                    filled_at=_utcnow(),
                )
            )
