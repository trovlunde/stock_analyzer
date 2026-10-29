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
class BrokerOrder:
    id: int
    ticker: str
    side: str
    shares: float
    signal_reason: str | None
    client_order_id: str | None
    broker_order_id: str | None
    status: str


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

    def get_running_run_ids(
        self, portfolio_id: int, older_than: datetime | None = None
    ) -> list[int]:
        """Run rows still marked "running" for this portfolio — a non-empty result
        means a previous run died mid-flight rather than finishing cleanly.

        `older_than` filters to runs started before that timestamp, so a run that is
        merely still in progress (e.g. a slow overlapping deploy) isn't mistaken for
        a crash — only pass it when checking for genuinely stale/orphaned runs."""
        with self.engine.connect() as conn:
            conditions = [
                paper_runs.c.portfolio_id == portfolio_id,
                paper_runs.c.status == "running",
            ]
            if older_than is not None:
                conditions.append(paper_runs.c.started_at < older_than)
            rows = conn.execute(select(paper_runs.c.id).where(*conditions)).all()
            return [row[0] for row in rows]

    def abandon_run(self, run_id: int, reason: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                update(paper_runs)
                .where(paper_runs.c.id == run_id)
                .values(
                    finished_at=_utcnow(),
                    status="abandoned",
                    summary_json=json.dumps({"reason": reason}),
                )
            )

    def get_kill_switch_state(self, portfolio_id: int) -> tuple[datetime | None, str | None]:
        with self.engine.connect() as conn:
            row = conn.execute(
                select(
                    paper_portfolios.c.kill_switch_tripped_at,
                    paper_portfolios.c.kill_switch_reason,
                ).where(paper_portfolios.c.id == portfolio_id)
            ).first()
            if row is None:
                return None, None
            return row[0], row[1]

    def trip_kill_switch(self, portfolio_id: int, reason: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                update(paper_portfolios)
                .where(paper_portfolios.c.id == portfolio_id)
                .values(
                    kill_switch_tripped_at=_utcnow(),
                    kill_switch_reason=reason,
                    updated_at=_utcnow(),
                )
            )

    def clear_kill_switch(self, portfolio_id: int) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                update(paper_portfolios)
                .where(paper_portfolios.c.id == portfolio_id)
                .values(
                    kill_switch_tripped_at=None,
                    kill_switch_reason=None,
                    updated_at=_utcnow(),
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
        broker_status: str | None = None,
    ) -> None:
        values = dict(
            status="FILLED",
            fill_price=fill_price,
            commission=commission,
            slippage=slippage,
            filled_at=_utcnow(),
        )
        if broker_status is not None:
            values["broker_status"] = broker_status
        with self.engine.begin() as conn:
            conn.execute(
                update(paper_orders).where(paper_orders.c.id == order_id).values(**values)
            )

    def cancel_pending_order(self, order_id: int) -> bool:
        """Cancel an order if it's still unfilled. Returns False if it doesn't exist or
        has already moved past a cancelable state (e.g. already filled)."""
        with self.engine.begin() as conn:
            result = conn.execute(
                update(paper_orders)
                .where(
                    paper_orders.c.id == order_id,
                    paper_orders.c.status.in_(["PENDING", "SUBMITTING", "SUBMITTED"]),
                )
                .values(status="CANCELED", broker_status="canceled")
            )
            return result.rowcount > 0

    def create_broker_order(
        self,
        portfolio_id: int,
        run_id: int,
        ticker: str,
        side: str,
        shares: float,
        signal_reason: str,
        client_order_id: str,
    ) -> int:
        """Persist the idempotency key *before* calling the broker, so a crash between
        this write and the broker confirming the order can be detected on retry via
        get_order_by_client_id() instead of blindly resubmitting."""
        with self.engine.begin() as conn:
            result = conn.execute(
                insert(paper_orders).values(
                    portfolio_id=portfolio_id,
                    run_id=run_id,
                    ticker=ticker,
                    side=side,
                    shares=shares,
                    status="SUBMITTING",
                    signal_reason=signal_reason,
                    client_order_id=client_order_id,
                    created_at=_utcnow(),
                )
            )
            return result.inserted_primary_key[0]

    def mark_order_submitted(
        self, order_id: int, broker_order_id: str, broker_status: str
    ) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                update(paper_orders)
                .where(paper_orders.c.id == order_id)
                .values(
                    status="SUBMITTED",
                    broker_order_id=broker_order_id,
                    broker_status=broker_status,
                )
            )

    def update_broker_order_status(
        self, order_id: int, status: str, broker_status: str
    ) -> None:
        """For terminal non-fill outcomes (rejected/canceled by the broker)."""
        with self.engine.begin() as conn:
            conn.execute(
                update(paper_orders)
                .where(paper_orders.c.id == order_id)
                .values(status=status, broker_status=broker_status)
            )

    def get_order_by_client_id(self, client_order_id: str) -> BrokerOrder | None:
        with self.engine.connect() as conn:
            row = conn.execute(
                select(paper_orders).where(paper_orders.c.client_order_id == client_order_id)
            ).mappings().first()
            if row is None:
                return None
            return BrokerOrder(
                id=row["id"],
                ticker=row["ticker"],
                side=row["side"],
                shares=row["shares"],
                signal_reason=row["signal_reason"],
                client_order_id=row["client_order_id"],
                broker_order_id=row["broker_order_id"],
                status=row["status"],
            )

    def get_open_broker_orders(self, portfolio_id: int) -> list[BrokerOrder]:
        """Orders submitted to a real broker that haven't reached a terminal state yet."""
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(paper_orders).where(
                    paper_orders.c.portfolio_id == portfolio_id,
                    paper_orders.c.status.in_(["SUBMITTING", "SUBMITTED"]),
                )
            ).mappings().all()
            return [
                BrokerOrder(
                    id=row["id"],
                    ticker=row["ticker"],
                    side=row["side"],
                    shares=row["shares"],
                    signal_reason=row["signal_reason"],
                    client_order_id=row["client_order_id"],
                    broker_order_id=row["broker_order_id"],
                    status=row["status"],
                )
                for row in rows
            ]

    def get_equity_curve(self, portfolio_id: int) -> list[tuple[str, float]]:
        """Return (session_date, portfolio_value) pairs ordered by date from completed non-dry-run runs."""
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(paper_runs.c.summary_json).where(
                    paper_runs.c.portfolio_id == portfolio_id,
                    paper_runs.c.dry_run == False,  # noqa: E712
                    paper_runs.c.finished_at.isnot(None),
                    paper_runs.c.summary_json.isnot(None),
                )
            ).mappings().all()

        points: list[tuple[str, float]] = []
        for row in rows:
            try:
                summary = json.loads(row["summary_json"])
                session_date = summary.get("session_date")
                portfolio_value = summary.get("portfolio_value")
                if session_date and portfolio_value is not None:
                    points.append((str(session_date), float(portfolio_value)))
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

        return sorted(points, key=lambda x: x[0])
