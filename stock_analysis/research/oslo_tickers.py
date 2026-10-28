"""Resolve Oslo / OSEBX Wikipedia symbols to working Yahoo Finance tickers."""

from __future__ import annotations

import contextlib
import io
import logging
from typing import Iterable

import yfinance as yf

# Wikipedia / OSE mnemonic → Yahoo symbol when they diverge.
YAHOO_SYMBOL_REMAP: dict[str, str] = {
    "AKA.OL": "AKAST.OL",  # Akastor
    "ARCHER.OL": "ARCH.OL",
    "GOGL.OL": "GOGL",  # Golden Ocean dual-listed
    "FLNG.OL": "FLNG",  # Flex LNG dual-listed
}

_OSL_EXCHANGES = frozenset({"OSL", "OSLO"})


def to_yahoo_ol_ticker(raw: str) -> str | None:
    """Normalize Wikipedia OSE ticker text to yfinance ``SYMBOL.OL``."""
    if not raw:
        return None
    symbol = raw.strip().replace("\xa0", " ")
    if symbol.startswith("OSE:"):
        symbol = symbol[4:]
    symbol = symbol.split()[0] if symbol else ""
    symbol = symbol.rstrip("*").strip()
    if not symbol:
        return None
    if symbol.upper().endswith(".OL"):
        base = symbol[:-3]
    else:
        base = symbol
    if not base.replace("-", "").isalnum():
        return None
    return f"{base.upper()}.OL"


def candidates_for(symbol: str) -> list[str]:
    """Ordered Yahoo candidates for one normalized ``*.OL`` symbol.

    Bare non-``.OL`` tickers are only tried when listed in ``YAHOO_SYMBOL_REMAP``
    (known dual listings). Blindly trying the US ticker of the same mnemonic
    risks matching the wrong company (e.g. AMSC.OL → AMSC).
    """
    if symbol in YAHOO_SYMBOL_REMAP:
        remapped = YAHOO_SYMBOL_REMAP[symbol]
        ordered: list[str] = []
        for candidate in (remapped, symbol):
            if candidate and candidate not in ordered:
                ordered.append(candidate)
        return ordered
    return [symbol]


def validate_yahoo_symbols(
    symbols: Iterable[str],
    *,
    period: str = "5d",
    chunk_size: int = 40,
) -> set[str]:
    """Return symbols that have recent Yahoo price data. Suppresses download noise."""
    unique = [s for s in dict.fromkeys(symbols) if s]
    if not unique:
        return set()

    ok: set[str] = set()
    log = logging.getLogger("yfinance")
    prev_level = log.level
    log.setLevel(logging.CRITICAL)
    try:
        for i in range(0, len(unique), chunk_size):
            chunk = unique[i : i + chunk_size]
            with contextlib.redirect_stderr(io.StringIO()):
                df = yf.download(
                    chunk,
                    period=period,
                    progress=False,
                    auto_adjust=True,
                    threads=True,
                    group_by="ticker",
                )
            if df is None or df.empty:
                continue
            if len(chunk) == 1:
                sym = chunk[0]
                if not df.dropna(how="all").empty:
                    ok.add(sym)
                continue
            tickers_level = df.columns.get_level_values(0)
            for sym in chunk:
                if sym not in tickers_level:
                    continue
                sub = df[sym].dropna(how="all")
                if not sub.empty:
                    ok.add(sym)
    finally:
        log.setLevel(prev_level)
    return ok


def search_osl_symbol(company_name: str) -> str | None:
    """Best-effort Yahoo search for an Oslo-listed equity by company name."""
    if not company_name or company_name == "N/A":
        return None
    query = f"{company_name} Oslo"
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            quotes = yf.Search(query, max_results=8).quotes or []
    except Exception:
        return None

    for quote in quotes:
        if quote.get("quoteType") not in (None, "EQUITY"):
            continue
        symbol = quote.get("symbol") or ""
        exchange = (quote.get("exchange") or "").upper()
        if exchange in _OSL_EXCHANGES or symbol.endswith(".OL"):
            return symbol
    return None


def resolve_oslo_yahoo_tickers(
    entries: list[tuple[str, str]],
    *,
    validate: bool = True,
    use_search_fallback: bool = True,
    max_search: int = 40,
) -> tuple[list[str], dict[str, str]]:
    """Map ``(company_name, wikipedia_ose_raw)`` to working Yahoo tickers.

    Returns ``(tickers, remaps)`` where ``remaps`` maps original ``*.OL`` → chosen.
    """
    planned: list[tuple[str, str, list[str]]] = []
    for company, raw in entries:
        original = to_yahoo_ol_ticker(raw)
        if not original:
            continue
        planned.append((company, original, candidates_for(original)))

    if not planned:
        return [], {}

    if not validate:
        chosen: list[str] = []
        remaps: dict[str, str] = {}
        seen: set[str] = set()
        for _company, original, cands in planned:
            pick = cands[0]
            if pick not in seen:
                seen.add(pick)
                chosen.append(pick)
            if pick != original:
                remaps[original] = pick
        return chosen, remaps

    all_candidates = [c for *_rest, cands in planned for c in cands]
    alive = validate_yahoo_symbols(all_candidates)

    resolved: list[str] = []
    remaps: dict[str, str] = {}
    seen: set[str] = set()
    unresolved: list[tuple[str, str]] = []

    for company, original, cands in planned:
        pick = next((c for c in cands if c in alive), None)
        if pick is None:
            unresolved.append((company, original))
            continue
        if pick not in seen:
            seen.add(pick)
            resolved.append(pick)
        if pick != original:
            remaps[original] = pick

    searches = 0
    if use_search_fallback:
        for company, original in unresolved:
            if searches >= max_search:
                break
            searches += 1
            found = search_osl_symbol(company)
            if not found:
                continue
            if found not in alive:
                if found not in validate_yahoo_symbols([found]):
                    continue
                alive.add(found)
            if found not in seen:
                seen.add(found)
                resolved.append(found)
            if found != original:
                remaps[original] = found

    return resolved, remaps
