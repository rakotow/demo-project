"""
Historical data fetcher for Polymarket BTC 5-minute markets.

Two data sources:
  1. Gamma API  — closed market metadata + resolved outcomes
  2. CLOB API   — per-token price history (1-min candles) and individual trades

Saves everything to data/raw/ as CSV for reproducible backtests.
"""

import csv
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Gamma API helpers
# ---------------------------------------------------------------------------

def fetch_closed_btc_5min_markets(
    max_pages: int = 20,
    page_size: int = 100,
) -> list[dict]:
    """
    Return all closed BTC 5-minute markets from the Gamma API.

    Each dict includes at minimum:
      conditionId, question, endDate, outcomePrices, tokens
    """
    session = requests.Session()
    all_markets: list[dict] = []
    offset = 0

    for page in range(max_pages):
        params = {
            "closed": "true",
            "active": "false",
            "tag": "crypto",
            "limit": page_size,
            "offset": offset,
        }
        try:
            resp = session.get(f"{GAMMA_API}/markets", params=params, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.error("Gamma API error on page %d: %s", page, exc)
            break

        batch = resp.json()
        if not batch:
            break

        for m in batch:
            question: str = (m.get("question") or "").lower()
            if "btc" in question and (
                "5 min" in question or "5min" in question or "5-min" in question
            ):
                all_markets.append(m)

        logger.info(
            "Page %d: fetched %d markets, %d BTC-5min so far",
            page,
            len(batch),
            len(all_markets),
        )

        if len(batch) < page_size:
            break
        offset += page_size
        time.sleep(0.3)  # be polite

    logger.info("Total closed BTC 5-min markets found: %d", len(all_markets))
    return all_markets


def parse_outcome(market: dict) -> Optional[str]:
    """
    Return 'YES' or 'NO' based on resolved outcomePrices, or None if unresolved.

    Polymarket sets the winning token to price '1' and the losing token to '0'.
    """
    prices = market.get("outcomePrices")
    if not prices:
        return None
    try:
        prices_float = [float(p) for p in prices]
    except (TypeError, ValueError):
        return None
    outcomes = market.get("outcomes") or ["YES", "NO"]
    # Find the outcome whose price resolved to 1
    for outcome, price in zip(outcomes, prices_float):
        if price == 1.0:
            return str(outcome).upper()
    return None


# ---------------------------------------------------------------------------
# CLOB API helpers
# ---------------------------------------------------------------------------

def fetch_price_history(
    token_id: str,
    start_ts: int,
    end_ts: int,
    fidelity: int = 1,
) -> list[dict]:
    """
    Fetch OHLC price history for a single token from the CLOB API.

    fidelity: candle size in minutes (1 = 1-min candles)

    Returns list of dicts: {t, o, h, l, c} where t is Unix timestamp (seconds).
    """
    params = {
        "market": token_id,
        "startTs": start_ts,
        "endTs": end_ts,
        "fidelity": fidelity,
    }
    try:
        resp = requests.get(
            f"{CLOB_API}/prices-history", params=params, timeout=15
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("history", [])
    except requests.RequestException as exc:
        logger.warning("Price history fetch failed for %s: %s", token_id, exc)
        return []


def fetch_trades(token_id: str, limit: int = 500) -> list[dict]:
    """
    Fetch individual trade ticks for a token (most recent first).

    Returns list of dicts: {price, size, side, timestamp}.
    """
    params = {"market": token_id, "limit": limit}
    try:
        resp = requests.get(f"{CLOB_API}/trades", params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else data.get("data", [])
    except requests.RequestException as exc:
        logger.warning("Trades fetch failed for %s: %s", token_id, exc)
        return []


# ---------------------------------------------------------------------------
# Main data collection pipeline
# ---------------------------------------------------------------------------

def _parse_end_ts(market: dict) -> Optional[int]:
    """Return market end time as Unix timestamp (int), or None."""
    raw = market.get("endDate") or market.get("end_date_iso") or ""
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return int(dt.timestamp())
    except ValueError:
        return None


def collect_market_data(
    markets: list[dict],
    lookback_minutes: int = 10,
) -> list[dict]:
    """
    For each market, fetch price history for the last `lookback_minutes` minutes
    and the final trades.  Returns an enriched list of market dicts.
    """
    enriched = []
    for i, market in enumerate(markets):
        condition_id = market.get("conditionId") or market.get("condition_id", "")
        question = market.get("question", "")
        end_ts = _parse_end_ts(market)

        if not end_ts:
            logger.debug("Skipping market with no end_ts: %s", condition_id)
            continue

        tokens: list[dict] = market.get("tokens") or []
        if not tokens:
            # Some markets expose token IDs differently
            clob_ids = market.get("clobTokenIds") or []
            outcomes = market.get("outcomes") or ["YES", "NO"]
            tokens = [
                {"token_id": tid, "outcome": out}
                for tid, out in zip(clob_ids, outcomes)
            ]

        if not tokens:
            logger.debug("No tokens for market %s", condition_id)
            continue

        resolved_outcome = parse_outcome(market)
        start_ts = end_ts - (lookback_minutes * 60)

        token_data = []
        for token in tokens:
            token_id = token.get("token_id") or token.get("tokenId") or ""
            outcome = token.get("outcome", "")
            if not token_id:
                continue

            candles = fetch_price_history(token_id, start_ts, end_ts, fidelity=1)
            trades = fetch_trades(token_id, limit=200)

            # Filter trades to the lookback window
            trades_in_window = [
                t for t in trades
                if start_ts <= int(t.get("timestamp") or t.get("created_at") or 0) <= end_ts
            ]

            token_data.append({
                "token_id": token_id,
                "outcome": outcome,
                "candles": candles,
                "trades": trades_in_window,
                "resolved": resolved_outcome == outcome.upper() if resolved_outcome else None,
            })

            time.sleep(0.15)

        enriched.append({
            **market,
            "end_ts": end_ts,
            "resolved_outcome": resolved_outcome,
            "token_data": token_data,
        })

        if (i + 1) % 10 == 0:
            logger.info("Collected %d / %d markets", i + 1, len(markets))

    return enriched


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_markets_csv(markets: list[dict], path: Optional[Path] = None) -> Path:
    """Save high-level market metadata to CSV."""
    if path is None:
        path = RAW_DIR / "btc_5min_markets.csv"

    fieldnames = [
        "condition_id", "question", "end_date", "end_ts",
        "resolved_outcome", "num_tokens",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in markets:
            writer.writerow({
                "condition_id": m.get("conditionId") or m.get("condition_id", ""),
                "question": m.get("question", ""),
                "end_date": m.get("endDate") or m.get("end_date_iso", ""),
                "end_ts": m.get("end_ts", ""),
                "resolved_outcome": m.get("resolved_outcome", ""),
                "num_tokens": len(m.get("token_data", [])),
            })
    logger.info("Saved market metadata to %s", path)
    return path


def save_candles_csv(markets: list[dict], path: Optional[Path] = None) -> Path:
    """Save all candle data (one row per candle per token) to CSV."""
    if path is None:
        path = RAW_DIR / "btc_5min_candles.csv"

    fieldnames = [
        "condition_id", "token_id", "outcome", "resolved",
        "end_ts", "candle_t", "open", "high", "low", "close",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in markets:
            condition_id = m.get("conditionId") or m.get("condition_id", "")
            for td in m.get("token_data", []):
                for candle in td.get("candles", []):
                    writer.writerow({
                        "condition_id": condition_id,
                        "token_id": td["token_id"],
                        "outcome": td["outcome"],
                        "resolved": td.get("resolved"),
                        "end_ts": m.get("end_ts", ""),
                        "candle_t": candle.get("t", ""),
                        "open": candle.get("o", ""),
                        "high": candle.get("h", ""),
                        "low": candle.get("l", ""),
                        "close": candle.get("c", ""),
                    })
    logger.info("Saved candle data to %s", path)
    return path


def save_trades_csv(markets: list[dict], path: Optional[Path] = None) -> Path:
    """Save all individual trade ticks to CSV."""
    if path is None:
        path = RAW_DIR / "btc_5min_trades.csv"

    fieldnames = [
        "condition_id", "token_id", "outcome", "resolved",
        "end_ts", "trade_ts", "price", "size", "side",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in markets:
            condition_id = m.get("conditionId") or m.get("condition_id", "")
            for td in m.get("token_data", []):
                for trade in td.get("trades", []):
                    ts = trade.get("timestamp") or trade.get("created_at") or ""
                    writer.writerow({
                        "condition_id": condition_id,
                        "token_id": td["token_id"],
                        "outcome": td["outcome"],
                        "resolved": td.get("resolved"),
                        "end_ts": m.get("end_ts", ""),
                        "trade_ts": ts,
                        "price": trade.get("price", ""),
                        "size": trade.get("size", ""),
                        "side": trade.get("side", ""),
                    })
    logger.info("Saved trade tick data to %s", path)
    return path


def save_raw_json(markets: list[dict], path: Optional[Path] = None) -> Path:
    """Save full enriched market data as JSON (for debugging / re-runs)."""
    if path is None:
        path = RAW_DIR / "btc_5min_raw.json"
    with open(path, "w") as f:
        json.dump(markets, f, indent=2, default=str)
    logger.info("Saved raw JSON to %s", path)
    return path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def run_fetch(lookback_minutes: int = 10, max_pages: int = 20) -> list[dict]:
    """Fetch and save all historical data. Returns enriched market list."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("Fetching closed BTC 5-min markets from Gamma API…")
    markets = fetch_closed_btc_5min_markets(max_pages=max_pages)

    if not markets:
        logger.warning("No markets found. Check API availability.")
        return []

    logger.info("Collecting price history and trades (lookback=%dm)…", lookback_minutes)
    enriched = collect_market_data(markets, lookback_minutes=lookback_minutes)

    save_markets_csv(enriched)
    save_candles_csv(enriched)
    save_trades_csv(enriched)
    save_raw_json(enriched)

    return enriched


if __name__ == "__main__":
    run_fetch()
