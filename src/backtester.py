"""
Backtest the Polymarket BTC 5-min trading strategy.

Strategy recap
--------------
  For each closed BTC 5-min market:
    - Look at prices in the last ENTRY_WINDOW_SECONDS before close.
    - If any outcome token's ask was >= PRICE_THRESHOLD, simulate buying it.
    - The trade resolves to 1.00 USDC per token if that outcome won, else 0.

Two data modes
--------------
  "trades"  — use individual trade ticks (most accurate for 40-second window)
  "candles" — fall back to 1-min OHLC candles (the last candle covers ~60s)

Usage
-----
  # Fetch data first (only needed once):
  python -m src.data_fetcher

  # Run backtest on fetched data:
  python -m src.backtester

  # Override parameters:
  python -m src.backtester --threshold 0.85 --window 40 --size 10 --mode candles
"""

import argparse
import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
REPORTS_DIR = Path(__file__).parent.parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    condition_id: str
    question: str
    end_ts: int
    token_id: str
    outcome: str           # 'YES' or 'NO'
    entry_price: float     # price paid per token
    size_usdc: float       # USDC spent
    tokens_bought: float   # size_usdc / entry_price
    resolved: bool         # did this outcome win?
    pnl_usdc: float        # net profit/loss in USDC
    data_source: str       # 'trade_tick' | 'candle_close' | 'candle_high'
    seconds_before_close: float  # how far before close the signal fired


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    markets_scanned: int = 0
    markets_with_signal: int = 0
    markets_no_data: int = 0

    # Computed in summarise()
    total_invested: float = 0.0
    total_returned: float = 0.0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    roi_pct: float = 0.0
    avg_entry_price: float = 0.0


# ---------------------------------------------------------------------------
# Loaders from CSV / JSON
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _load_candles_csv(path: Path) -> dict[str, list[dict]]:
    """
    Returns {condition_id + '_' + token_id: [candle, ...]} sorted by time asc.
    """
    rows: dict[str, list[dict]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = row["condition_id"] + "|" + row["token_id"]
            rows.setdefault(key, []).append(row)
    # sort ascending by candle timestamp
    for key in rows:
        rows[key].sort(key=lambda r: float(r["candle_t"] or 0))
    return rows


def _load_trades_csv(path: Path) -> dict[str, list[dict]]:
    """
    Returns {condition_id + '|' + token_id: [trade, ...]} sorted by time asc.
    """
    rows: dict[str, list[dict]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = row["condition_id"] + "|" + row["token_id"]
            rows.setdefault(key, []).append(row)
    for key in rows:
        rows[key].sort(key=lambda r: float(r["trade_ts"] or 0))
    return rows


# ---------------------------------------------------------------------------
# Signal detection
# ---------------------------------------------------------------------------

def _price_in_window_from_trades(
    trades: list[dict],
    end_ts: int,
    window_secs: int,
    threshold: float,
) -> Optional[tuple[float, float]]:
    """
    Scan trade ticks inside (end_ts - window_secs, end_ts].
    Return (price, secs_before_close) for the first ask >= threshold, else None.

    Polymarket trade `side` can be 'BUY' (taker bought = ask side) or 'SELL'.
    We treat any trade price >= threshold as a valid signal (the market was
    clearing at that price when we would have looked).
    """
    window_start = end_ts - window_secs
    for t in trades:
        ts = float(t.get("trade_ts") or 0)
        if ts < window_start or ts > end_ts:
            continue
        price = float(t.get("price") or 0)
        if price >= threshold:
            return price, end_ts - ts
    return None


def _price_in_window_from_candles(
    candles: list[dict],
    end_ts: int,
    window_secs: int,
    threshold: float,
) -> Optional[tuple[float, float, str]]:
    """
    Use 1-min candle data as a proxy for the entry window.

    Returns (price, secs_before_close, source_label) or None.
    source_label is 'candle_close' or 'candle_high'.
    """
    window_start = end_ts - window_secs
    # Look at candles that START inside or overlap the entry window.
    # A 1-min candle at time t covers [t, t+60).
    relevant = [
        c for c in candles
        if float(c.get("candle_t") or 0) + 60 >= window_start
        and float(c.get("candle_t") or 0) <= end_ts
    ]
    if not relevant:
        return None
    # Use the last candle's close as the best estimate for "current price".
    last = relevant[-1]
    close = float(last.get("close") or 0)
    high = float(last.get("high") or 0)
    candle_t = float(last.get("candle_t") or 0)
    secs_before = end_ts - candle_t

    if close >= threshold:
        return close, secs_before, "candle_close"
    if high >= threshold:
        return high, secs_before, "candle_high"
    return None


# ---------------------------------------------------------------------------
# Core backtest engine
# ---------------------------------------------------------------------------

def run_backtest(
    markets: list[dict],
    candle_index: dict[str, list[dict]],
    trade_index: dict[str, list[dict]],
    threshold: float = 0.80,
    window_secs: int = 40,
    trade_size_usdc: float = 5.0,
    mode: str = "trades",  # 'trades' | 'candles'
) -> BacktestResult:
    result = BacktestResult()
    result.markets_scanned = len(markets)

    for market in markets:
        condition_id = market.get("conditionId") or market.get("condition_id", "")
        question = market.get("question", "")
        end_ts = int(market.get("end_ts") or 0)
        resolved_outcome = (market.get("resolved_outcome") or "").upper()

        if not end_ts:
            result.markets_no_data += 1
            continue

        token_data: list[dict] = market.get("token_data") or []
        if not token_data:
            result.markets_no_data += 1
            continue

        fired_this_market = False

        for td in token_data:
            token_id = td.get("token_id", "")
            outcome = (td.get("outcome") or "").upper()
            resolved: Optional[bool] = td.get("resolved")

            # If we don't know whether this token resolved, skip it.
            if resolved is None and not resolved_outcome:
                continue

            key = condition_id + "|" + token_id
            signal = None

            if mode == "trades":
                ticks = trade_index.get(key, [])
                hit = _price_in_window_from_trades(ticks, end_ts, window_secs, threshold)
                if hit:
                    price, secs_before = hit
                    signal = (price, secs_before, "trade_tick")
            else:
                candles = candle_index.get(key, [])
                hit = _price_in_window_from_candles(candles, end_ts, window_secs, threshold)
                if hit:
                    price, secs_before, src = hit
                    signal = (price, secs_before, src)

            if signal is None:
                continue

            if fired_this_market:
                # Only trade the first signal per market (same as live bot)
                break

            entry_price, secs_before_close, data_source = signal
            tokens_bought = trade_size_usdc / entry_price

            # Did this outcome win?
            won = (resolved is True) or (
                resolved_outcome != "" and outcome == resolved_outcome
            )
            returned = tokens_bought * 1.0 if won else 0.0
            pnl = returned - trade_size_usdc

            trade = Trade(
                condition_id=condition_id,
                question=question,
                end_ts=end_ts,
                token_id=token_id,
                outcome=outcome,
                entry_price=entry_price,
                size_usdc=trade_size_usdc,
                tokens_bought=tokens_bought,
                resolved=won,
                pnl_usdc=pnl,
                data_source=data_source,
                seconds_before_close=secs_before_close,
            )
            result.trades.append(trade)
            fired_this_market = True

        if fired_this_market:
            result.markets_with_signal += 1

    _summarise(result)
    return result


def _summarise(result: BacktestResult) -> None:
    if not result.trades:
        return
    result.total_invested = sum(t.size_usdc for t in result.trades)
    result.total_returned = sum(
        t.tokens_bought * 1.0 for t in result.trades if t.resolved
    )
    result.total_pnl = result.total_returned - result.total_invested
    wins = [t for t in result.trades if t.resolved]
    result.win_rate = len(wins) / len(result.trades) * 100
    result.roi_pct = (result.total_pnl / result.total_invested * 100
                      if result.total_invested else 0.0)
    result.avg_entry_price = (
        sum(t.entry_price for t in result.trades) / len(result.trades)
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(result: BacktestResult, threshold: float, window_secs: int) -> None:
    sep = "=" * 62
    print(f"\n{sep}")
    print("  POLYMARKET BTC 5-MIN BOT — BACKTEST REPORT")
    print(sep)
    print(f"  Strategy   : buy when ask >= {threshold:.2f} in last {window_secs}s")
    print(f"  Markets scanned         : {result.markets_scanned:>6}")
    print(f"  Markets with no data    : {result.markets_no_data:>6}")
    print(f"  Markets with signal     : {result.markets_with_signal:>6}")
    print(f"  Total trades simulated  : {len(result.trades):>6}")
    print(sep)
    if not result.trades:
        print("  No trades fired. Try lowering --threshold or --window.")
        print(sep + "\n")
        return
    print(f"  Total invested (USDC)   : {result.total_invested:>9.2f}")
    print(f"  Total returned (USDC)   : {result.total_returned:>9.2f}")
    print(f"  Net P&L (USDC)          : {result.total_pnl:>+9.2f}")
    print(f"  Win rate                : {result.win_rate:>8.1f}%")
    print(f"  ROI                     : {result.roi_pct:>+8.1f}%")
    print(f"  Avg entry price         : {result.avg_entry_price:>9.4f}")
    print(sep)

    # Per-trade breakdown (last 20)
    print(f"  {'DATE':10}  {'OUTCOME':4}  {'ENTRY':6}  {'P&L':>7}  {'WIN?':5}  QUESTION")
    print(f"  {'-'*10}  {'-'*4}  {'-'*6}  {'-'*7}  {'-'*5}  {'-'*25}")
    for t in result.trades[-20:]:
        from datetime import datetime, timezone
        date_str = datetime.fromtimestamp(t.end_ts, tz=timezone.utc).strftime("%Y-%m-%d")
        win_str = "YES" if t.resolved else "NO"
        pnl_str = f"{t.pnl_usdc:+.2f}"
        q_short = t.question[:35] + ("…" if len(t.question) > 35 else "")
        print(
            f"  {date_str}  {t.outcome:<4}  {t.entry_price:.4f}  {pnl_str:>7}  {win_str:<5}  {q_short}"
        )
    if len(result.trades) > 20:
        print(f"  … and {len(result.trades) - 20} more trades (see reports/backtest_trades.csv)")
    print(sep + "\n")


def save_trades_report(result: BacktestResult, path: Optional[Path] = None) -> Path:
    if path is None:
        path = REPORTS_DIR / "backtest_trades.csv"
    fieldnames = [
        "date", "condition_id", "outcome", "entry_price", "size_usdc",
        "tokens_bought", "resolved", "pnl_usdc", "roi_pct",
        "secs_before_close", "data_source", "question",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in result.trades:
            from datetime import datetime, timezone
            writer.writerow({
                "date": datetime.fromtimestamp(t.end_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
                "condition_id": t.condition_id,
                "outcome": t.outcome,
                "entry_price": round(t.entry_price, 6),
                "size_usdc": t.size_usdc,
                "tokens_bought": round(t.tokens_bought, 4),
                "resolved": t.resolved,
                "pnl_usdc": round(t.pnl_usdc, 4),
                "roi_pct": round(t.pnl_usdc / t.size_usdc * 100, 2) if t.size_usdc else 0,
                "secs_before_close": round(t.seconds_before_close, 1),
                "data_source": t.data_source,
                "question": t.question,
            })
    logger.info("Trade report saved to %s", path)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest BTC 5-min bot on Polymarket history")
    p.add_argument("--threshold", type=float, default=0.80,
                   help="Minimum price to trigger a trade (default: 0.80)")
    p.add_argument("--window", type=int, default=40,
                   help="Entry window in seconds before close (default: 40)")
    p.add_argument("--size", type=float, default=5.0,
                   help="USDC trade size per market (default: 5.0)")
    p.add_argument("--mode", choices=["trades", "candles"], default="trades",
                   help="Data source: 'trades' (tick-level) or 'candles' (1-min OHLC)")
    p.add_argument("--json", default=str(RAW_DIR / "btc_5min_raw.json"),
                   help="Path to raw JSON file from data_fetcher")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()

    json_path = Path(args.json)
    if not json_path.exists():
        print(f"[ERROR] Data file not found: {json_path}")
        print("Run  python -m src.data_fetcher  first to download historical data.")
        return

    logger.info("Loading data from %s", json_path)
    markets = _load_json(json_path)
    logger.info("Loaded %d markets", len(markets))

    candles_path = RAW_DIR / "btc_5min_candles.csv"
    trades_path = RAW_DIR / "btc_5min_trades.csv"

    candle_index: dict[str, list[dict]] = {}
    trade_index: dict[str, list[dict]] = {}

    if candles_path.exists():
        candle_index = _load_candles_csv(candles_path)
        logger.info("Loaded candles for %d token×market pairs", len(candle_index))

    if trades_path.exists():
        trade_index = _load_trades_csv(trades_path)
        logger.info("Loaded trade ticks for %d token×market pairs", len(trade_index))

    if args.mode == "trades" and not trade_index:
        logger.warning("No trade tick data found — falling back to candles mode")
        args.mode = "candles"

    if args.mode == "candles" and not candle_index:
        print("[ERROR] No candle data found and trade data unavailable. Re-run data_fetcher.")
        return

    logger.info(
        "Running backtest | threshold=%.2f | window=%ds | size=%.2f | mode=%s",
        args.threshold, args.window, args.size, args.mode,
    )
    result = run_backtest(
        markets=markets,
        candle_index=candle_index,
        trade_index=trade_index,
        threshold=args.threshold,
        window_secs=args.window,
        trade_size_usdc=args.size,
        mode=args.mode,
    )

    print_report(result, threshold=args.threshold, window_secs=args.window)

    if result.trades:
        report_path = save_trades_report(result)
        print(f"  Full trade log saved to: {report_path}\n")


if __name__ == "__main__":
    main()
