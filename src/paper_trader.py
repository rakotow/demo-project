"""
Polymarket BTC 5-min paper trading (forward test).

Runs the same strategy as the live bot but never places real orders.
Instead it logs every signal as a simulated trade, waits for the market
to resolve, then records the outcome and prints a running P&L dashboard.

State is persisted to data/paper_trades.json so the session survives
restarts and you can leave it running for days.

Usage
-----
    python -m src.paper_trader

    # Override parameters:
    python -m src.paper_trader --threshold 0.85 --window 40 --size 10

    # Print current results and exit (no live loop):
    python -m src.paper_trader --report
"""

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PRICE_THRESHOLD = float(os.getenv("PRICE_THRESHOLD", "0.80"))
ENTRY_WINDOW_SECONDS = int(os.getenv("ENTRY_WINDOW_SECONDS", "40"))
TRADE_SIZE_USDC = float(os.getenv("TRADE_SIZE_USDC", "5.0"))

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = DATA_DIR / "paper_trades.json"

POLL_NORMAL = 10       # seconds between market scans when no signal pending
POLL_FAST = 2          # seconds when inside an entry window
RESOLUTION_POLL = 30   # seconds between resolution checks on open positions


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class PaperTrade:
    # Identity
    condition_id: str
    question: str
    token_id: str
    outcome: str          # 'YES' or 'NO'

    # Entry
    entry_price: float
    size_usdc: float
    tokens_bought: float
    entry_time: str       # ISO-8601

    # Market timing
    end_ts: int           # Unix timestamp of market close

    # Resolution (filled in later)
    resolved_outcome: Optional[str] = None   # 'YES' | 'NO' | None
    won: Optional[bool] = None
    pnl_usdc: Optional[float] = None
    resolution_time: Optional[str] = None


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def _load_state() -> tuple[dict[str, PaperTrade], list[PaperTrade]]:
    """Return (open_positions, closed_positions) from disk."""
    if not STATE_FILE.exists():
        return {}, []
    with open(STATE_FILE) as f:
        raw = json.load(f)
    open_pos = {cid: PaperTrade(**d) for cid, d in raw.get("open", {}).items()}
    closed = [PaperTrade(**d) for d in raw.get("closed", [])]
    return open_pos, closed


def _save_state(open_pos: dict[str, PaperTrade], closed: list[PaperTrade]) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(
            {
                "open": {cid: asdict(t) for cid, t in open_pos.items()},
                "closed": [asdict(t) for t in closed],
            },
            f,
            indent=2,
        )


# ---------------------------------------------------------------------------
# Polymarket API helpers (no auth required for read-only)
# ---------------------------------------------------------------------------

_session = requests.Session()


def _get(url: str, params: dict = None, timeout: int = 10) -> dict | list:
    resp = _session.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def fetch_active_btc_5min_markets() -> list[dict]:
    data = _get(f"{GAMMA_API}/markets", {
        "active": "true", "closed": "false", "tag": "crypto", "limit": 100,
    })
    results = []
    for m in data:
        q = (m.get("question") or "").lower()
        if "btc" in q and ("5 min" in q or "5min" in q or "5-min" in q):
            m["end_date_iso"] = m.get("endDate") or ""
            results.append(m)
    return results


def fetch_market_by_condition(condition_id: str) -> Optional[dict]:
    """Fetch a single market by condition ID to check resolution."""
    try:
        data = _get(f"{GAMMA_API}/markets", {"conditionId": condition_id, "limit": 1})
        if isinstance(data, list) and data:
            return data[0]
        if isinstance(data, dict):
            return data
    except Exception as exc:
        logger.debug("Could not fetch market %s: %s", condition_id, exc)
    return None


def get_best_ask(token_id: str) -> Optional[float]:
    """Return the lowest ask price from the CLOB order book."""
    try:
        book = _get(f"{CLOB_API}/book", {"token_id": token_id})
        asks = book.get("asks") or []
        if not asks:
            return None
        return float(min(asks, key=lambda o: float(o["price"]))["price"])
    except Exception as exc:
        logger.debug("Order book fetch failed for %s: %s", token_id, exc)
    # Fallback: midpoint
    try:
        mid = _get(f"{CLOB_API}/midpoint", {"token_id": token_id})
        val = mid.get("mid")
        return float(val) if val else None
    except Exception:
        return None


def seconds_until_close(end_date_iso: str) -> float:
    end_dt = datetime.fromisoformat(end_date_iso.replace("Z", "+00:00"))
    return (end_dt - datetime.now(tz=timezone.utc)).total_seconds()


def parse_outcome(market: dict) -> Optional[str]:
    prices = market.get("outcomePrices")
    if not prices:
        return None
    outcomes = market.get("outcomes") or ["YES", "NO"]
    try:
        for outcome, price in zip(outcomes, prices):
            if float(price) == 1.0:
                return str(outcome).upper()
    except (TypeError, ValueError):
        pass
    return None


# ---------------------------------------------------------------------------
# Core paper trader
# ---------------------------------------------------------------------------

class PaperTrader:
    def __init__(
        self,
        threshold: float = PRICE_THRESHOLD,
        window_secs: int = ENTRY_WINDOW_SECONDS,
        trade_size: float = TRADE_SIZE_USDC,
    ):
        self.threshold = threshold
        self.window_secs = window_secs
        self.trade_size = trade_size

        self.open_pos, self.closed = _load_state()
        logger.info(
            "Loaded state: %d open, %d closed positions",
            len(self.open_pos), len(self.closed),
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        logger.info(
            "Paper trader started | threshold=%.2f | window=%ds | size=%.2f USDC",
            self.threshold, self.window_secs, self.trade_size,
        )
        _last_resolution_check = 0.0

        while True:
            try:
                in_window = self._scan_active_markets()

                # Periodically try to resolve open positions
                now = time.time()
                if now - _last_resolution_check >= RESOLUTION_POLL:
                    self._resolve_open_positions()
                    _last_resolution_check = now
                    self._print_dashboard()

                time.sleep(POLL_FAST if in_window else POLL_NORMAL)

            except KeyboardInterrupt:
                logger.info("Interrupted — saving state and exiting.")
                _save_state(self.open_pos, self.closed)
                self._print_dashboard()
                break
            except Exception as exc:
                logger.error("Error in main loop: %s", exc, exc_info=True)
                time.sleep(POLL_NORMAL)

    # ------------------------------------------------------------------
    # Market scanning
    # ------------------------------------------------------------------

    def _scan_active_markets(self) -> bool:
        """Scan active markets for entry signals. Returns True if any market is in the entry window."""
        try:
            markets = fetch_active_btc_5min_markets()
        except Exception as exc:
            logger.warning("Failed to fetch active markets: %s", exc)
            return False

        in_window = False
        for market in markets:
            condition_id = market.get("conditionId") or market.get("condition_id", "")
            if not condition_id or condition_id in self.open_pos:
                continue  # already have a position

            end_date_iso = market.get("end_date_iso", "")
            if not end_date_iso:
                continue

            secs_left = seconds_until_close(end_date_iso)
            if secs_left <= 0 or secs_left > self.window_secs:
                continue

            in_window = True
            logger.info(
                "In entry window (%.1fs left) | %s",
                secs_left, market.get("question", ""),
            )

            self._evaluate_market(market, secs_left)

        return in_window

    def _evaluate_market(self, market: dict, secs_left: float) -> None:
        condition_id = market.get("conditionId") or market.get("condition_id", "")
        question = market.get("question", "")
        end_date_iso = market.get("end_date_iso", "")
        end_ts = int(datetime.fromisoformat(end_date_iso.replace("Z", "+00:00")).timestamp())

        tokens: list[dict] = market.get("tokens") or []
        if not tokens:
            clob_ids = market.get("clobTokenIds") or []
            outcomes_list = market.get("outcomes") or ["YES", "NO"]
            tokens = [
                {"token_id": tid, "outcome": out}
                for tid, out in zip(clob_ids, outcomes_list)
            ]

        for token in tokens:
            token_id = token.get("token_id") or token.get("tokenId") or ""
            outcome = token.get("outcome", "")
            if not token_id:
                continue

            ask = get_best_ask(token_id)
            if ask is None:
                logger.debug("No ask for %s (%s)", token_id[:8], outcome)
                continue

            logger.info(
                "  %s ask=%.4f  threshold=%.2f",
                outcome or token_id[:8], ask, self.threshold,
            )

            if ask >= self.threshold:
                self._record_paper_trade(
                    condition_id=condition_id,
                    question=question,
                    token_id=token_id,
                    outcome=outcome,
                    ask=ask,
                    end_ts=end_ts,
                )
                return  # one trade per market

    def _record_paper_trade(
        self,
        condition_id: str,
        question: str,
        token_id: str,
        outcome: str,
        ask: float,
        end_ts: int,
    ) -> None:
        tokens_bought = self.trade_size / ask
        trade = PaperTrade(
            condition_id=condition_id,
            question=question,
            token_id=token_id,
            outcome=outcome,
            entry_price=ask,
            size_usdc=self.trade_size,
            tokens_bought=tokens_bought,
            entry_time=datetime.now(tz=timezone.utc).isoformat(),
            end_ts=end_ts,
        )
        self.open_pos[condition_id] = trade
        _save_state(self.open_pos, self.closed)

        logger.info(
            "PAPER TRADE | %s @ %.4f | %.4f tokens | %s",
            outcome, ask, tokens_bought, question[:50],
        )

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def _resolve_open_positions(self) -> None:
        if not self.open_pos:
            return

        now_ts = int(time.time())
        to_resolve = [
            cid for cid, t in self.open_pos.items()
            if now_ts > t.end_ts + 10  # give 10s grace after close
        ]

        for condition_id in to_resolve:
            self._try_resolve(condition_id)

    def _try_resolve(self, condition_id: str) -> None:
        trade = self.open_pos[condition_id]
        market = fetch_market_by_condition(condition_id)

        if market is None:
            logger.debug("Market not found yet for %s", condition_id)
            return

        resolved_outcome = parse_outcome(market)
        if resolved_outcome is None:
            logger.debug("Market %s not yet resolved", condition_id)
            return

        won = resolved_outcome == trade.outcome.upper()
        returned = trade.tokens_bought * 1.0 if won else 0.0
        pnl = returned - trade.size_usdc

        trade.resolved_outcome = resolved_outcome
        trade.won = won
        trade.pnl_usdc = round(pnl, 4)
        trade.resolution_time = datetime.now(tz=timezone.utc).isoformat()

        self.closed.append(trade)
        del self.open_pos[condition_id]
        _save_state(self.open_pos, self.closed)

        result_str = "WON " if won else "LOST"
        logger.info(
            "RESOLVED [%s] %s | pnl=%+.2f USDC | resolved=%s | %s",
            result_str, trade.outcome, pnl, resolved_outcome, trade.question[:50],
        )

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    def _print_dashboard(self) -> None:
        sep = "=" * 60
        now_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"\n{sep}")
        print(f"  PAPER TRADING DASHBOARD   {now_str}")
        print(sep)
        print(f"  Strategy : ask >= {self.threshold:.2f} in last {self.window_secs}s | size={self.trade_size:.2f} USDC")
        print(f"  Open positions  : {len(self.open_pos)}")
        print(f"  Closed trades   : {len(self.closed)}")

        if self.open_pos:
            print(f"\n  {'OPEN POSITIONS':}")
            for cid, t in self.open_pos.items():
                secs_left = t.end_ts - int(time.time())
                status = f"closes in {secs_left}s" if secs_left > 0 else "awaiting resolution"
                print(f"    {t.outcome:<3} @ {t.entry_price:.4f}  [{status}]  {t.question[:40]}")

        if not self.closed:
            print(f"\n  No resolved trades yet.")
            print(sep + "\n")
            return

        total_invested = sum(t.size_usdc for t in self.closed)
        total_pnl = sum(t.pnl_usdc for t in self.closed if t.pnl_usdc is not None)
        wins = [t for t in self.closed if t.won]
        win_rate = len(wins) / len(self.closed) * 100
        roi = total_pnl / total_invested * 100 if total_invested else 0.0

        print(f"\n  {'Metric':<26} {'Value':>10}")
        print(f"  {'-'*26}  {'-'*10}")
        print(f"  {'Total invested (USDC)':<26} {total_invested:>10.2f}")
        print(f"  {'Total P&L (USDC)':<26} {total_pnl:>+10.2f}")
        print(f"  {'Win rate':<26} {win_rate:>9.1f}%")
        print(f"  {'ROI':<26} {roi:>+9.1f}%")

        print(f"\n  Recent trades (last 10):")
        print(f"  {'DATE':16}  {'OUT':3}  {'ENTRY':6}  {'P&L':>7}  {'W?':2}  QUESTION")
        print(f"  {'-'*16}  {'-'*3}  {'-'*6}  {'-'*7}  {'-'*2}  {'-'*28}")
        for t in self.closed[-10:]:
            date_str = datetime.fromisoformat(t.entry_time).strftime("%m-%d %H:%M")
            pnl_str = f"{t.pnl_usdc:+.2f}" if t.pnl_usdc is not None else "  n/a"
            win_str = "W" if t.won else "L"
            q = t.question[:30] + ("…" if len(t.question) > 30 else "")
            print(f"  {date_str:<16}  {t.outcome:<3}  {t.entry_price:.4f}  {pnl_str:>7}  {win_str:<2}  {q}")

        print(sep + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Polymarket BTC 5-min paper trader")
    p.add_argument("--threshold", type=float, default=PRICE_THRESHOLD,
                   help=f"Min ask price to trigger trade (default: {PRICE_THRESHOLD})")
    p.add_argument("--window", type=int, default=ENTRY_WINDOW_SECONDS,
                   help=f"Entry window in seconds before close (default: {ENTRY_WINDOW_SECONDS})")
    p.add_argument("--size", type=float, default=TRADE_SIZE_USDC,
                   help=f"Simulated USDC per trade (default: {TRADE_SIZE_USDC})")
    p.add_argument("--report", action="store_true",
                   help="Print current results from saved state and exit")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    trader = PaperTrader(
        threshold=args.threshold,
        window_secs=args.window,
        trade_size=args.size,
    )
    if args.report:
        trader._resolve_open_positions()
        trader._print_dashboard()
        return
    trader.run()


if __name__ == "__main__":
    main()
