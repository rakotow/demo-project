"""
Polymarket BTC 5-min paper trader — Dual Limit Strategy (v2)

Strategy
--------
For every upcoming "Bitcoin Up or Down - 5 Minutes" market:

  1. PRE-OPEN  — Simulate placing a BUY limit @ LIMIT_PRICE (default 47c)
                 on BOTH the UP token and the DOWN token.

  2. ACTIVE    — Every few seconds, check the best ask on each token.
                 A leg is "filled" when ask <= LIMIT_PRICE.

  3. STOP-LOSS — In the last STOP_LOSS_WINDOW seconds (default 30s):
                 If only ONE leg has filled and that token's mid-price
                 <= STOP_LOSS_PRICE (default 20c), simulate a market
                 sell at the current best bid (exit the losing leg).

  4. RESOLVE   — When the market closes, calculate P&L:
                 • Both filled → one wins, one loses.
                   Net = (1.00 × tokens_won) − (fill_up + fill_down) per token
                 • One filled, held to close → full win or full loss on that leg
                 • One filled, stopped out  → partial loss (sold at ~bid)
                 • Neither filled            → P&L = 0  (orders never hit)

P&L formula (both legs filled)
  cost    = fill_price_up + fill_price_down   (e.g. 0.47 + 0.47 = 0.94)
  return  = 1.00  (winning token pays out)
  profit  = 1.00 − cost  (e.g. +0.06 per unit)

Usage
-----
  python -m src.paper_trader_v2              # start live loop
  python -m src.paper_trader_v2 --report     # print current state and exit

  Options:
    --limit   FLOAT   Limit price for both legs (default 0.47)
    --stop    FLOAT   Stop-loss threshold (default 0.20)
    --window  INT     Seconds before close to check stop-loss (default 30)
    --size    FLOAT   USDC per leg (default 5.0, so 10 USDC max per market)
"""

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
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

LIMIT_PRICE = float(os.getenv("LIMIT_PRICE", "0.47"))
STOP_LOSS_PRICE = float(os.getenv("STOP_LOSS_PRICE", "0.20"))
STOP_LOSS_WINDOW = int(os.getenv("STOP_LOSS_WINDOW", "30"))
LEG_SIZE_USDC = float(os.getenv("TRADE_SIZE_USDC", "5.0"))

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = DATA_DIR / "paper_trades_v2.json"

POLL_ACTIVE = 3    # seconds between price checks during a market
POLL_IDLE = 15     # seconds between market scans when nothing is active
# How far into a market's life (seconds) we still accept placing orders.
# Accounts for any delay between market creation and our scan cycle.
ORDER_PLACEMENT_CUTOFF = 90


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class MarketStatus(str, Enum):
    PENDING = "PENDING"       # orders placed, market not yet open
    ACTIVE = "ACTIVE"         # market running, monitoring fills
    STOPPED = "STOPPED"       # stop-loss triggered, waiting for close
    RESOLVED = "RESOLVED"     # market closed, P&L calculated


@dataclass
class Leg:
    token_id: str
    outcome: str         # 'YES'/'UP' or 'NO'/'DOWN'
    limit_price: float

    filled: bool = False
    fill_price: Optional[float] = None

    stopped_out: bool = False
    stop_exit_price: Optional[float] = None   # bid price at stop-loss exit

    resolved_win: Optional[bool] = None       # did this leg resolve to 1.00?


@dataclass
class DualLimitTrade:
    condition_id: str
    question: str
    start_ts: int    # Unix timestamp when market opens
    end_ts: int      # Unix timestamp when market closes

    up_leg: Leg
    down_leg: Leg

    status: str = MarketStatus.ACTIVE
    resolved_outcome: Optional[str] = None   # 'YES' | 'NO'
    pnl_usdc: Optional[float] = None
    close_reason: Optional[str] = None       # 'both_filled' | 'one_filled' | 'stopped' | 'no_fill'

    created_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def _serialize(obj):
    if isinstance(obj, Leg):
        return asdict(obj)
    if isinstance(obj, DualLimitTrade):
        d = asdict(obj)
        d["up_leg"] = asdict(obj.up_leg)
        d["down_leg"] = asdict(obj.down_leg)
        return d
    return str(obj)


def _deserialize_trade(d: dict) -> DualLimitTrade:
    d["up_leg"] = Leg(**d["up_leg"])
    d["down_leg"] = Leg(**d["down_leg"])
    return DualLimitTrade(**d)


def _load_state() -> tuple[dict[str, DualLimitTrade], list[DualLimitTrade]]:
    if not STATE_FILE.exists():
        return {}, []
    with open(STATE_FILE) as f:
        raw = json.load(f)
    active = {cid: _deserialize_trade(d) for cid, d in raw.get("active", {}).items()}
    history = [_deserialize_trade(d) for d in raw.get("history", [])]
    return active, history


def _save_state(active: dict[str, DualLimitTrade], history: list[DualLimitTrade]) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(
            {
                "active": {cid: _serialize(t) for cid, t in active.items()},
                "history": [_serialize(t) for t in history],
            },
            f,
            indent=2,
        )


# ---------------------------------------------------------------------------
# Polymarket API helpers
# ---------------------------------------------------------------------------

_session = requests.Session()


def _get(url: str, params: dict = None) -> dict | list:
    r = _session.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def fetch_btc_5min_markets(include_upcoming: bool = True) -> list[dict]:
    """Return active + recently opened BTC 5-min markets."""
    results = []
    for active_flag in (["true"], ["false"]) if include_upcoming else (["true"],):
        try:
            batch = _get(f"{GAMMA_API}/markets", {
                "active": active_flag[0],
                "closed": "false",
                "tag": "crypto",
                "limit": 100,
            })
        except Exception as exc:
            logger.warning("Gamma API error: %s", exc)
            continue
        for m in batch:
            q = (m.get("question") or "").lower()
            if "btc" in q and ("5 min" in q or "5min" in q or "5-min" in q):
                results.append(m)
    return results


def fetch_market_resolution(condition_id: str) -> Optional[str]:
    """Return 'YES'/'NO' if resolved, else None."""
    try:
        data = _get(f"{GAMMA_API}/markets", {"conditionId": condition_id, "limit": 1})
        market = data[0] if isinstance(data, list) and data else data if isinstance(data, dict) else None
        if not market:
            return None
        prices = market.get("outcomePrices")
        outcomes = market.get("outcomes") or ["YES", "NO"]
        if prices:
            for outcome, price in zip(outcomes, prices):
                if float(price) == 1.0:
                    return str(outcome).upper()
    except Exception as exc:
        logger.debug("Resolution fetch failed for %s: %s", condition_id, exc)
    return None


def get_best_ask(token_id: str) -> Optional[float]:
    try:
        book = _get(f"{CLOB_API}/book", {"token_id": token_id})
        asks = book.get("asks") or []
        if asks:
            return float(min(asks, key=lambda o: float(o["price"]))["price"])
    except Exception:
        pass
    # Fallback: midpoint
    try:
        mid = _get(f"{CLOB_API}/midpoint", {"token_id": token_id})
        val = mid.get("mid")
        if val:
            return float(val)
    except Exception:
        pass
    return None


def get_best_bid(token_id: str) -> Optional[float]:
    try:
        book = _get(f"{CLOB_API}/book", {"token_id": token_id})
        bids = book.get("bids") or []
        if bids:
            return float(max(bids, key=lambda o: float(o["price"]))["price"])
    except Exception:
        pass
    return None


def get_mid_price(token_id: str) -> Optional[float]:
    try:
        mid = _get(f"{CLOB_API}/midpoint", {"token_id": token_id})
        val = mid.get("mid")
        return float(val) if val else None
    except Exception:
        return None


def _parse_ts(iso: str) -> Optional[int]:
    if not iso:
        return None
    try:
        return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return None


def _now_ts() -> int:
    return int(time.time())


# ---------------------------------------------------------------------------
# Core paper trader
# ---------------------------------------------------------------------------

class DualLimitPaperTrader:

    def __init__(
        self,
        limit_price: float = LIMIT_PRICE,
        stop_loss_price: float = STOP_LOSS_PRICE,
        stop_loss_window: int = STOP_LOSS_WINDOW,
        leg_size: float = LEG_SIZE_USDC,
    ):
        self.limit_price = limit_price
        self.stop_loss_price = stop_loss_price
        self.stop_loss_window = stop_loss_window
        self.leg_size = leg_size

        self.active, self.history = _load_state()
        logger.info(
            "Loaded state: %d active trades, %d in history",
            len(self.active), len(self.history),
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        logger.info(
            "Dual-limit paper trader started | limit=%.2f | stop=%.2f | window=%ds | leg=%.2f USDC",
            self.limit_price, self.stop_loss_price, self.stop_loss_window, self.leg_size,
        )
        last_dashboard = 0.0

        while True:
            try:
                self._discover_new_markets()
                self._monitor_active_trades()
                self._resolve_closed_trades()

                now = time.time()
                if now - last_dashboard >= 60:
                    self._print_dashboard()
                    last_dashboard = now

                sleep = POLL_ACTIVE if self.active else POLL_IDLE
                time.sleep(sleep)

            except KeyboardInterrupt:
                logger.info("Interrupted — saving and exiting.")
                _save_state(self.active, self.history)
                self._print_dashboard()
                break
            except Exception as exc:
                logger.error("Main loop error: %s", exc, exc_info=True)
                time.sleep(POLL_IDLE)

    # ------------------------------------------------------------------
    # Step 1 — Discover new markets and "place" orders
    # ------------------------------------------------------------------

    def _discover_new_markets(self) -> None:
        try:
            markets = fetch_btc_5min_markets(include_upcoming=True)
        except Exception as exc:
            logger.warning("Market discovery failed: %s", exc)
            return

        now = _now_ts()
        for m in markets:
            condition_id = m.get("conditionId") or m.get("condition_id", "")
            if not condition_id or condition_id in self.active:
                continue
            if any(t.condition_id == condition_id for t in self.history):
                continue  # already processed

            end_ts = _parse_ts(m.get("endDate") or "")
            start_ts = _parse_ts(m.get("startDate") or "")

            if not end_ts or end_ts <= now:
                continue  # already closed

            # Accept: upcoming (start > now) OR opened within the last ORDER_PLACEMENT_CUTOFF secs
            if start_ts and (start_ts - now) > 300:
                # More than 5 minutes until open — skip for now, will catch it closer
                continue
            if start_ts and (now - start_ts) > ORDER_PLACEMENT_CUTOFF:
                continue  # opened too long ago

            tokens: list[dict] = m.get("tokens") or []
            if not tokens:
                clob_ids = m.get("clobTokenIds") or []
                outcomes_raw = m.get("outcomes") or ["YES", "NO"]
                tokens = [{"token_id": tid, "outcome": out}
                          for tid, out in zip(clob_ids, outcomes_raw)]
            if len(tokens) < 2:
                continue

            up_token = self._find_token(tokens, ["YES", "UP", "HIGHER", "ABOVE"])
            down_token = self._find_token(tokens, ["NO", "DOWN", "LOWER", "BELOW"])

            if not up_token or not down_token:
                # Fall back to index order
                up_token = tokens[0]
                down_token = tokens[1]

            trade = DualLimitTrade(
                condition_id=condition_id,
                question=m.get("question", ""),
                start_ts=start_ts or now,
                end_ts=end_ts,
                up_leg=Leg(
                    token_id=up_token.get("token_id") or up_token.get("tokenId", ""),
                    outcome=up_token.get("outcome", "UP"),
                    limit_price=self.limit_price,
                ),
                down_leg=Leg(
                    token_id=down_token.get("token_id") or down_token.get("tokenId", ""),
                    outcome=down_token.get("outcome", "DOWN"),
                    limit_price=self.limit_price,
                ),
                status=MarketStatus.ACTIVE,
            )

            self.active[condition_id] = trade
            _save_state(self.active, self.history)

            open_in = max(0, (trade.start_ts or now) - now)
            logger.info(
                "ORDERS PLACED | limit=%.2f both legs | opens in %.0fs | %s",
                self.limit_price, open_in, trade.question[:55],
            )

    @staticmethod
    def _find_token(tokens: list[dict], keywords: list[str]) -> Optional[dict]:
        for t in tokens:
            outcome = (t.get("outcome") or "").upper()
            if any(k in outcome for k in keywords):
                return t
        return None

    # ------------------------------------------------------------------
    # Step 2 — Monitor active trades (fill detection + stop-loss)
    # ------------------------------------------------------------------

    def _monitor_active_trades(self) -> None:
        now = _now_ts()
        for condition_id, trade in list(self.active.items()):
            if trade.status == MarketStatus.RESOLVED:
                continue

            secs_left = trade.end_ts - now
            if secs_left <= 0:
                continue  # let _resolve_closed_trades handle it

            # --- Fill detection ---
            for leg in (trade.up_leg, trade.down_leg):
                if leg.filled or not leg.token_id:
                    continue
                ask = get_best_ask(leg.token_id)
                if ask is None:
                    continue
                if ask <= self.limit_price:
                    leg.filled = True
                    leg.fill_price = ask
                    logger.info(
                        "FILLED | %s @ %.4f (limit %.2f) | %s",
                        leg.outcome, ask, self.limit_price, trade.question[:45],
                    )

            # --- Stop-loss check (last N seconds, only if one leg is partial) ---
            if secs_left <= self.stop_loss_window:
                up_filled = trade.up_leg.filled and not trade.up_leg.stopped_out
                down_filled = trade.down_leg.filled and not trade.down_leg.stopped_out

                # Exactly one leg is filled and live (unhedged)
                if up_filled != down_filled:
                    exposed_leg = trade.up_leg if up_filled else trade.down_leg
                    mid = get_mid_price(exposed_leg.token_id)

                    if mid is not None and mid <= self.stop_loss_price:
                        bid = get_best_bid(exposed_leg.token_id) or mid
                        exposed_leg.stopped_out = True
                        exposed_leg.stop_exit_price = bid
                        trade.status = MarketStatus.STOPPED
                        logger.info(
                            "STOP-LOSS | %s mid=%.4f <= %.2f | selling @ bid=%.4f | %s",
                            exposed_leg.outcome, mid, self.stop_loss_price,
                            bid, trade.question[:45],
                        )

            _save_state(self.active, self.history)

    # ------------------------------------------------------------------
    # Step 3 — Resolve closed markets
    # ------------------------------------------------------------------

    def _resolve_closed_trades(self) -> None:
        now = _now_ts()
        to_resolve = [
            cid for cid, t in self.active.items()
            if now > t.end_ts + 15  # 15s grace period
        ]
        for condition_id in to_resolve:
            self._settle(condition_id)

    def _settle(self, condition_id: str) -> None:
        trade = self.active[condition_id]
        resolved = fetch_market_resolution(condition_id)
        if resolved is None:
            logger.debug("Market %s not yet resolved", condition_id)
            return

        trade.resolved_outcome = resolved
        outcomes = {"YES", "UP", "HIGHER", "ABOVE"}

        up_won = resolved.upper() in outcomes
        trade.up_leg.resolved_win = up_won
        trade.down_leg.resolved_win = not up_won

        pnl = self._calculate_pnl(trade)
        trade.pnl_usdc = round(pnl, 4)
        trade.status = MarketStatus.RESOLVED
        trade.close_reason = self._close_reason(trade)

        self.history.append(trade)
        del self.active[condition_id]
        _save_state(self.active, self.history)

        emoji = "+" if pnl >= 0 else "-"
        logger.info(
            "SETTLED [%s] | pnl=%+.4f USDC | %s | resolved=%s | %s",
            trade.close_reason, pnl, emoji * abs(int(pnl * 10)),
            resolved, trade.question[:50],
        )

    def _calculate_pnl(self, trade: DualLimitTrade) -> float:
        """
        P&L calculation per scenario:

        Both filled, no stop:
          cost  = fill_up + fill_down  (in USDC, proportional to leg_size)
          return = leg_size / fill_price_winning  (tokens × 1.00)
          net   = return_winning - cost_both_legs

        One filled, not stopped:
          win  → (leg_size / fill_price) × 1.00 − leg_size
          lose → −leg_size

        One filled, stopped out:
          exit_proceeds = tokens_held × stop_exit_bid
          net = exit_proceeds − leg_size

        Neither filled → 0
        """
        up = trade.up_leg
        down = trade.down_leg
        size = self.leg_size

        up_tokens = size / up.fill_price if up.filled and up.fill_price else 0.0
        down_tokens = size / down.fill_price if down.filled and down.fill_price else 0.0

        pnl = 0.0

        # UP leg
        if up.filled:
            if up.stopped_out and up.stop_exit_price is not None:
                pnl += (up_tokens * up.stop_exit_price) - size
            elif up.resolved_win:
                pnl += (up_tokens * 1.0) - size
            else:
                pnl += -size

        # DOWN leg
        if down.filled:
            if down.stopped_out and down.stop_exit_price is not None:
                pnl += (down_tokens * down.stop_exit_price) - size
            elif down.resolved_win:
                pnl += (down_tokens * 1.0) - size
            else:
                pnl += -size

        return pnl

    @staticmethod
    def _close_reason(trade: DualLimitTrade) -> str:
        up_f = trade.up_leg.filled
        down_f = trade.down_leg.filled
        stopped = trade.up_leg.stopped_out or trade.down_leg.stopped_out
        if up_f and down_f:
            return "both_filled"
        if (up_f or down_f) and stopped:
            return "stopped_out"
        if up_f or down_f:
            return "one_filled"
        return "no_fill"

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    def _print_dashboard(self) -> None:
        sep = "=" * 65
        now_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"\n{sep}")
        print(f"  DUAL-LIMIT PAPER TRADER   {now_str}")
        print(sep)
        print(
            f"  Limit={self.limit_price:.2f}c  |  Stop={self.stop_loss_price:.2f}c  |  "
            f"StopWindow={self.stop_loss_window}s  |  Leg={self.leg_size:.2f} USDC"
        )

        # Active markets
        if self.active:
            print(f"\n  ACTIVE ({len(self.active)})")
            now = _now_ts()
            for cid, t in self.active.items():
                secs_left = max(0, t.end_ts - now)
                up_s = "FILLED" if t.up_leg.filled else f"bid@{self.limit_price:.2f}"
                down_s = "FILLED" if t.down_leg.filled else f"bid@{self.limit_price:.2f}"
                status = f"[STOPPED]" if t.status == MarketStatus.STOPPED else f"{secs_left:.0f}s left"
                print(f"    UP:{up_s}  DOWN:{down_s}  {status}  {t.question[:38]}")

        # Summary stats
        resolved = [t for t in self.history if t.pnl_usdc is not None]
        print(f"\n  HISTORY  ({len(resolved)} resolved trades)")

        if not resolved:
            print("  No resolved trades yet.")
            print(sep + "\n")
            return

        total_invested = sum(
            self.leg_size * (t.up_leg.filled + t.down_leg.filled)
            for t in resolved
        )
        total_pnl = sum(t.pnl_usdc for t in resolved)
        wins = [t for t in resolved if (t.pnl_usdc or 0) > 0]
        by_reason: dict[str, int] = {}
        for t in resolved:
            by_reason[t.close_reason or "?"] = by_reason.get(t.close_reason or "?", 0) + 1

        print(f"  {'Total invested (USDC)':<28} {total_invested:>8.2f}")
        print(f"  {'Total P&L (USDC)':<28} {total_pnl:>+8.2f}")
        roi = total_pnl / total_invested * 100 if total_invested else 0
        print(f"  {'ROI':<28} {roi:>+7.1f}%")
        print(f"  {'Profitable trades':<28} {len(wins):>4} / {len(resolved)}")
        for reason, count in sorted(by_reason.items()):
            print(f"  {'  ' + reason:<28} {count:>4}")

        # Recent trades
        print(f"\n  {'DATE':16}  {'UP':6}  {'DN':6}  {'P&L':>7}  {'REASON':12}  QUESTION")
        print(f"  {'-'*16}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*12}  {'-'*25}")
        for t in self.history[-15:]:
            date = datetime.fromisoformat(t.created_at).strftime("%m-%d %H:%M")
            up_s = f"{t.up_leg.fill_price:.2f}" if t.up_leg.filled else "  --  "
            dn_s = f"{t.down_leg.fill_price:.2f}" if t.down_leg.filled else "  --  "
            pnl_s = f"{t.pnl_usdc:>+.4f}" if t.pnl_usdc is not None else "    n/a"
            q = t.question[:28] + ("…" if len(t.question) > 28 else "")
            print(f"  {date:<16}  {up_s:<6}  {dn_s:<6}  {pnl_s}  {t.close_reason or '?':<12}  {q}")

        print(sep + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BTC 5-min dual-limit paper trader")
    p.add_argument("--limit", type=float, default=LIMIT_PRICE,
                   help=f"Limit price for both legs (default: {LIMIT_PRICE})")
    p.add_argument("--stop", type=float, default=STOP_LOSS_PRICE,
                   help=f"Stop-loss price threshold (default: {STOP_LOSS_PRICE})")
    p.add_argument("--window", type=int, default=STOP_LOSS_WINDOW,
                   help=f"Seconds before close to check stop (default: {STOP_LOSS_WINDOW})")
    p.add_argument("--size", type=float, default=LEG_SIZE_USDC,
                   help=f"USDC per leg (default: {LEG_SIZE_USDC})")
    p.add_argument("--report", action="store_true",
                   help="Print dashboard from saved state and exit")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    trader = DualLimitPaperTrader(
        limit_price=args.limit,
        stop_loss_price=args.stop,
        stop_loss_window=args.window,
        leg_size=args.size,
    )
    if args.report:
        trader._resolve_closed_trades()
        trader._print_dashboard()
        return
    trader.run()


if __name__ == "__main__":
    main()
