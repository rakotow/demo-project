"""
Polymarket BTC 5-minute trading bot.

Strategy
--------
- Continuously scan active "BTC up/down in 5 minutes" markets.
- For each market, monitor the time remaining until close.
- In the last ENTRY_WINDOW_SECONDS (default 40s), check the YES and NO token prices.
- If any token's ask price is >= PRICE_THRESHOLD (default 0.80 USDC), buy it.
- Each market is only traded once per lifetime to avoid duplicate orders.

Usage
-----
    pip install -r requirements.txt
    cp .env.example .env        # fill in your credentials
    python -m src.bot

Environment variables (see .env.example)
-----------------------------------------
PRIVATE_KEY             EVM private key for your Polygon wallet
CLOB_API_KEY            Polymarket CLOB API key
CLOB_SECRET             Polymarket CLOB API secret
CLOB_PASSPHRASE         Polymarket CLOB API passphrase
TRADE_SIZE_USDC         USDC to spend per trade (default: 5.0)
PRICE_THRESHOLD         Minimum price to trigger trade (default: 0.80)
ENTRY_WINDOW_SECONDS    Seconds before close to start watching (default: 40)
CHAIN_ID                137 for Polygon mainnet (default: 137)
"""

import logging
import os
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

from src.polymarket_client import PolymarketClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (read once at startup)
# ---------------------------------------------------------------------------

PRIVATE_KEY = os.getenv("PRIVATE_KEY", "")
CLOB_API_KEY = os.getenv("CLOB_API_KEY", "")
CLOB_SECRET = os.getenv("CLOB_SECRET", "")
CLOB_PASSPHRASE = os.getenv("CLOB_PASSPHRASE", "")
TRADE_SIZE_USDC = float(os.getenv("TRADE_SIZE_USDC", "5.0"))
PRICE_THRESHOLD = float(os.getenv("PRICE_THRESHOLD", "0.80"))
ENTRY_WINDOW_SECONDS = int(os.getenv("ENTRY_WINDOW_SECONDS", "40"))
CHAIN_ID = int(os.getenv("CHAIN_ID", "137"))

# How often to poll when outside the entry window (seconds)
POLL_INTERVAL_NORMAL = 10
# How often to poll when inside the entry window (seconds)
POLL_INTERVAL_FAST = 2


def _validate_config() -> None:
    missing = [
        name
        for name, val in [
            ("PRIVATE_KEY", PRIVATE_KEY),
            ("CLOB_API_KEY", CLOB_API_KEY),
            ("CLOB_SECRET", CLOB_SECRET),
            ("CLOB_PASSPHRASE", CLOB_PASSPHRASE),
        ]
        if not val or val.startswith("your-") or val == "0xYOUR_PRIVATE_KEY_HERE"
    ]
    if missing:
        raise EnvironmentError(
            f"Missing or placeholder credentials in .env: {', '.join(missing)}\n"
            "Copy .env.example to .env and fill in your real values."
        )


# ---------------------------------------------------------------------------
# Bot logic
# ---------------------------------------------------------------------------

class BtcBot:
    def __init__(self, client: PolymarketClient):
        self.client = client
        # Track markets we've already traded to avoid double-orders.
        # Maps condition_id -> outcome token_id that was bought.
        self._traded: dict[str, str] = {}

    def run(self) -> None:
        logger.info(
            "Bot started | threshold=%.2f | entry_window=%ds | trade_size=%.2f USDC",
            PRICE_THRESHOLD,
            ENTRY_WINDOW_SECONDS,
            TRADE_SIZE_USDC,
        )
        while True:
            try:
                self._tick()
            except KeyboardInterrupt:
                logger.info("Interrupted — shutting down.")
                break
            except Exception as exc:
                logger.error("Unexpected error in tick: %s", exc, exc_info=True)
                time.sleep(POLL_INTERVAL_NORMAL)

    def _tick(self) -> None:
        markets = self.client.get_active_btc_5min_markets()
        if not markets:
            logger.info("No active BTC 5-min markets found. Sleeping %ds…", POLL_INTERVAL_NORMAL)
            time.sleep(POLL_INTERVAL_NORMAL)
            return

        any_in_window = False

        for market in markets:
            condition_id: str = market.get("conditionId") or market.get("condition_id", "")
            question: str = market.get("question", "")
            end_date_iso: str = market.get("end_date_iso", "")

            if not condition_id or not end_date_iso:
                continue

            if condition_id in self._traded:
                continue  # already acted on this market

            secs_left = self.client.seconds_until_close(end_date_iso)

            if secs_left <= 0:
                logger.debug("Market already closed: %s", question)
                continue

            if secs_left > ENTRY_WINDOW_SECONDS:
                logger.debug(
                    "%.0fs until close — not yet in window (%s)",
                    secs_left,
                    question,
                )
                continue

            # Inside the entry window — start checking prices.
            any_in_window = True
            logger.info(
                "In entry window (%.1fs left) | %s", secs_left, question
            )

            tokens: list[dict] = market.get("tokens") or market.get("clobTokenIds") or []
            if not tokens:
                logger.warning("No tokens found for market %s", condition_id)
                continue

            self._evaluate_and_trade(condition_id, question, tokens)

        sleep_duration = POLL_INTERVAL_FAST if any_in_window else POLL_INTERVAL_NORMAL
        time.sleep(sleep_duration)

    def _evaluate_and_trade(
        self,
        condition_id: str,
        question: str,
        tokens: list[dict],
    ) -> None:
        """Check each token's price; buy the first one that meets the threshold."""
        for token in tokens:
            # Tokens may come as {"token_id": ..., "outcome": ...} or just a string ID.
            if isinstance(token, dict):
                token_id: str = token.get("token_id") or token.get("tokenId") or ""
                outcome: str = token.get("outcome", "")
            else:
                token_id = str(token)
                outcome = ""

            if not token_id:
                continue

            price = self.client.get_best_price(token_id, side="BUY")

            if price is None:
                logger.info("No ask price available for token %s (%s)", token_id, outcome)
                continue

            logger.info(
                "Price check | outcome=%-3s | ask=%.4f | threshold=%.2f",
                outcome or token_id[:8],
                price,
                PRICE_THRESHOLD,
            )

            if price >= PRICE_THRESHOLD:
                logger.info(
                    "SIGNAL: price %.4f >= %.2f for %s [%s] — placing order",
                    price,
                    PRICE_THRESHOLD,
                    outcome,
                    question,
                )
                try:
                    resp = self.client.place_market_buy(token_id, TRADE_SIZE_USDC)
                    logger.info("Order response: %s", resp)
                    # Mark this market as traded regardless of fill outcome.
                    self._traded[condition_id] = token_id
                except Exception as exc:
                    logger.error("Order placement failed: %s", exc, exc_info=True)
                # Only attempt one token per market per cycle.
                return


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    _validate_config()
    client = PolymarketClient(
        private_key=PRIVATE_KEY,
        api_key=CLOB_API_KEY,
        api_secret=CLOB_SECRET,
        api_passphrase=CLOB_PASSPHRASE,
        chain_id=CHAIN_ID,
    )
    bot = BtcBot(client)
    bot.run()


if __name__ == "__main__":
    main()
