"""
Polymarket CLOB API client wrapper.

Handles market discovery (Gamma API) and order placement (CLOB API).
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import requests
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    ApiCreds,
    MarketOrderArgs,
    OrderType,
    Side,
)
from py_clob_client.constants import POLYGON

logger = logging.getLogger(__name__)

GAMMA_API_BASE = "https://gamma-api.polymarket.com"
CLOB_HOST = "https://clob.polymarket.com"


class PolymarketClient:
    """Thin wrapper around py-clob-client plus Gamma API for market discovery."""

    def __init__(
        self,
        private_key: str,
        api_key: str,
        api_secret: str,
        api_passphrase: str,
        chain_id: int = POLYGON,
    ):
        creds = ApiCreds(
            api_key=api_key,
            api_secret=api_secret,
            api_passphrase=api_passphrase,
        )
        self.clob = ClobClient(
            host=CLOB_HOST,
            chain_id=chain_id,
            key=private_key,
            creds=creds,
            signature_type=1,  # EOA signature
        )
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Market discovery
    # ------------------------------------------------------------------

    def get_active_btc_5min_markets(self) -> list[dict]:
        """
        Fetch active BTC up/down 5-minute markets from the Gamma API.

        Returns a list of market dicts, each containing at minimum:
          - condition_id
          - question
          - end_date_iso (ISO-8601 string)
          - tokens: list of {token_id, outcome}
        """
        params = {
            "active": "true",
            "closed": "false",
            "tag": "crypto",
            "limit": 100,
        }
        resp = self._session.get(f"{GAMMA_API_BASE}/markets", params=params, timeout=10)
        resp.raise_for_status()
        markets = resp.json()

        btc_5min = []
        for m in markets:
            question: str = (m.get("question") or "").lower()
            # Match markets like "Will BTC be higher in the next 5 minutes?"
            if "btc" in question and ("5 min" in question or "5min" in question or "5-min" in question):
                # Normalise field names for downstream use
                m["end_date_iso"] = m.get("endDate") or m.get("end_date_iso") or ""
                btc_5min.append(m)

        logger.debug("Found %d active BTC 5-min markets", len(btc_5min))
        return btc_5min

    # ------------------------------------------------------------------
    # Price helpers
    # ------------------------------------------------------------------

    def get_best_price(self, token_id: str, side: str = "BUY") -> Optional[float]:
        """
        Return the best available price for a token on the given side.

        side: "BUY" (taker pays this to acquire YES/NO tokens)
              "SELL" (taker receives this when selling)

        Returns None if the order book is empty.
        """
        try:
            book = self.clob.get_order_book(token_id)
            if side.upper() == "BUY":
                # Asks are what we pay to buy; take the lowest ask.
                asks = book.asks
                if not asks:
                    return None
                return float(min(asks, key=lambda o: float(o.price)).price)
            else:
                bids = book.bids
                if not bids:
                    return None
                return float(max(bids, key=lambda o: float(o.price)).price)
        except Exception as exc:
            logger.warning("Failed to fetch order book for %s: %s", token_id, exc)
            return None

    def get_midpoint_price(self, token_id: str) -> Optional[float]:
        """Return the midpoint price (average of best bid and best ask)."""
        try:
            midpoint = self.clob.get_midpoint(token_id)
            return float(midpoint.mid) if midpoint and midpoint.mid else None
        except Exception as exc:
            logger.warning("Failed to fetch midpoint for %s: %s", token_id, exc)
            return None

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    def place_market_buy(self, token_id: str, usdc_amount: float) -> dict:
        """
        Place a market buy order for `usdc_amount` USDC worth of `token_id`.

        Returns the order response dict from the CLOB.
        """
        order_args = MarketOrderArgs(
            token_id=token_id,
            amount=usdc_amount,
        )
        signed_order = self.clob.create_market_order(order_args)
        resp = self.clob.post_order(signed_order, OrderType.FOK)
        logger.info(
            "Market buy submitted | token=%s amount=%.2f USDC | response=%s",
            token_id,
            usdc_amount,
            resp,
        )
        return resp

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def seconds_until_close(end_date_iso: str) -> float:
        """Return seconds remaining until the market end time (negative if past)."""
        end_dt = datetime.fromisoformat(end_date_iso.replace("Z", "+00:00"))
        now = datetime.now(tz=timezone.utc)
        return (end_dt - now).total_seconds()
