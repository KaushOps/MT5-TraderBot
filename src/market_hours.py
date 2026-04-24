"""Market-hours detection for MT5 symbols.

Different asset classes trade on very different schedules. If the bot opens a
market order when the symbol is actually closed, MT5 either rejects the order
(retcode 10018 = MARKET_CLOSED) or fills a stale price that gaps at next open —
both bad outcomes.

This module provides `is_tradable(symbol)` and `is_near_close(symbol)` using a
two-layered approach:

  1. **Tick freshness** (primary, works for ALL brokers / symbols):
     If the most recent tick is older than a threshold, the market is closed.
  2. **Known asset-class windows** (fallback for near-close detection):
     Asset class is inferred from MT5 symbol_info.path (e.g. "Stocks\\US\\NFLX",
     "Forex\\Majors\\EURUSD", "Metals\\XAUUSD", "Energies\\USOIL"). For each
     class we know the approximate UTC open/close to flag "about to close".

Nothing here is broker-specific beyond the path string convention, which Exness
and most MT5 brokers follow.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, time as dt_time, timezone
from typing import Optional

import MetaTrader5 as mt5

from src.config_loader import CONFIG

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Asset classification
# ---------------------------------------------------------------------------

# Asset classes used internally. Each has a trade-hours profile below.
CLASS_CRYPTO   = "crypto"
CLASS_FOREX    = "forex"
CLASS_METALS   = "metals"
CLASS_ENERGIES = "energies"
CLASS_STOCKS   = "stocks"
CLASS_INDICES  = "indices"
CLASS_UNKNOWN  = "unknown"


def _classify_by_path(path: str, symbol: str) -> str:
    """Infer asset class from symbol_info.path (case-insensitive substring match)."""
    p = (path or "").lower()
    s = (symbol or "").upper()

    if "crypto" in p or s in {"BTCUSD", "ETHUSD", "LTCUSD", "XRPUSD", "SOLUSD"}:
        return CLASS_CRYPTO
    if "stock" in p or "share" in p or "equit" in p:
        return CLASS_STOCKS
    if "metal" in p or s in {"XAUUSD", "XAGUSD", "XAUEUR", "XPTUSD", "XPDUSD"}:
        return CLASS_METALS
    if "energ" in p or "oil" in p or "gas" in p or s in {"USOIL", "UKOIL", "WTIUSD", "BRENTUSD", "NGAS"}:
        return CLASS_ENERGIES
    if "ind" in p or s in {"US30", "SPX500", "NAS100", "GER40", "UK100", "JP225"}:
        return CLASS_INDICES
    if "forex" in p or "fx" in p or (len(s) == 6 and s.isalpha()):
        return CLASS_FOREX
    return CLASS_UNKNOWN


# ---------------------------------------------------------------------------
# Trading session windows (UTC)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SessionWindow:
    """Weekly trading window. weekday_from/to are Python ints (Mon=0, Sun=6).

    open_utc and close_utc are time-of-day in UTC. A symbol is open if the
    current UTC time is within ANY of the class's SessionWindow entries.

    For 24/7 markets (crypto) we use a single always-open window.
    """
    weekday_from: int        # inclusive, 0 = Monday
    weekday_to:   int        # inclusive
    open_utc:   dt_time
    close_utc:  dt_time

    def contains(self, now: datetime) -> bool:
        wd = now.weekday()
        if not (self.weekday_from <= wd <= self.weekday_to):
            return False
        t = now.timetz().replace(tzinfo=None)
        if self.open_utc <= self.close_utc:
            return self.open_utc <= t < self.close_utc
        # Session crosses midnight
        return t >= self.open_utc or t < self.close_utc


# Schedules are approximate broker defaults. Individual brokers may differ by
# ±1 hour around DST. They serve as a "belt and braces" check on top of tick
# freshness — not a substitute for it.
_SESSION_WINDOWS: dict[str, list[SessionWindow]] = {
    CLASS_CRYPTO: [
        # Always open.
        SessionWindow(0, 6, dt_time(0, 0), dt_time(23, 59, 59)),
    ],
    CLASS_FOREX: [
        # Sun 22:00 UTC → Fri 22:00 UTC, continuous.
        SessionWindow(6, 6, dt_time(22, 0), dt_time(23, 59, 59)),  # Sunday evening
        SessionWindow(0, 3, dt_time(0, 0), dt_time(23, 59, 59)),   # Mon–Thu full
        SessionWindow(4, 4, dt_time(0, 0), dt_time(22, 0)),        # Friday until 22:00
    ],
    CLASS_METALS: [
        # Same as forex, with a small daily break (~21:00–22:00 UTC) we ignore
        # because tick-freshness will catch it.
        SessionWindow(6, 6, dt_time(22, 0), dt_time(23, 59, 59)),
        SessionWindow(0, 3, dt_time(0, 0), dt_time(23, 59, 59)),
        SessionWindow(4, 4, dt_time(0, 0), dt_time(22, 0)),
    ],
    CLASS_ENERGIES: [
        # USOIL/UKOIL: ~Sun 23:00 UTC → Fri 22:00 UTC with daily 1h break.
        SessionWindow(6, 6, dt_time(23, 0), dt_time(23, 59, 59)),
        SessionWindow(0, 3, dt_time(0, 0), dt_time(23, 59, 59)),
        SessionWindow(4, 4, dt_time(0, 0), dt_time(22, 0)),
    ],
    CLASS_STOCKS: [
        # Regular US cash session: 13:30–20:00 UTC Mon–Fri (9:30–16:00 ET).
        # Pre/post-market varies by broker; we deliberately stay conservative.
        SessionWindow(0, 4, dt_time(13, 30), dt_time(20, 0)),
    ],
    CLASS_INDICES: [
        # US equity index CFDs usually run Sun 22:00 → Fri 21:00 with daily break.
        SessionWindow(6, 6, dt_time(22, 0), dt_time(23, 59, 59)),
        SessionWindow(0, 3, dt_time(0, 0), dt_time(23, 59, 59)),
        SessionWindow(4, 4, dt_time(0, 0), dt_time(21, 0)),
    ],
    CLASS_UNKNOWN: [
        # Default to 24/5 — we'll still rely primarily on tick freshness.
        SessionWindow(6, 6, dt_time(22, 0), dt_time(23, 59, 59)),
        SessionWindow(0, 3, dt_time(0, 0), dt_time(23, 59, 59)),
        SessionWindow(4, 4, dt_time(0, 0), dt_time(22, 0)),
    ],
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class MarketHours:
    """Lightweight check for whether a given MT5 symbol is currently tradable."""

    def __init__(self, mt5_api):
        self.mt5_api = mt5_api
        self.enabled = bool(CONFIG.get("enable_market_hours", True))
        # Tick is considered stale if older than this many seconds.
        self.max_tick_age_sec = int(float(CONFIG.get("max_tick_age_sec") or 120))
        # Block new entries within this many seconds of session close.
        self.pre_close_block_sec = int(float(CONFIG.get("pre_close_block_sec") or 900))
        self._class_cache: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Classification (cached per symbol)
    # ------------------------------------------------------------------

    def classify(self, symbol: str) -> str:
        if symbol in self._class_cache:
            return self._class_cache[symbol]
        try:
            actual = self.mt5_api.resolve_symbol(symbol) or symbol
            info = mt5.symbol_info(actual)
            path = getattr(info, "path", "") if info else ""
        except Exception:
            path = ""
        cls = _classify_by_path(path, symbol)
        self._class_cache[symbol] = cls
        return cls

    # ------------------------------------------------------------------
    # Primary check: is the market tradable right now?
    # ------------------------------------------------------------------

    def is_tradable(self, symbol: str) -> tuple[bool, str]:
        """Return (tradable, reason). reason is empty when tradable=True."""
        if not self.enabled:
            return True, ""

        actual = self.mt5_api.resolve_symbol(symbol)
        if actual is None:
            return False, f"symbol_not_found:{symbol}"

        # Check broker-reported trade mode first — fast path.
        info = mt5.symbol_info(actual)
        if info is None:
            return False, "symbol_info_unavailable"
        # mt5.SYMBOL_TRADE_MODE_DISABLED = 0, CLOSEONLY = 3
        trade_mode = getattr(info, "trade_mode", None)
        if trade_mode == getattr(mt5, "SYMBOL_TRADE_MODE_DISABLED", 0):
            return False, "broker_trade_mode_disabled"
        if trade_mode == getattr(mt5, "SYMBOL_TRADE_MODE_CLOSEONLY", 3):
            return False, "broker_trade_mode_closeonly"

        # Tick freshness — the most reliable signal.
        tick = mt5.symbol_info_tick(actual)
        if tick is None:
            return False, "no_tick"
        tick_age = int(time.time() - int(tick.time))
        if tick_age > self.max_tick_age_sec:
            return False, f"stale_tick:{tick_age}s>{self.max_tick_age_sec}s"

        # Session window as safety net.
        cls = self.classify(symbol)
        now_utc = datetime.now(timezone.utc)
        windows = _SESSION_WINDOWS.get(cls, _SESSION_WINDOWS[CLASS_UNKNOWN])
        if not any(w.contains(now_utc) for w in windows):
            return False, f"outside_session:{cls}"

        return True, ""

    # ------------------------------------------------------------------
    # Secondary check: is the market about to close?
    # ------------------------------------------------------------------

    def seconds_until_close(self, symbol: str) -> Optional[int]:
        """Best-effort seconds until the end of the current session window.

        Returns None if the market is currently closed, or if the asset is
        24/7 crypto (no close).
        """
        cls = self.classify(symbol)
        if cls == CLASS_CRYPTO:
            return None
        now_utc = datetime.now(timezone.utc)
        windows = _SESSION_WINDOWS.get(cls, _SESSION_WINDOWS[CLASS_UNKNOWN])
        active = next((w for w in windows if w.contains(now_utc)), None)
        if active is None:
            return None
        # Build close datetime today (UTC)
        close_today = now_utc.replace(
            hour=active.close_utc.hour,
            minute=active.close_utc.minute,
            second=active.close_utc.second,
            microsecond=0,
        )
        delta = (close_today - now_utc).total_seconds()
        if delta < 0:
            # Session wraps past midnight; treat as "plenty of time"
            return 24 * 3600
        return int(delta)

    def is_near_close(self, symbol: str) -> tuple[bool, int | None]:
        """True if the market is within `pre_close_block_sec` of closing."""
        if not self.enabled or self.pre_close_block_sec <= 0:
            return False, None
        remaining = self.seconds_until_close(symbol)
        if remaining is None:
            return False, None
        return (remaining <= self.pre_close_block_sec, remaining)
