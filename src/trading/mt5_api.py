"""MT5 exchange client — replaces hyperliquid_api.py.

Wraps the MetaTrader5 Python library to expose the same interface that
``HyperliquidAPI`` provided so that ``main.py`` and ``decision_maker.py``
require only minimal changes.

IMPORTANT: MetaTrader5 Python bindings are synchronous and communicate
with a locally running MT5 terminal via IPC.  All public methods here
are synchronous; ``main.py`` calls them via ``asyncio.to_thread`` to
keep the event loop non-blocking.
"""

import logging
import time
import math
from datetime import datetime, timezone
from typing import Optional

import MetaTrader5 as mt5

from src.config_loader import CONFIG

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Interval string → MT5 timeframe constant
# ---------------------------------------------------------------------------
_TF_MAP: dict[str, int] = {
    "1m":  mt5.TIMEFRAME_M1,
    "3m":  mt5.TIMEFRAME_M3,
    "5m":  mt5.TIMEFRAME_M5,
    "15m": mt5.TIMEFRAME_M15,
    "30m": mt5.TIMEFRAME_M30,
    "1h":  mt5.TIMEFRAME_H1,
    "2h":  mt5.TIMEFRAME_H2,
    "4h":  mt5.TIMEFRAME_H4,
    "6h":  mt5.TIMEFRAME_H6,
    "8h":  mt5.TIMEFRAME_H8,
    "12h": mt5.TIMEFRAME_H12,
    "1d":  mt5.TIMEFRAME_D1,
    "1w":  mt5.TIMEFRAME_W1,
    "1M":  mt5.TIMEFRAME_MN1,
}

# Suffix variants to auto-detect Exness symbol names
_SYMBOL_VARIANTS = ["", "m", ".pro", "#", "c", "m."]


class MT5API:
    """Facade around the MetaTrader5 library with the same interface as HyperliquidAPI.

    All methods are synchronous.  Call them via ``asyncio.to_thread`` from an
    async context to avoid blocking the event loop.
    """

    def __init__(self):
        self._connected = False
        self._symbol_map: dict[str, str] = {}   # canonical → actual MT5 name
        self._symbol_info_cache: dict[str, object] = {}
        self._typical_spreads: dict[str, float] = {}
        self._account_currency = "USD"

        # Config
        self._login    = CONFIG["mt5_login"]
        self._password = CONFIG["mt5_password"]
        self._server   = CONFIG["mt5_server"]
        self._magic    = int(CONFIG.get("mt5_magic") or 20260419)
        self._slippage = int(CONFIG.get("mt5_slippage") or 20)

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Initialize the MT5 terminal connection.  Returns True on success."""
        for attempt in range(1, 4):
            # Try attaching to already-running terminal first
            if mt5.initialize():
                info = mt5.account_info()
                if info and info.login == self._login:
                    self._connected = True
                    self._account_currency = info.currency
                    logger.info(
                        "Connected to MT5 | Account: %s | Server: %s | "
                        "Balance: %.2f %s | Leverage: 1:%s",
                        info.login, info.server, info.balance,
                        info.currency, info.leverage,
                    )
                    return True

            # Force login with credentials
            if mt5.initialize(
                login=self._login,
                server=self._server,
                password=self._password,
            ):
                self._connected = True
                info = mt5.account_info()
                self._account_currency = info.currency if info else "USD"
                logger.info(
                    "Connected to MT5 | Account: %s | Server: %s | "
                    "Balance: %.2f %s | Leverage: 1:%s",
                    info.login, info.server, info.balance,
                    info.currency, info.leverage,
                )
                return True

            err = mt5.last_error()
            logger.warning("MT5 connect attempt %d/3 failed: %s", attempt, err)
            time.sleep(2.0 * attempt)

        logger.error("Failed to connect to MT5 after 3 attempts.")
        self._connected = False
        return False

    def disconnect(self):
        """Shut down the MT5 connection gracefully."""
        mt5.shutdown()
        self._connected = False
        logger.info("Disconnected from MT5.")

    def _ensure_connected(self):
        """Re-connect if the terminal link was dropped."""
        if not self._connected or mt5.account_info() is None:
            logger.warning("MT5 connection lost — reconnecting …")
            self.connect()

    # ------------------------------------------------------------------
    # Symbol resolution
    # ------------------------------------------------------------------

    def resolve_symbol(self, symbol: str) -> Optional[str]:
        """Return the actual MT5 symbol name for a canonical symbol string.

        Tries common Exness suffix variants and caches the result.
        """
        if symbol in self._symbol_map:
            return self._symbol_map[symbol]

        for suffix in _SYMBOL_VARIANTS:
            candidate = symbol + suffix
            info = mt5.symbol_info(candidate)
            if info is not None:
                if not info.visible:
                    mt5.symbol_select(candidate, True)
                self._symbol_map[symbol] = candidate
                self._symbol_info_cache[candidate] = info
                logger.info("Symbol resolved: %s → %s", symbol, candidate)
                return candidate

        logger.error("Symbol not found on broker: %s", symbol)
        return None

    def _symbol_info(self, symbol: str):
        """Return cached symbol_info or fetch fresh."""
        actual = self.resolve_symbol(symbol)
        if actual is None:
            return None
        if actual not in self._symbol_info_cache:
            self._symbol_info_cache[actual] = mt5.symbol_info(actual)
        return self._symbol_info_cache[actual]

    # ------------------------------------------------------------------
    # Lot size calculation
    # ------------------------------------------------------------------

    def calc_lots(self, symbol: str, allocation_usd: float, price: float) -> float:
        """Convert a USD notional allocation to a valid lot size.

        lots = allocation_usd / (contract_size × price)
        Rounded to the symbol's volume_step and clamped to [volume_min, volume_max].
        """
        info = self._symbol_info(symbol)
        if info is None or price <= 0:
            return 0.0

        contract_size = info.trade_contract_size  # e.g. 100 000 for EURUSD, 1 for BTCUSD
        raw_lots = allocation_usd / (contract_size * price)

        step = info.volume_step
        lots = math.floor(raw_lots / step) * step
        lots = round(lots, 8)
        lots = max(info.volume_min, min(info.volume_max, lots))

        # Round to the number of decimal places implied by volume_step
        decimals = max(0, -int(math.floor(math.log10(step)))) if step < 1 else 0
        lots = round(lots, decimals)

        logger.debug(
            "calc_lots %s: alloc=$%.2f price=%.5f contract=%g → lots=%.5f",
            symbol, allocation_usd, price, contract_size, lots,
        )
        return lots

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def get_user_state(self) -> dict:
        """Return account balance, equity, and all open positions.

        Mirrors the shape returned by HyperliquidAPI.get_user_state().
        """
        self._ensure_connected()
        acc = mt5.account_info()
        if acc is None:
            logger.error("mt5.account_info() returned None")
            return {"balance": 0.0, "total_value": 0.0, "positions": []}

        raw_positions = mt5.positions_get()
        positions = []
        if raw_positions:
            for p in raw_positions:
                # ── CRITICAL: only manage positions placed by THIS bot (magic filter) ──
                if p.magic != self._magic:
                    continue
                info = mt5.symbol_info(p.symbol)
                contract_size = info.trade_contract_size if info else 1.0
                current_px = self._mid_price(p.symbol)
                if p.type == mt5.ORDER_TYPE_BUY:
                    pnl = (current_px - p.price_open) * p.volume * contract_size
                else:
                    pnl = (p.price_open - current_px) * p.volume * contract_size
                positions.append({
                    "symbol":       p.symbol,
                    "ticket":       p.ticket,
                    "type":         "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL",
                    "volume":       p.volume,
                    "price_open":   p.price_open,
                    "current_price": round(current_px, 5),
                    "sl":           p.sl,
                    "tp":           p.tp,
                    "profit":       p.profit,
                    "swap":         p.swap,
                    "pnl":          round(pnl, 2),
                    "contract_size": contract_size,
                    "magic":        p.magic,
                    "comment":      p.comment,
                })

        return {
            "balance":     acc.balance,
            "total_value": acc.equity,   # equity = balance + open P&L
            "positions":   positions,
        }

    def _mid_price(self, symbol: str) -> float:
        """Return (bid+ask)/2 for a symbol, or 0.0 on error."""
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return 0.0
        return (tick.bid + tick.ask) / 2.0

    def get_current_price(self, symbol: str) -> float:
        """Return mid-price for ``symbol``.  Falls back to 0.0."""
        self._ensure_connected()
        actual = self.resolve_symbol(symbol)
        if actual is None:
            return 0.0
        return self._mid_price(actual)

    def get_account_currency(self) -> str:
        return self._account_currency

    def get_conversion_rate(self, base: str, quote: str) -> float:
        """Get exchange rate from base to quote (e.g. USD to INR)."""
        if base == quote:
            return 1.0
        self._ensure_connected()
        symbol = self.resolve_symbol(f"{base}{quote}")
        if symbol:
            cur = self._mid_price(symbol)
            if cur > 0: return cur
        symbol_inv = self.resolve_symbol(f"{quote}{base}")
        if symbol_inv:
            cur = self._mid_price(symbol_inv)
            if cur > 0: return 1.0 / cur
        return 1.0

    def get_effective_spread(self, symbol: str) -> tuple[float, float, float]:
        """Return (capped_spread_dist, live_spread, typical_spread) in price distance terms."""
        self._ensure_connected()
        actual = self.resolve_symbol(symbol)
        if actual is None:
            return 0.0, 0.0, 0.0
        tick = mt5.symbol_info_tick(actual)
        if not tick:
            return 0.0, 0.0, 0.0
            
        live_spread = abs(tick.ask - tick.bid)
        
        # EMA of spread
        if actual not in self._typical_spreads:
            self._typical_spreads[actual] = live_spread
        else:
            self._typical_spreads[actual] = 0.9 * self._typical_spreads[actual] + 0.1 * live_spread
            
        typical = self._typical_spreads[actual]
        capped = min(live_spread, typical * 3.0)
        return capped, live_spread, typical

    def get_candles(self, symbol: str, interval: str = "5m", count: int = 100) -> list[dict]:
        """Fetch OHLCV bars from MT5 and return them as a list of dicts.

        Returns:
            List of dicts with keys: t, open, high, low, close, volume
        """
        self._ensure_connected()
        actual = self.resolve_symbol(symbol)
        if actual is None:
            return []

        tf = _TF_MAP.get(interval, mt5.TIMEFRAME_M5)
        rates = mt5.copy_rates_from_pos(actual, tf, 0, count)
        if rates is None or len(rates) == 0:
            logger.warning("No candles for %s (%s)", symbol, interval)
            return []

        candles = []
        for r in rates:
            candles.append({
                "t":      int(r["time"]) * 1000,   # ms epoch, like HL
                "open":   float(r["open"]),
                "high":   float(r["high"]),
                "low":    float(r["low"]),
                "close":  float(r["close"]),
                "volume": float(r["tick_volume"]),  # real_volume often 0 on CFDs
            })
        return candles

    def get_open_orders(self) -> list[dict]:
        """Fetch pending orders placed by this bot (matching magic number)."""
        self._ensure_connected()
        orders = mt5.orders_get()
        if orders is None:
            return []
        result = []
        for o in orders:
            if o.magic != self._magic:
                continue
            result.append({
                "ticket":      o.ticket,
                "coin":        o.symbol,
                "symbol":      o.symbol,
                "is_buy":      o.type in (mt5.ORDER_TYPE_BUY, mt5.ORDER_TYPE_BUY_LIMIT,
                                          mt5.ORDER_TYPE_BUY_STOP),
                "size":        o.volume_current,
                "price":       o.price_open,
                "sl":          o.sl,
                "tp":          o.tp,
                "order_type":  o.type,
                "oid":         o.ticket,
            })
        return result

    def get_recent_fills(self, limit: int = 50) -> list[dict]:
        """Return recent closed deals for this bot."""
        self._ensure_connected()
        from_date = datetime(2000, 1, 1)
        to_date = datetime.now()
        deals = mt5.history_deals_get(from_date, to_date)
        if deals is None:
            return []
        # Filter to bot magic, entry/exit deals only, most recent first
        bot_deals = [
            d for d in deals
            if d.magic == self._magic and d.entry in (
                mt5.DEAL_ENTRY_IN, mt5.DEAL_ENTRY_OUT
            )
        ]
        bot_deals = sorted(bot_deals, key=lambda d: d.time, reverse=True)[:limit]
        result = []
        for d in bot_deals:
            result.append({
                "timestamp": datetime.fromtimestamp(d.time, tz=timezone.utc).isoformat(),
                "coin":      d.symbol,
                "symbol":    d.symbol,
                "is_buy":    d.type == mt5.DEAL_TYPE_BUY,
                "size":      d.volume,
                "price":     d.price,
                "profit":    d.profit,
            })
        return result

    def get_funding_rate(self, symbol: str) -> Optional[float]:
        """Return the overnight swap rate for ``symbol`` (3-day swap on Wed).

        Note: MT5 uses swap rates (points per lot per night), not funding rates.
        Returns swap_long as a fractional rate ≈ equivalent of funding rate.
        """
        info = self._symbol_info(symbol)
        if info is None:
            return None
        try:
            # swap_long is points per lot per night; normalise to a small fraction
            price = self._mid_price(self.resolve_symbol(symbol) or symbol)
            if price <= 0:
                return None
            point = info.point
            swap_pts = info.swap_long
            # Convert: value_per_pip = contract_size * point; daily_pnl = swap_pts * point * contract_size
            swap_daily_usd = swap_pts * point * info.trade_contract_size
            swap_rate = swap_daily_usd / (info.trade_contract_size * price) if price > 0 else 0
            return round(swap_rate, 8)
        except Exception:
            return None

    def get_open_interest(self, symbol: str) -> Optional[float]:
        """Open interest is not available in MT5 — returns None gracefully."""
        return None

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def _send_order(self, request: dict, retries: int = 3) -> dict:
        """Send an mt5.order_send() request with retry on REQUOTE / busy."""
        last_result = None
        for attempt in range(1, retries + 1):
            result = mt5.order_send(request)
            if result is None:
                err = mt5.last_error()
                logger.warning("order_send returned None (attempt %d/%d): %s", attempt, retries, err)
                time.sleep(0.5 * attempt)
                continue
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                return {"retcode": result.retcode, "order": result.order,
                        "deal": result.deal, "comment": result.comment}
            # Retryable codes: requote, price off, busy
            if result.retcode in (
                mt5.TRADE_RETCODE_REQUOTE,
                mt5.TRADE_RETCODE_PRICE_OFF,
                mt5.TRADE_RETCODE_SERVER_DISABLES_AT,
                mt5.TRADE_RETCODE_CONNECTION,
            ):
                logger.warning(
                    "order_send retcode %d '%s' (attempt %d/%d) — retrying",
                    result.retcode, result.comment, attempt, retries,
                )
                # Refresh price for market orders
                if request.get("type") in (mt5.ORDER_TYPE_BUY, mt5.ORDER_TYPE_SELL):
                    tick = mt5.symbol_info_tick(request["symbol"])
                    if tick:
                        request["price"] = tick.ask if request["type"] == mt5.ORDER_TYPE_BUY else tick.bid
                time.sleep(0.5 * attempt)
                last_result = result
                continue
            # Non-retryable error
            logger.error(
                "order_send failed: retcode=%d comment='%s'",
                result.retcode, result.comment,
            )
            return {"retcode": result.retcode, "order": 0, "comment": result.comment}
        retcode = last_result.retcode if last_result else -1
        return {"retcode": retcode, "order": 0, "comment": "max retries exceeded"}

    def _filling_mode(self, symbol: str) -> int:
        """Return a supported filling mode for the symbol."""
        info = self._symbol_info(symbol)
        if info is None:
            return mt5.ORDER_FILLING_IOC
        fm = info.filling_mode
        if fm & mt5.ORDER_FILLING_FOK:
            return mt5.ORDER_FILLING_FOK
        if fm & mt5.ORDER_FILLING_IOC:
            return mt5.ORDER_FILLING_IOC
        return mt5.ORDER_FILLING_RETURN

    def place_buy_order(self, symbol: str, allocation_usd: float,
                        tp_price: float = 0.0, sl_price: float = 0.0) -> dict:
        """Open a market BUY (long) position.

        Args:
            symbol:         Canonical symbol name (e.g. 'BTCUSD').
            allocation_usd: Notional USD value to allocate.
            tp_price:       Take-profit price (0 = none).
            sl_price:       Stop-loss price (0 = none).

        Returns:
            Dict with retcode, order ticket, and deal ticket.
        """
        self._ensure_connected()
        actual = self.resolve_symbol(symbol)
        if actual is None:
            return {"retcode": -1, "comment": f"Symbol not found: {symbol}"}
        tick = mt5.symbol_info_tick(actual)
        if tick is None:
            return {"retcode": -1, "comment": f"No tick for {actual}"}
        price = tick.ask
        lots = self.calc_lots(symbol, allocation_usd, price)
        if lots <= 0:
            return {"retcode": -1, "comment": f"Invalid lot size for ${allocation_usd}"}
        request = {
            "action":    mt5.TRADE_ACTION_DEAL,
            "symbol":    actual,
            "volume":    lots,
            "type":      mt5.ORDER_TYPE_BUY,
            "price":     price,
            "sl":        sl_price,
            "tp":        tp_price,
            "deviation": self._slippage,
            "magic":     self._magic,
            "comment":   "V2_BUY",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_mode(symbol),
        }
        logger.info("BUY %s lots=%.5f price=%.5f sl=%.5f tp=%.5f", actual, lots, price, sl_price, tp_price)
        return self._send_order(request)

    def place_sell_order(self, symbol: str, allocation_usd: float,
                         tp_price: float = 0.0, sl_price: float = 0.0) -> dict:
        """Open a market SELL (short) position.

        Args:
            symbol:         Canonical symbol name.
            allocation_usd: Notional USD value to allocate.
            tp_price:       Take-profit price (0 = none).
            sl_price:       Stop-loss price (0 = none).

        Returns:
            Dict with retcode, order ticket, and deal ticket.
        """
        self._ensure_connected()
        actual = self.resolve_symbol(symbol)
        if actual is None:
            return {"retcode": -1, "comment": f"Symbol not found: {symbol}"}
        tick = mt5.symbol_info_tick(actual)
        if tick is None:
            return {"retcode": -1, "comment": f"No tick for {actual}"}
        price = tick.bid
        lots = self.calc_lots(symbol, allocation_usd, price)
        if lots <= 0:
            return {"retcode": -1, "comment": f"Invalid lot size for ${allocation_usd}"}
        request = {
            "action":    mt5.TRADE_ACTION_DEAL,
            "symbol":    actual,
            "volume":    lots,
            "type":      mt5.ORDER_TYPE_SELL,
            "price":     price,
            "sl":        sl_price,
            "tp":        tp_price,
            "deviation": self._slippage,
            "magic":     self._magic,
            "comment":   "V2_SELL",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_mode(symbol),
        }
        logger.info("SELL %s lots=%.5f price=%.5f sl=%.5f tp=%.5f", actual, lots, price, sl_price, tp_price)
        return self._send_order(request)

    def place_limit_buy(self, symbol: str, allocation_usd: float, limit_price: float, tp_price: float = 0.0, sl_price: float = 0.0) -> dict:
        """Place a pending BUY_LIMIT order at ``limit_price``."""
        self._ensure_connected()
        actual = self.resolve_symbol(symbol)
        if actual is None:
            return {"retcode": -1, "comment": f"Symbol not found: {symbol}"}
        price = self.get_current_price(symbol)
        lots = self.calc_lots(symbol, allocation_usd, limit_price or price)
        if lots <= 0:
            return {"retcode": -1, "comment": "Invalid lot size"}
        request = {
            "action":    mt5.TRADE_ACTION_PENDING,
            "symbol":    actual,
            "volume":    lots,
            "type":      mt5.ORDER_TYPE_BUY_LIMIT,
            "price":     limit_price,
            "sl":        sl_price,
            "tp":        tp_price,
            "deviation": self._slippage,
            "magic":     self._magic,
            "comment":   "V2_BUY_LIMIT",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_mode(symbol),
        }
        logger.info("BUY_LIMIT %s lots=%.5f @ %.5f sl=%.5f tp=%.5f", actual, lots, limit_price, sl_price, tp_price)
        return self._send_order(request)

    def place_limit_sell(self, symbol: str, allocation_usd: float, limit_price: float, tp_price: float = 0.0, sl_price: float = 0.0) -> dict:
        """Place a pending SELL_LIMIT order at ``limit_price``."""
        self._ensure_connected()
        actual = self.resolve_symbol(symbol)
        if actual is None:
            return {"retcode": -1, "comment": f"Symbol not found: {symbol}"}
        price = self.get_current_price(symbol)
        lots = self.calc_lots(symbol, allocation_usd, limit_price or price)
        if lots <= 0:
            return {"retcode": -1, "comment": "Invalid lot size"}
        request = {
            "action":    mt5.TRADE_ACTION_PENDING,
            "symbol":    actual,
            "volume":    lots,
            "type":      mt5.ORDER_TYPE_SELL_LIMIT,
            "price":     limit_price,
            "sl":        sl_price,
            "tp":        tp_price,
            "deviation": self._slippage,
            "magic":     self._magic,
            "comment":   "V2_SELL_LIMIT",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_mode(symbol),
        }
        logger.info("SELL_LIMIT %s lots=%.5f @ %.5f sl=%.5f tp=%.5f", actual, lots, limit_price, sl_price, tp_price)
        return self._send_order(request)

    def place_take_profit(self, symbol: str, is_buy: bool, _amount, tp_price: float) -> dict:
        """Modify the open position's TP price (MT5 uses SLTP modify, not separate orders).

        ``_amount`` is accepted for interface parity but ignored — MT5 always
        modifies the position's TP/SL on the position itself.
        """
        return self._modify_position_sltp(symbol, is_buy, tp=tp_price, sl=None)

    def place_stop_loss(self, symbol: str, is_buy: bool, _amount, sl_price: float) -> dict:
        """Modify the open position's SL price."""
        return self._modify_position_sltp(symbol, is_buy, tp=None, sl=sl_price)

    def _modify_position_sltp(self, symbol: str, is_buy: bool,
                               tp: Optional[float], sl: Optional[float]) -> dict:
        """Find the open position for ``symbol`` and update its TP/SL."""
        self._ensure_connected()
        actual = self.resolve_symbol(symbol)
        if actual is None:
            return {"retcode": -1, "comment": f"Symbol not found: {symbol}"}

        pos_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
        positions = mt5.positions_get(symbol=actual)
        if not positions:
            logger.warning("No open position for %s to set TP/SL", actual)
            return {"retcode": -1, "comment": "No open position"}

        # Pick the most recent bot position
        bot_pos = [p for p in positions if p.magic == self._magic]
        if not bot_pos:
            bot_pos = list(positions)
        pos = sorted(bot_pos, key=lambda p: p.time, reverse=True)[0]

        new_tp = tp if tp is not None else pos.tp
        new_sl = sl if sl is not None else pos.sl

        request = {
            "action":   mt5.TRADE_ACTION_SLTP,
            "symbol":   actual,
            "position": pos.ticket,
            "sl":       new_sl or 0.0,
            "tp":       new_tp or 0.0,
        }
        logger.info("Modify SLTP %s ticket=%d sl=%.5f tp=%.5f", actual, pos.ticket, new_sl or 0, new_tp or 0)
        return self._send_order(request)

    def force_modify_sltp_on_market_order(self, symbol: str, is_buy: bool, tp: float, sl: float) -> None:
        """Helper to force attach SL and TP onto a market order after opening."""
        self._modify_position_sltp(symbol, is_buy, tp, sl)

    def close_position(self, symbol: str, ticket: int, volume: float, is_buy: bool) -> dict:
        """Close (or partially close) an open position by ticket."""
        self._ensure_connected()
        actual = self.resolve_symbol(symbol)
        if actual is None:
            return {"retcode": -1, "comment": f"Symbol not found: {symbol}"}
        tick = mt5.symbol_info_tick(actual)
        if tick is None:
            return {"retcode": -1, "comment": "No tick"}
        close_type = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY
        price = tick.bid if is_buy else tick.ask
        request = {
            "action":    mt5.TRADE_ACTION_DEAL,
            "symbol":    actual,
            "volume":    volume,
            "type":      close_type,
            "position":  ticket,
            "price":     price,
            "deviation": self._slippage,
            "magic":     self._magic,
            "comment":   "V2_CLOSE",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_mode(symbol),
        }
        logger.info("CLOSE %s ticket=%d vol=%.5f @ %.5f", actual, ticket, volume, price)
        return self._send_order(request)

    def cancel_all_orders(self, symbol: str) -> dict:
        """Cancel all pending orders for ``symbol`` placed by this bot."""
        self._ensure_connected()
        actual = self.resolve_symbol(symbol)
        if actual is None:
            return {"status": "error", "comment": f"Symbol not found: {symbol}"}
        orders = mt5.orders_get(symbol=actual)
        cancelled = 0
        if orders:
            for o in orders:
                if o.magic != self._magic:
                    continue
                request = {
                    "action": mt5.TRADE_ACTION_REMOVE,
                    "order":  o.ticket,
                }
                result = mt5.order_send(request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    cancelled += 1
                    logger.info("Cancelled pending order %d for %s", o.ticket, actual)
                else:
                    err = result.comment if result else mt5.last_error()
                    logger.warning("Failed to cancel order %d: %s", o.ticket, err)
        return {"status": "ok", "cancelled_count": cancelled}

    # ------------------------------------------------------------------
    # Compatibility shims (kept so main.py needs no further changes)
    # ------------------------------------------------------------------

    def extract_oids(self, order_result: dict) -> list:
        """Return a list containing the order ticket, if present."""
        ticket = order_result.get("order", 0)
        return [ticket] if ticket else []

    async def get_meta_and_ctxs(self, dex=None):
        """No-op shim — MT5 doesn't need a meta-cache pre-load step."""
        return None
