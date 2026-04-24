"""Centralized risk management for the trading agent.

All safety guards are enforced here, independent of LLM decisions.
The LLM cannot override these limits — they are hard-coded checks
applied before every trade execution.
"""

import logging
from datetime import datetime, timezone

from src.config_loader import CONFIG


class RiskManager:
    """Enforces risk limits on every trade before execution."""

    def __init__(self):
        self.max_position_pct = float(CONFIG.get("max_position_pct") or 10)
        self.max_loss_per_position_pct = float(CONFIG.get("max_loss_per_position_pct") or 20)
        self.max_leverage = float(CONFIG.get("max_leverage") or 10)
        self.max_total_exposure_pct = float(CONFIG.get("max_total_exposure_pct") or 50)
        self.daily_loss_circuit_breaker_pct = float(CONFIG.get("daily_loss_circuit_breaker_pct") or 10)
        self.mandatory_sl_pct = float(CONFIG.get("mandatory_sl_pct") or 2)
        self.max_concurrent_positions = int(CONFIG.get("max_concurrent_positions") or 5)
        self.min_balance_reserve_pct = float(CONFIG.get("min_balance_reserve_pct") or 20)

        # ATR-anchored TP/SL bounds (Phase 1)
        self.tp_atr_mult_min = float(CONFIG.get("tp_atr_mult_min") or 0.8)
        self.tp_atr_mult_max = float(CONFIG.get("tp_atr_mult_max") or 2.0)
        self.sl_atr_mult_min = float(CONFIG.get("sl_atr_mult_min") or 0.8)
        self.sl_atr_mult_max = float(CONFIG.get("sl_atr_mult_max") or 1.5)
        self.min_rr = float(CONFIG.get("min_rr") or 1.2)
        self.max_rr = float(CONFIG.get("max_rr") or 2.5)
        self.default_sl_atr_mult = float(CONFIG.get("default_sl_atr_mult") or 1.0)
        self.default_tp_atr_mult = float(CONFIG.get("default_tp_atr_mult") or 1.5)

        # Daily tracking
        self.daily_high_value = None
        self.daily_high_date = None
        self.circuit_breaker_active = False
        self.circuit_breaker_date = None

    def _reset_daily_if_needed(self, account_value: float):
        """Reset daily high watermark at UTC day boundary."""
        today = datetime.now(timezone.utc).date()
        if self.daily_high_date != today:
            self.daily_high_value = account_value
            self.daily_high_date = today
            self.circuit_breaker_active = False
            self.circuit_breaker_date = None
        elif account_value > self.daily_high_value:
            self.daily_high_value = account_value

    # ------------------------------------------------------------------
    # Individual checks — each returns (allowed: bool, reason: str)
    # ------------------------------------------------------------------

    def check_position_size(self, alloc_usd: float, account_value: float) -> tuple[bool, str]:
        """Single position cannot exceed max_position_pct of account."""
        if account_value <= 0:
            return False, "Account value is zero or negative"
        max_alloc = account_value * (self.max_position_pct / 100.0)
        if alloc_usd > max_alloc:
            return False, (
                f"Allocation ${alloc_usd:.2f} exceeds {self.max_position_pct}% "
                f"of account (${max_alloc:.2f})"
            )
        return True, ""

    def check_total_exposure(self, positions: list[dict], new_alloc: float,
                              account_value: float) -> tuple[bool, str]:
        """Sum of all position notionals + new allocation cannot exceed max_total_exposure_pct."""
        current_exposure = 0.0
        for pos in positions:
            qty = abs(float(pos.get("quantity") or pos.get("volume") or 0))
            entry = float(pos.get("entry_price") or pos.get("price_open") or 0)
            contract = float(pos.get("contract_size") or 1.0)
            current_exposure += qty * entry * contract
        total = current_exposure + new_alloc
        max_exposure = account_value * (self.max_total_exposure_pct / 100.0)
        if total > max_exposure:
            return False, (
                f"Total exposure ${total:.2f} would exceed {self.max_total_exposure_pct}% "
                f"of account (${max_exposure:.2f})"
            )
        return True, ""

    def check_leverage(self, alloc_usd: float, balance: float) -> tuple[bool, str]:
        """Effective leverage of new trade cannot exceed max_leverage."""
        if balance <= 0:
            return False, "Balance is zero or negative"
        effective_lev = alloc_usd / balance
        if effective_lev > self.max_leverage:
            return False, (
                f"Effective leverage {effective_lev:.1f}x exceeds max {self.max_leverage}x"
            )
        return True, ""

    def check_daily_drawdown(self, account_value: float) -> tuple[bool, str]:
        """Activate circuit breaker if account drops max % from daily high."""
        self._reset_daily_if_needed(account_value)
        if self.circuit_breaker_active:
            return False, "Daily loss circuit breaker is active — no new trades until tomorrow (UTC)"
        if self.daily_high_value and self.daily_high_value > 0:
            drawdown_pct = ((self.daily_high_value - account_value) / self.daily_high_value) * 100
            if drawdown_pct >= self.daily_loss_circuit_breaker_pct:
                self.circuit_breaker_active = True
                self.circuit_breaker_date = datetime.now(timezone.utc).date()
                return False, (
                    f"Daily drawdown {drawdown_pct:.2f}% exceeds circuit breaker "
                    f"threshold of {self.daily_loss_circuit_breaker_pct}%"
                )
        return True, ""

    def check_concurrent_positions(self, current_count: int) -> tuple[bool, str]:
        """Limit number of simultaneous open positions."""
        if current_count >= self.max_concurrent_positions:
            return False, (
                f"Already at max concurrent positions ({self.max_concurrent_positions})"
            )
        return True, ""

    def check_balance_reserve(self, balance: float, initial_balance: float) -> tuple[bool, str]:
        """Don't trade if balance falls below reserve threshold."""
        if initial_balance <= 0:
            return True, ""
        min_balance = initial_balance * (self.min_balance_reserve_pct / 100.0)
        if balance < min_balance:
            return False, (
                f"Balance ${balance:.2f} below minimum reserve "
                f"${min_balance:.2f} ({self.min_balance_reserve_pct}% of initial)"
            )
        return True, ""

    # ------------------------------------------------------------------
    # Stop-loss enforcement
    # ------------------------------------------------------------------

    def enforce_stop_loss(self, sl_price: float | None, entry_price: float,
                           is_buy: bool) -> float:
        """Ensure every trade has a stop-loss. Auto-set if missing."""
        if sl_price is not None:
            return sl_price
        # Auto-set SL at mandatory_sl_pct from entry
        sl_distance = entry_price * (self.mandatory_sl_pct / 100.0)
        if is_buy:
            return round(entry_price - sl_distance, 5)
        else:
            return round(entry_price + sl_distance, 5)

    # ------------------------------------------------------------------
    # ATR-anchored TP/SL (Phase 1)
    # ------------------------------------------------------------------

    def enforce_atr_bounds(
        self,
        entry_price: float,
        is_buy: bool,
        tp_price: float | None,
        sl_price: float | None,
        atr: float | None,
    ) -> tuple[float | None, float, dict]:
        """Clamp TP/SL distances to sensible ATR multiples and enforce R:R.

        Returns (tp_price, sl_price, adjustments_log). If ATR is unavailable
        or non-positive, falls back to the legacy percent-based mandatory SL.
        The adjustments_log dict carries human-readable reasons for logging.
        """
        log: dict = {}

        if atr is None or atr <= 0 or entry_price <= 0:
            # Fallback: no ATR → use legacy mandatory_sl_pct based SL
            final_sl = self.enforce_stop_loss(sl_price, entry_price, is_buy)
            log["atr_fallback"] = "no_atr_available_using_pct_sl"
            return tp_price, final_sl, log

        sl_min_dist = atr * self.sl_atr_mult_min
        sl_max_dist = atr * self.sl_atr_mult_max
        tp_min_dist = atr * self.tp_atr_mult_min
        tp_max_dist = atr * self.tp_atr_mult_max

        # --- SL ---
        if sl_price is None or float(sl_price) == 0:
            sl_dist = atr * self.default_sl_atr_mult
            log["sl_source"] = f"defaulted_to_{self.default_sl_atr_mult}xATR"
        else:
            sl_dist = abs(entry_price - float(sl_price))
            if sl_dist < sl_min_dist:
                log["sl_widened"] = (
                    f"from {sl_dist:.6f} to {sl_min_dist:.6f} ({self.sl_atr_mult_min}xATR)"
                )
                sl_dist = sl_min_dist
            elif sl_dist > sl_max_dist:
                log["sl_tightened"] = (
                    f"from {sl_dist:.6f} to {sl_max_dist:.6f} ({self.sl_atr_mult_max}xATR)"
                )
                sl_dist = sl_max_dist

        # --- TP ---
        if tp_price is None or float(tp_price) == 0:
            tp_dist = atr * self.default_tp_atr_mult
            log["tp_source"] = f"defaulted_to_{self.default_tp_atr_mult}xATR"
        else:
            tp_dist = abs(float(tp_price) - entry_price)
            if tp_dist < tp_min_dist:
                log["tp_widened"] = (
                    f"from {tp_dist:.6f} to {tp_min_dist:.6f} ({self.tp_atr_mult_min}xATR)"
                )
                tp_dist = tp_min_dist
            elif tp_dist > tp_max_dist:
                log["tp_tightened"] = (
                    f"from {tp_dist:.6f} to {tp_max_dist:.6f} ({self.tp_atr_mult_max}xATR)"
                )
                tp_dist = tp_max_dist

        # --- Enforce R:R bounds by adjusting TP (never widen SL beyond max) ---
        if sl_dist > 0:
            rr = tp_dist / sl_dist
            if rr < self.min_rr:
                new_tp_dist = sl_dist * self.min_rr
                new_tp_dist = min(new_tp_dist, tp_max_dist)
                log["rr_bumped"] = f"rr={rr:.2f} -> min={self.min_rr} (tp {tp_dist:.6f} -> {new_tp_dist:.6f})"
                tp_dist = new_tp_dist
            elif rr > self.max_rr:
                new_tp_dist = sl_dist * self.max_rr
                new_tp_dist = max(new_tp_dist, tp_min_dist)
                log["rr_capped"] = f"rr={rr:.2f} -> max={self.max_rr} (tp {tp_dist:.6f} -> {new_tp_dist:.6f})"
                tp_dist = new_tp_dist

        # --- Convert distances back to prices on correct side of entry ---
        if is_buy:
            final_sl = round(entry_price - sl_dist, 5)
            final_tp = round(entry_price + tp_dist, 5)
        else:
            final_sl = round(entry_price + sl_dist, 5)
            final_tp = round(entry_price - tp_dist, 5)

        return final_tp, final_sl, log

    # ------------------------------------------------------------------
    # Force-close losing positions
    # ------------------------------------------------------------------

    def check_losing_positions(self, positions: list[dict]) -> list[dict]:
        """Return positions that should be force-closed due to excessive loss.

        Args:
            positions: List of position dicts with keys:
                symbol, volume, price_open, profit, contract_size

        Returns:
            List of positions that exceed the max loss threshold.
        """
        to_close = []
        for pos in positions:
            symbol = pos.get("symbol") or pos.get("coin")
            entry_px = float(pos.get("price_open") or pos.get("entryPx") or 0)
            size = float(pos.get("volume") or pos.get("szi") or 0)
            profit = float(pos.get("profit") or pos.get("pnl") or 0)
            contract_size = float(pos.get("contract_size") or 1.0)
            pos_type = pos.get("type", "BUY")  # "BUY" or "SELL"

            if entry_px == 0 or size == 0:
                continue

            notional = abs(size) * entry_px * contract_size
            if notional == 0:
                continue

            loss_pct = abs(profit / notional) * 100 if profit < 0 else 0

            if loss_pct >= self.max_loss_per_position_pct:
                logging.warning(
                    "RISK: Force-closing %s — loss %.2f%% exceeds max %.2f%%",
                    symbol, loss_pct, self.max_loss_per_position_pct
                )
                to_close.append({
                    "symbol": symbol,
                    "ticket": pos.get("ticket"),
                    "volume": abs(size),
                    "is_buy": pos_type == "BUY",
                    "loss_pct": round(loss_pct, 2),
                    "profit": round(profit, 2),
                })
        return to_close

    # ------------------------------------------------------------------
    # Composite validation — run all checks before a trade
    # ------------------------------------------------------------------

    def validate_trade(self, trade: dict, account_state: dict,
                        initial_balance: float) -> tuple[bool, str, dict]:
        """Run all safety checks on a proposed trade.

        Args:
            trade: LLM trade decision with keys:
                asset, action, allocation_usd, tp_price, sl_price
            account_state: Current account with keys:
                balance, total_value (equity), positions
            initial_balance: Starting balance for reserve check

        Returns:
            (allowed, reason, adjusted_trade)
            adjusted_trade may have modified sl_price if it was missing.
        """
        action = trade.get("action", "hold")
        if action == "hold":
            return True, "", trade

        alloc_usd = float(trade.get("allocation_usd", 0))
        if alloc_usd <= 0:
            return False, "Zero or negative allocation", trade

        # MT5 minimum notional guard — broker-dependent; use $10 as floor
        if alloc_usd < 10.0:
            alloc_usd = 10.0
            trade = {**trade, "allocation_usd": alloc_usd}
            logging.info("RISK: Bumped allocation to $10 (minimum notional)")

        account_value = float(account_state.get("total_value", 0) or
                              account_state.get("equity", 0))
        balance = float(account_state.get("balance", 0))
        positions = account_state.get("positions", [])
        is_buy = action == "buy"

        # 1. Daily drawdown circuit breaker
        ok, reason = self.check_daily_drawdown(account_value)
        if not ok:
            return False, reason, trade

        # 2. Balance reserve
        ok, reason = self.check_balance_reserve(balance, initial_balance)
        if not ok:
            return False, reason, trade

        # 3. Position size limit — cap rather than reject
        ok, reason = self.check_position_size(alloc_usd, account_value)
        if not ok:
            max_alloc = account_value * (self.max_position_pct / 100.0)
            max_alloc = max(max_alloc, 10.0)
            logging.warning("RISK: Capping allocation from $%.2f to $%.2f", alloc_usd, max_alloc)
            alloc_usd = max_alloc
            trade = {**trade, "allocation_usd": alloc_usd}

        # 4. Total exposure
        ok, reason = self.check_total_exposure(positions, alloc_usd, account_value)
        if not ok:
            return False, reason, trade

        # 5. Leverage check
        ok, reason = self.check_leverage(alloc_usd, balance)
        if not ok:
            return False, reason, trade

        # 6. Concurrent positions
        active_count = sum(1 for p in positions if abs(float(p.get("volume") or p.get("szi") or 0)) > 0)
        ok, reason = self.check_concurrent_positions(active_count)
        if not ok:
            return False, reason, trade

        # 7. TP/SL direction sanity — the LLM sometimes inverts these for SELL orders
        current_price = float(trade.get("current_price", 0))
        entry_price = current_price if current_price > 0 else 1.0
        sl_price  = trade.get("sl_price")
        tp_price  = trade.get("tp_price")

        if entry_price > 0 and sl_price and tp_price:
            sl_val = float(sl_price)
            tp_val = float(tp_price)
            if is_buy:
                # BUY: SL < entry, TP > entry
                if sl_val > entry_price and tp_val < entry_price:
                    # Both inverted — swap them
                    logging.warning("RISK: BUY TP/SL inverted, swapping sl=%.5f tp=%.5f", sl_val, tp_val)
                    sl_price, tp_price = tp_val, sl_val
                elif sl_val > entry_price:
                    logging.warning("RISK: BUY SL %.5f above entry %.5f — nulling SL", sl_val, entry_price)
                    sl_price = None
                elif tp_val < entry_price:
                    logging.warning("RISK: BUY TP %.5f below entry %.5f — nulling TP", tp_val, entry_price)
                    tp_price = None
            else:
                # SELL: SL > entry, TP < entry
                if sl_val < entry_price and tp_val > entry_price:
                    # Both inverted — swap them
                    logging.warning("RISK: SELL TP/SL inverted, swapping sl=%.5f tp=%.5f", sl_val, tp_val)
                    sl_price, tp_price = tp_val, sl_val
                elif sl_val < entry_price:
                    logging.warning("RISK: SELL SL %.5f below entry %.5f — nulling SL", sl_val, entry_price)
                    sl_price = None
                elif tp_val > entry_price:
                    logging.warning("RISK: SELL TP %.5f above entry %.5f — nulling TP", tp_val, entry_price)
                    tp_price = None

        # 8. ATR-anchored TP/SL — clamp distances to sensible ATR multiples and R:R.
        #    `atr_at_entry` is injected by main.py (5m ATR14 at entry time).
        atr_at_entry = trade.get("atr_at_entry")
        try:
            atr_val = float(atr_at_entry) if atr_at_entry is not None else None
        except (TypeError, ValueError):
            atr_val = None

        sl_float = float(sl_price) if sl_price else None
        tp_float = float(tp_price) if tp_price else None
        final_tp, final_sl, atr_log = self.enforce_atr_bounds(
            entry_price, is_buy, tp_float, sl_float, atr_val
        )
        if atr_log:
            logging.info("RISK: ATR-bounds adjustments %s: %s", trade.get("asset"), atr_log)

        # 9. Minimum stop distance — MT5 rejects stops too close to market price
        #    Use at least 0.1% distance to avoid retcode 10016
        min_dist = entry_price * 0.001
        if is_buy:
            if final_sl > 0 and (entry_price - final_sl) < min_dist:
                final_sl = round(entry_price - min_dist, 5)
                logging.warning("RISK: BUY SL too close, adjusted to %.5f", final_sl)
            if final_tp and float(final_tp) > 0 and (float(final_tp) - entry_price) < min_dist:
                final_tp = round(entry_price + min_dist, 5)
                logging.warning("RISK: BUY TP too close, adjusted to %.5f", final_tp)
        else:
            if final_sl > 0 and (final_sl - entry_price) < min_dist:
                final_sl = round(entry_price + min_dist, 5)
                logging.warning("RISK: SELL SL too close, adjusted to %.5f", final_sl)
            if final_tp and float(final_tp) > 0 and (entry_price - float(final_tp)) < min_dist:
                final_tp = round(entry_price - min_dist, 5)
                logging.warning("RISK: SELL TP too close, adjusted to %.5f", final_tp)

        trade = {**trade, "sl_price": final_sl, "tp_price": final_tp,
                 "atr_adjustments": atr_log}

        return True, "", trade

    # ------------------------------------------------------------------
    # LLM-driven TP/SL adjustment (Phase 4)
    # ------------------------------------------------------------------

    def validate_adjust(
        self,
        symbol: str,
        position: dict,
        new_tp: float | None,
        new_sl: float | None,
        atr: float | None,
    ) -> tuple[bool, str, float | None, float | None]:
        """Validate an LLM 'adjust' action against ATR bounds and direction.

        Returns (allowed, reason, final_tp, final_sl).
        final_tp / final_sl may be None if the LLM asked to leave them unchanged.
        """
        if position is None:
            return False, "No open position for adjust", None, None
        is_buy = str(position.get("type", "BUY")).upper() == "BUY"
        entry = float(position.get("price_open") or 0)
        current_sl = float(position.get("sl") or 0)
        current_tp = float(position.get("tp") or 0)

        if new_tp is None and new_sl is None:
            return False, "adjust requires at least one of tp_price or sl_price", None, None

        # Start from the existing values so callers can update only one side
        tp_for_check = new_tp if new_tp is not None else (current_tp or None)
        sl_for_check = new_sl if new_sl is not None else (current_sl or None)

        # Run through ATR-bounds helper — this also enforces direction (BUY TP>entry etc.)
        # but we need the reference price to be the ORIGINAL entry (not current price)
        # so distances are measured from entry consistently.
        final_tp, final_sl, log = self.enforce_atr_bounds(
            entry, is_buy, tp_for_check, sl_for_check, atr
        )
        if log:
            logging.info("RISK(adjust) %s bounds adjustments: %s", symbol, log)

        # Never ALLOW adjust to loosen the existing stop (moving SL away from price).
        if new_sl is not None and current_sl > 0:
            if is_buy and final_sl < current_sl:
                logging.warning(
                    "RISK(adjust) %s: refusing to loosen BUY SL %.5f -> %.5f",
                    symbol, current_sl, final_sl,
                )
                final_sl = current_sl  # keep existing, tighter stop
            elif (not is_buy) and final_sl > current_sl:
                logging.warning(
                    "RISK(adjust) %s: refusing to loosen SELL SL %.5f -> %.5f",
                    symbol, current_sl, final_sl,
                )
                final_sl = current_sl

        # Return None for fields the caller didn't want changed
        out_tp = final_tp if new_tp is not None else None
        out_sl = final_sl if new_sl is not None else None
        return True, "", out_tp, out_sl

    def get_risk_summary(self) -> dict:
        """Return current risk parameters for inclusion in LLM context."""
        return {
            "max_position_pct": self.max_position_pct,
            "max_loss_per_position_pct": self.max_loss_per_position_pct,
            "max_leverage": self.max_leverage,
            "max_total_exposure_pct": self.max_total_exposure_pct,
            "daily_loss_circuit_breaker_pct": self.daily_loss_circuit_breaker_pct,
            "mandatory_sl_pct": self.mandatory_sl_pct,
            "max_concurrent_positions": self.max_concurrent_positions,
            "min_balance_reserve_pct": self.min_balance_reserve_pct,
            "circuit_breaker_active": self.circuit_breaker_active,
            "tp_atr_mult_range": [self.tp_atr_mult_min, self.tp_atr_mult_max],
            "sl_atr_mult_range": [self.sl_atr_mult_min, self.sl_atr_mult_max],
            "rr_range": [self.min_rr, self.max_rr],
        }
