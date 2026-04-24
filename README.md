# Exness MT5 AI Trading Agent — v2

An LLM-driven trading agent for **Exness MetaTrader 5** that analyses candle data, runs indicators locally, and executes short-timeframe (5m default) CFD / spot trades. v2 is a major upgrade over the original port: the core "signal → entry" flow is unchanged, but **exit management** is now handled by deterministic, ATR-anchored server-side logic instead of relying on the LLM to babysit positions.

---

## Why v2 exists

The v1 system had a common failure mode: **trades went into profit but the market flipped before TP was hit**. Root-causing from the trade log showed four compounding issues:

1. TPs were set at round numbers (4700, 1.165, 92) unrelated to symbol volatility, so they were unreachable on a 5-minute loop.
2. No trailing stop or breakeven move — winners routinely reverted to entry or worse.
3. No partial profit-taking — every trade needed to travel the full TP distance to "count".
4. The LLM had no way to modify an existing TP/SL — `buy` / `sell` / `hold` were the only actions.

v2 fixes all four.

---

## What's new in v2

### Phase 1 — ATR-anchored TP / SL enforcement

- `RiskManager.enforce_atr_bounds()` clamps every TP and SL distance to multiples of the **intraday (5m) ATR14** at entry.
- Enforces a configurable reward:risk band (default `1.2 ≤ R:R ≤ 2.5`).
- Intraday ATR14 and ATR3 are now included in the LLM market-data payload — the model is explicitly instructed to size TP/SL as ATR multiples with a worked example.
- Falls back to the legacy percent-based mandatory SL only if ATR is unavailable.

### Phase 2 — Server-side breakeven, trailing stop, TP tightening

- New module `src/position_manager.py` runs every cycle **before** the LLM call.
- For each open bot position, it computes the unrealised **R-multiple** (= (current − entry) / initial SL distance) and applies, in order:
  - **TP tighten** at `+TIGHTEN_TP_R`: pulls TP in to `current ± TIGHTEN_TP_ATR_MULT × ATR` so it becomes reachable.
  - **Breakeven move** at `+BREAKEVEN_ACTIVATE_R`: moves SL to entry ± a small ATR buffer.
  - **Trailing stop** at `+TRAIL_ACTIVATE_R`: trails SL at `TRAIL_ATR_MULT × ATR` behind the current price.
- Only ever tightens SL — never loosens.
- Idempotent — state is reconciled against MT5 each cycle, so a bot restart cold-starts cleanly from the existing SL.

### Phase 3 — Partial profit taking

- At `+PARTIAL_TP_R`, the manager closes `PARTIAL_TP_FRACTION` of the position (default 50% at +1R) and lets the remaining portion ride with breakeven + trailing stop protection.
- Respects each symbol's `volume_step` and `volume_min`.

### Phase 4 — LLM-driven `adjust` action

- New action verb `adjust` in the LLM schema, validated by `RiskManager.validate_adjust()`.
- Lets the LLM modify `tp_price` / `sl_price` on an existing open position without opening a new trade. Refuses to loosen an existing SL.

### Phase 5 — Market-hours filter

- New module `src/market_hours.py` classifies each symbol as crypto / forex / metals / energies / stocks / indices by inspecting `symbol_info.path`, then uses:
  - Broker-reported `trade_mode` (disabled / close-only).
  - Last tick age vs `MAX_TICK_AGE_SEC`.
  - Known UTC session windows per asset class.
- Symbols are auto-held when the market is closed (logged to diary as `market_closed_skip`).
- New entries are blocked within `PRE_CLOSE_BLOCK_SEC` of session close to avoid overnight gap risk (logged as `pre_close_block`). Crypto is 24/7 and unaffected.

---

## Traded Symbols (Exness Zero)

All symbols the bot ships with support by default:

| Symbol   | Asset        | Session                 | Class     |
|----------|--------------|-------------------------|-----------|
| `BTCUSD` | Bitcoin CFD  | 24/7                    | crypto    |
| `ETHUSD` | Ethereum CFD | 24/7                    | crypto    |
| `XAUUSD` | Gold (XAU)   | Sun 22:00 → Fri 22:00   | metals    |
| `EURUSD` | EUR/USD      | Sun 22:00 → Fri 22:00   | forex     |
| `USOIL`  | US Crude Oil | Sun 23:00 → Fri 22:00   | energies  |
| `NFLX`, `MSFT`, `AMD` and other US equities | Mon–Fri 13:30–20:00 UTC | stocks |

### Recommended symbols for this system

The bot's edge comes from (a) **short-timeframe entries** on a 5-minute loop and (b) **ATR-based exit management** that depends on continuous, liquid price action. That makes some asset classes a much better fit than others.

**Best — use these:**
- `BTCUSD` ⭐ — 24/7 liquidity, clean technicals, high ATR but predictable, no gap risk, no session filter needed. The ideal symbol for this design.
- `ETHUSD` ⭐ — same reasons as BTC; slightly more reactive to news but still excellent.
- `XAUUSD` ⭐ — very liquid, cleanly trending, ATR stable within a session. Near 24/5.
- `EURUSD` — most liquid FX pair on the planet, tight spread, predictable intraday ATR. Ideal for the ATR-bounded R:R model.

**Acceptable — use with care:**
- `USOIL` / `UKOIL` — trends well, but news-sensitive (weekly EIA, OPEC headlines) and has daily session breaks. Let the market-hours filter handle it.
- `GBPUSD`, `USDJPY`, `AUDUSD`, `USDCAD` — fine, similar to EURUSD but wider spreads; size allocations down.
- US index CFDs (`US30`, `SPX500`, `NAS100`) — trendy and deep, but have daily breaks and respond violently to US data releases.

**Avoid or disable unless you really want them:**
- **US single stocks** (`NFLX`, `MSFT`, `AMD`, `TSLA`, `NVDA`, etc.):
  - Only ~6.5 tradable hours/day → the bot will spend most of its cycles auto-holding these.
  - **Overnight gap risk** — a stop that made sense at 19:55 UTC can be blown through at next open; the pre-close gate helps but doesn't eliminate this.
  - Earnings events cause moves no ATR-based stop can handle. The system has no earnings-date awareness.
  - Lower volatility per unit time than FX/crypto — 5-minute ATR often tiny relative to spread.
- Low-volume CFDs on exotic FX pairs or minor commodities — spreads eat the edge.
- Anything with `trade_mode` = close-only on your broker.

**Recommended starting config** (copy into `.env`):

```bash
ASSETS="BTCUSD ETHUSD XAUUSD EURUSD"
INTERVAL=5m
```

If you want to add oil, add `USOIL` once you've verified the market-hours filter is behaving correctly.

---

## Safety Guards

All enforced in code, independent of the LLM. Everything is configurable via `.env`.

### Entry / exposure guards

| Guard                    | Default | Description |
|--------------------------|---------|-------------|
| `MAX_POSITION_PCT`       | 10%     | Single position capped at 10% of equity |
| `MAX_LOSS_PER_POSITION_PCT` | 20%  | Auto-close at 20% unrealised loss on notional |
| `MAX_LEVERAGE`           | 10x     | Hard leverage cap |
| `MAX_TOTAL_EXPOSURE_PCT` | 50%     | All positions combined capped at 50% of equity |
| `DAILY_LOSS_CIRCUIT_BREAKER_PCT` | 10% | Stops new trades at 10% daily drawdown |
| `MAX_CONCURRENT_POSITIONS` | 5     | Concurrent position limit |
| `MIN_BALANCE_RESERVE_PCT` | 20%    | Don't trade below 20% of initial balance |

### ATR-anchored TP / SL (Phase 1)

| Knob                 | Default | Description |
|----------------------|---------|-------------|
| `TP_ATR_MULT_MIN`    | 0.8     | Minimum TP distance as multiple of 5m ATR14 |
| `TP_ATR_MULT_MAX`    | 2.0     | Maximum TP distance as multiple of 5m ATR14 |
| `SL_ATR_MULT_MIN`    | 0.8     | Minimum SL distance as multiple of 5m ATR14 |
| `SL_ATR_MULT_MAX`    | 1.5     | Maximum SL distance as multiple of 5m ATR14 |
| `MIN_RR`             | 1.2     | Minimum reward:risk on entry |
| `MAX_RR`             | 2.5     | Maximum reward:risk on entry (keeps TP reachable) |
| `DEFAULT_SL_ATR_MULT`| 1.0     | Fallback SL if LLM returns null |
| `DEFAULT_TP_ATR_MULT`| 1.5     | Fallback TP if LLM returns null |

### Active position management (Phases 2 & 3)

| Knob                    | Default | Description |
|-------------------------|---------|-------------|
| `ENABLE_POSITION_MANAGER` | true  | Master switch |
| `BREAKEVEN_ACTIVATE_R`  | 0.8     | R-multiple at which SL moves to entry |
| `BREAKEVEN_BUFFER_ATR`  | 0.05    | Small ATR buffer above/below entry |
| `TRAIL_ACTIVATE_R`      | 1.5     | R-multiple at which trailing SL engages |
| `TRAIL_ATR_MULT`        | 1.0     | Trailing distance in ATR multiples |
| `TIGHTEN_TP_R`          | 1.0     | R-multiple at which TP is pulled in |
| `TIGHTEN_TP_ATR_MULT`   | 1.0     | New TP distance in ATR multiples from current price |
| `PARTIAL_TP_R`          | 1.0     | R-multiple at which partial close fires |
| `PARTIAL_TP_FRACTION`   | 0.5     | Fraction of position to close at partial TP |

### Market hours filter (Phase 5)

| Knob                   | Default | Description |
|------------------------|---------|-------------|
| `ENABLE_MARKET_HOURS`  | true    | Master switch |
| `MAX_TICK_AGE_SEC`     | 120     | Consider market closed if last tick is older than this |
| `PRE_CLOSE_BLOCK_SEC`  | 900     | Block new entries within N seconds of session close |

---

## Prerequisites

- **Windows PC** with MetaTrader 5 installed and logged into your Exness Zero account
- Python 3.12+
- An LLM API key (default: Groq — free tier works; OpenAI-compatible endpoints supported)

## Setup

```bash
cd mt5-trading-agent-v2

pip install -r requirements.txt

copy .env.example .env
# Edit .env with your MT5 login, password, server, and LLM API key

# Ensure MT5 terminal is running and logged in, then:
python src/main.py
```

Or with CLI args:

```bash
python src/main.py --assets BTCUSD ETHUSD XAUUSD --interval 5m
```

---

## Structure

```
src/
  main.py                  # Entry point, trading loop, API server
  config_loader.py         # Environment config with defaults
  risk_manager.py          # Entry guards + ATR TP/SL enforcement + adjust validator
  position_manager.py      # Server-side breakeven / trail / partial / TP-tighten
  market_hours.py          # Session windows + tick-freshness filter
  agent/
    decision_maker.py      # LLM integration, tool calling, action schema
  indicators/
    local_indicators.py    # EMA, RSI, MACD, ATR, BBands, ADX, OBV, VWAP
  trading/
    mt5_api.py             # MT5 order execution, candles, account state
  utils/
    formatting.py          # Number formatting
    prompt_utils.py        # JSON serialisation helpers
```

## Loop lifecycle (per cycle)

1. Fetch account state (balance, equity, open positions from MT5).
2. Force-close any position at ≥ `MAX_LOSS_PER_POSITION_PCT` unrealised loss.
3. **Market-hours classification** — auto-hold symbols whose session is closed.
4. Gather candles and indicators for evaluated assets (includes intraday ATR).
5. **Run `PositionManager.manage()`** — deterministic breakeven / trail / partial-TP / TP-tighten.
6. Send market context to the LLM (only for active / due-to-poll assets).
7. LLM returns decisions: `buy` / `sell` / `hold` / `adjust`.
8. `RiskManager.validate_trade()` applies ATR-bounded TP/SL and all entry guards; `validate_adjust()` handles `adjust`.
9. Execute approved trades; register new entries with `PositionManager`.
10. Sleep `INTERVAL` and repeat.

## Diary entries (new in v2)

`diary.jsonl` now contains these additional action types:

| Action                 | Source            | Meaning |
|------------------------|-------------------|---------|
| `pm_breakeven`         | PositionManager   | SL moved to breakeven |
| `pm_trail`             | PositionManager   | Trailing stop updated |
| `pm_tighten_tp`        | PositionManager   | TP pulled in to a reachable level |
| `pm_partial_close`     | PositionManager   | Partial TP close executed |
| `adjust`               | LLM / main loop   | LLM-requested TP/SL change applied |
| `adjust_skipped`       | main loop         | LLM asked to adjust a non-existent position |
| `adjust_blocked`       | RiskManager       | `adjust` failed risk validation |
| `market_closed_skip`   | main loop         | Entry skipped because market is closed |
| `pre_close_block`      | main loop         | Entry blocked near session close |

Each entry carries `unrealised_R`, `new_sl`, `new_tp`, and MT5 `retcode` where relevant — so you can audit exactly when and why the bot acted.

## API endpoints

When running, a local REST API is available:

- `GET http://localhost:3000/diary` — Recent trade-diary entries as JSON
- `GET http://localhost:3000/diary?download=1` — Download full diary
- `GET http://localhost:3000/logs` — LLM request logs

## Tuning guidance

- **If you're getting stopped out too often**: widen `SL_ATR_MULT_MIN`/`MAX` or raise `BREAKEVEN_ACTIVATE_R` (e.g. from 0.8 → 1.0) so breakeven doesn't trigger on noise.
- **If winners still reverse before TP**: lower `TIGHTEN_TP_R` (e.g. to 0.7) so TP pulls in earlier. Or raise `PARTIAL_TP_FRACTION` (e.g. to 0.75) to bank more on the first target.
- **If trailing cuts winners too early**: raise `TRAIL_ATR_MULT` from 1.0 to 1.5–2.0.
- **To compare v1 vs v2 behaviour**: set `ENABLE_POSITION_MANAGER=false` and run with `SL_ATR_MULT_MAX=99 TP_ATR_MULT_MAX=99 MIN_RR=0.1` to disable all Phase 1+2+3 clamping.

## Risk disclaimer

Use at your own risk. No guarantee of returns. Trading CFDs involves significant risk of capital loss.
