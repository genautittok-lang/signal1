#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scalp Signal Bot (LONG & SHORT)
- Scans futures USDT pairs via ccxt (Bybit default)
- Computes indicators and a weighted score for LONG and SHORT
- Sends Telegram alerts when score >= CONFIDENCE_THRESHOLD
- Logs signals to CSV and enforces per-symbol cooldown
- Designed for manual execution of trades (no auto-ordering)
"""
import os
import time
import math
import signal
import logging
import csv
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple

import ccxt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

try:
    from telegram import Bot
except Exception:
    Bot = None

load_dotenv()

# ----------------- CONFIG (from ENV) -----------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("scalp_signal_bot")

# Telegram
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

# Exchange and scanning
EXCHANGE_ID = os.environ.get("EXCHANGE_ID", "bybit")  # ccxt exchange id
USE_TESTNET = os.environ.get("USE_TESTNET", "false").lower() in ("1", "true", "yes")
TOP_N_BY_VOLUME = int(os.environ.get("TOP_N_BY_VOLUME", "50"))
SYMBOLS_ENV = os.environ.get("SYMBOLS", "").strip()  # comma-separated to override TOP_N
CHECK_INTERVAL = int(os.environ.get("CHECK_INTERVAL", "30"))  # seconds between cycles
ENTRY_TF = os.environ.get("ENTRY_TF", "1m")
TREND_TF = os.environ.get("TREND_TF", "5m")
OHLCV_LIMIT = int(os.environ.get("OHLCV_LIMIT", "200"))
RUN_ONCE = os.environ.get("RUN_ONCE", "false").lower() in ("1", "true", "yes")

# Risk / position hints
MARGIN_USD = float(os.environ.get("MARGIN_USD", "10"))  # suggested margin per position
LEVERAGE = int(os.environ.get("LEVERAGE", "50"))  # suggested leverage

# Scoring / thresholds
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.80"))  # 0..1
COOLDOWN_MINUTES = int(os.environ.get("COOLDOWN_MINUTES", "15"))
VOL_MULT = float(os.environ.get("VOL_MULT", "1.25"))
ADX_THRESHOLD = float(os.environ.get("ADX_THRESHOLD", "25"))
RSI_IMPULSE = float(os.environ.get("RSI_IMPULSE", "2.5"))
ATR_MIN_PCT = float(os.environ.get("ATR_MIN_PCT", str(0.2/100.0)))
TP_ATR_MULT = float(os.environ.get("TP_ATR_MULT", "3.0"))
SL_ATR_MULT = float(os.environ.get("SL_ATR_MULT", "1.5"))
TP_MIN_PCT = float(os.environ.get("TP_MIN_PCT", str(1.5/100.0)))
SL_MIN_PCT = float(os.environ.get("SL_MIN_PCT", str(0.5/100.0)))

# RSI bounds (for LONG and SHORT)
RSI_LONG_MIN = float(os.environ.get("RSI_LONG_MIN", "45"))
RSI_LONG_MAX = float(os.environ.get("RSI_LONG_MAX", "70"))
RSI_SHORT_MIN = float(os.environ.get("RSI_SHORT_MIN", "30"))
RSI_SHORT_MAX = float(os.environ.get("RSI_SHORT_MAX", "65"))

# files
SIGNAL_LOG_CSV = os.environ.get("SIGNAL_LOG_CSV", "signals_log.csv")

# optional charting (set to "true" to enable, requires matplotlib)
CHARTS_ENABLED = os.environ.get("CHARTS_ENABLED", "false").lower() in ("1", "true", "yes")
if CHARTS_ENABLED:
    import matplotlib.pyplot as plt

# ----------------- TELEGRAM SETUP -----------------
telegram_bot = None
if TELEGRAM_TOKEN and Bot is not None:
    try:
        telegram_bot = Bot(token=TELEGRAM_TOKEN)
        logger.info("Telegram bot initialized.")
    except Exception as e:
        logger.warning("Failed to initialize Telegram bot: %s", e)
else:
    if not TELEGRAM_TOKEN:
        logger.warning("TELEGRAM_TOKEN not set — telegram messages will be disabled.")
    else:
        logger.warning("python-telegram-bot not available — telegram messages disabled.")


def send_telegram(msg_text: str, png_bytes: Optional[bytes] = None):
    if telegram_bot and TELEGRAM_CHAT_ID:
        try:
            if png_bytes:
                # send photo with caption
                telegram_bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=png_bytes, caption=msg_text)
            else:
                telegram_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg_text)
            logger.info("Telegram message sent.")
        except Exception as e:
            logger.exception("Telegram send error: %s", e)
    else:
        logger.info("Telegram disabled or chat id missing. Message would be:\n%s", msg_text)


# ----------------- EXCHANGE SETUP -----------------
def create_exchange():
    args = {"enableRateLimit": True}
    # set defaultType for derivatives where applicable
    args["options"] = {"defaultType": "future"}
    ex_cls = getattr(ccxt, EXCHANGE_ID)
    exchange = ex_cls({"enableRateLimit": True, **args})
    if USE_TESTNET:
        try:
            exchange.set_sandbox_mode(True)
            logger.info("CCXT sandbox mode enabled (testnet).")
        except Exception:
            logger.warning("Exchange sandbox mode not supported or failed.")
    try:
        exchange.load_markets()
    except Exception as e:
        logger.warning("Could not load markets: %s", e)
    return exchange


exchange = create_exchange()


# ----------------- INDICATOR HELPERS -----------------
def ema(series: pd.Series, length: int):
    return series.ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=length - 1, adjust=False).mean()
    ma_down = down.ewm(com=length - 1, adjust=False).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, length: int = 14):
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal_len: int = 9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_len, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger(series: pd.Series, length: int = 20, mult: float = 2.0):
    mid = series.rolling(length).mean()
    std = series.rolling(length).std()
    upper = mid + std * mult
    lower = mid - std * mult
    return mid, upper, lower


# ----------------- DATA FETCH & UTIL -----------------
def normalize_symbol(sym: str) -> str:
    """
    Normalize symbol coming from tickers or env:
    - If symbol contains ':' (e.g. 'BTC/USDT:USDT'), return part before ':' (usually 'BTC/USDT').
    - Ensure symbol uses '/' separator. If symbol is like 'BTCUSDT' try to convert to 'BTC/USDT'.
    """
    s = str(sym).strip()
    # remove exchange suffix after colon (Bybit returns e.g. 'BTC/USDT:USDT')
    if ':' in s:
        s = s.split(':', 1)[0]
    # if there is no '/' but ends with USDT, try to convert 'BTCUSDT' -> 'BTC/USDT'
    if '/' not in s and s.upper().endswith('USDT') and len(s) > 4:
        base = s[:-4]
        s = f"{base}/{s[-4:]}"
    # normalize doubles/slashes and uppercase
    s = s.replace('//', '/').upper()
    return s


def fetch_ohlcv_dataframe(symbol: str, timeframe: str, limit: int = OHLCV_LIMIT) -> pd.DataFrame:
    sym = normalize_symbol(symbol)
    try:
        ohlcv = exchange.fetch_ohlcv(sym, timeframe=timeframe, limit=limit)
    except Exception as e:
        # fallback attempts with safer normalized forms
        logger.debug("Primary fetch failed for %s: %s — trying fallback normalization", sym, e)
        alt = sym
        # if contains repeated suffixes, collapse them
        if alt.count('/USDT') > 1:
            alt = alt.split('/USDT')[0] + '/USDT'
        # try symbol part before colon if original had it
        if ':' in str(symbol):
            alt = normalize_symbol(str(symbol).split(':', 1)[0])
        ohlcv = exchange.fetch_ohlcv(alt, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df


def get_top_symbols_by_volume(n: int = 50) -> List[str]:
    """
    Get top USDT pairs by 24h quote volume from tickers (best-effort).
    Uses ticker['symbol'] when available to avoid suffixes like ':USDT'.
    """
    if SYMBOLS_ENV:
        syms = [normalize_symbol(s) for s in SYMBOLS_ENV.split(",") if s.strip()]
        logger.info("Using SYMBOLS from env (count=%d).", len(syms))
        return syms
    try:
        tickers = exchange.fetch_tickers()
        items = []
        for key, t in tickers.items():
            # prefer ticker's normalized symbol if available
            sym = None
            if isinstance(t, dict):
                # some ccxt adapters put 'symbol' in ticker dict
                sym = t.get("symbol") or key
            else:
                sym = key
            sym = normalize_symbol(sym)
            if "/USDT" not in sym:
                continue
            # try to obtain quoteVolume; fallback to baseVolume
            qv = 0
            if isinstance(t, dict):
                qv = t.get("quoteVolume") or t.get("quoteVolume24h") or t.get("quoteVolume24Hour") or t.get("baseVolume") or 0
            try:
                qv = float(qv or 0)
            except Exception:
                qv = 0.0
            items.append((sym, qv))
        # dedupe by symbol keeping highest volume if duplicates
        seen = {}
        for sym, vol in items:
            if sym not in seen or vol > seen[sym]:
                seen[sym] = vol
        items = sorted(seen.items(), key=lambda x: x[1], reverse=True)
        top = [s for s, _ in items[:n]]
        logger.info("Selected top %d symbols by volume.", len(top))
        return top
    except Exception as e:
        logger.warning("Could not fetch tickers for top symbols: %s. Falling back to default pairs.", e)
        return [
            "BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "XRP/USDT",
            "AVAX/USDT", "DOT/USDT", "LINK/USDT", "MATIC/USDT", "LTC/USDT"
        ]


# ----------------- SCORING LOGIC -----------------
# Weights sum to 1.0 (tweakable via code if needed)
WEIGHTS = {
    "ema_alignment": 0.20,
    "trend": 0.15,
    "macd": 0.15,
    "volume": 0.15,
    "rsi": 0.10,
    "atr": 0.10,
    "candle": 0.05,
    "bollinger": 0.10,
}


def compute_checks(df_entry: pd.DataFrame, df_trend: pd.DataFrame) -> Dict:
    close = df_entry['close']
    ema5 = ema(close, 5)
    ema20 = ema(close, 20)
    ema50 = ema(close, 50)
    ema200 = ema(close, 200)
    rsi_v = rsi(close, 14)
    macd_line, macd_signal, _ = macd(close)
    bb_mid, bb_up, bb_low = bollinger(close, 20, 2)
    vol_ema20 = df_entry['volume'].ewm(span=20, adjust=False).mean()
    atr_v = atr(df_entry, 14)

    last = {
        "close": float(close.iloc[-1]),
        "ema5": float(ema5.iloc[-1]),
        "ema20": float(ema20.iloc[-1]),
        "ema50": float(ema50.iloc[-1]),
        "ema200": float(ema200.iloc[-1]) if len(ema200.dropna()) > 0 else float(ema50.iloc[-1]),
        "rsi": float(rsi_v.iloc[-1]),
        "macd": float(macd_line.iloc[-1]),
        "macd_signal": float(macd_signal.iloc[-1]),
        "bb_up": float(bb_up.iloc[-1]),
        "bb_low": float(bb_low.iloc[-1]),
        "volume": float(df_entry['volume'].iloc[-1]),
        "volEMA20": float(vol_ema20.iloc[-1]),
        "atr": float(atr_v.iloc[-1])
    }

    # trend (higher timeframe)
    close_tr = df_trend['close']
    ema20_t = ema(close_tr, 20).iloc[-1]
    ema50_t = ema(close_tr, 50).iloc[-1]
    trend_long_ok = ema20_t > ema50_t
    trend_short_ok = ema20_t < ema50_t

    # derived
    atr_pct = last['atr'] / last['close'] if last['close'] != 0 else 0
    vol_ok = last['volume'] > last['volEMA20'] * VOL_MULT and last['volEMA20'] > 0
    ema_distance_pct = abs(last['ema20'] - last['ema50']) / last['close'] * 100 if last['close'] != 0 else 0
    adx_ok = ema_distance_pct * 10 > ADX_THRESHOLD

    rsi_recent = rsi_v.iloc[-4:]
    rsi_delta = rsi_recent.iloc[-1] - rsi_recent.iloc[0] if len(rsi_recent) >= 4 else 0
    rsi_impulse_up = rsi_delta >= RSI_IMPULSE
    rsi_impulse_down = rsi_delta <= -RSI_IMPULSE

    prev_close = float(df_entry['close'].iloc[-2])
    candle_green = last['close'] > prev_close
    candle_red = last['close'] < prev_close

    checks = {
        "last": last,
        "trend_long_ok": trend_long_ok,
        "trend_short_ok": trend_short_ok,
        "atr_pct": atr_pct,
        "vol_ok": vol_ok,
        "adx_ok": adx_ok,
        "rsi_impulse_up": rsi_impulse_up,
        "rsi_impulse_down": rsi_impulse_down,
        "candle_green": candle_green,
        "candle_red": candle_red,
        "ema5": ema5,
        "ema20": ema20,
        "ema50": ema50,
        "ema200": ema200,
        "bb_mid": bb_mid,
        "bb_up": bb_up,
        "bb_low": bb_low,
        "macd_line": macd_line,
        "macd_signal": macd_signal,
    }
    return checks


def score_for_direction(checks: Dict, direction: str) -> Tuple[float, Dict[str, bool]]:
    """
    direction: "LONG" or "SHORT"
    returns: (score 0..1, dict of passed checks)
    """
    last = checks['last']
    passed = {}

    # EMA alignment: for LONG: ema5>ema20>ema50 and price>ema20; for SHORT: reverse
    if direction == "LONG":
        ema_align = (last['ema5'] > last['ema20'] > last['ema50'])
        price_above_ema20 = last['close'] > last['ema20']
        ema_alignment_ok = ema_align and price_above_ema20
    else:
        ema_align = (last['ema5'] < last['ema20'] < last['ema50'])
        price_below_ema20 = last['close'] < last['ema20']
        ema_alignment_ok = ema_align and price_below_ema20
    passed['ema_alignment'] = bool(ema_alignment_ok)

    # Trend (higher timeframe)
    passed['trend'] = checks['trend_long_ok'] if direction == "LONG" else checks['trend_short_ok']

    # MACD
    if direction == "LONG":
        passed['macd'] = checks['macd_line'].iloc[-1] > checks['macd_signal'].iloc[-1]
    else:
        passed['macd'] = checks['macd_line'].iloc[-1] < checks['macd_signal'].iloc[-1]

    # Volume spike
    passed['volume'] = bool(checks['vol_ok'])

    # RSI band + impulse
    rsi_val = last['rsi']
    if direction == "LONG":
        rsi_band = (rsi_val >= RSI_LONG_MIN and rsi_val <= RSI_LONG_MAX)
        rsi_impulse = checks['rsi_impulse_up']
    else:
        rsi_band = (rsi_val >= RSI_SHORT_MIN and rsi_val <= RSI_SHORT_MAX)
        rsi_impulse = checks['rsi_impulse_down']
    passed['rsi'] = bool(rsi_band and rsi_impulse)

    # ATR minimal
    passed['atr'] = bool(checks['atr_pct'] >= ATR_MIN_PCT)

    # Candle confirmation
    passed['candle'] = bool(checks['candle_green'] if direction == "LONG" else checks['candle_red'])

    # Bollinger proximity: ensure price not exactly on extreme (avoid catching exhaustion)
    last_close = last['close']
    bb_low = checks['bb_low'].iloc[-1]
    bb_up = checks['bb_up'].iloc[-1]
    if direction == "LONG":
        passed['bollinger'] = last_close > bb_low * 1.001 and last_close < bb_up * 0.999
    else:
        passed['bollinger'] = last_close < bb_up * 0.999 and last_close > bb_low * 1.001

    # compute weighted score
    score = 0.0
    for k, w in WEIGHTS.items():
        val = 1.0 if passed.get(k, False) else 0.0
        score += w * val

    return score, passed


# ----------------- SIGNALS LOGGING & COOLDOWN -----------------
def ensure_log_exists():
    if not os.path.exists(SIGNAL_LOG_CSV):
        with open(SIGNAL_LOG_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_utc", "symbol", "direction", "score", "entry", "tp", "sl", "margin_usd", "leverage", "qty", "uid"])


def append_signal_log(row: Dict):
    ensure_log_exists()
    with open(SIGNAL_LOG_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            row.get("timestamp_utc"),
            row.get("symbol"),
            row.get("direction"),
            f"{row.get('score'):.3f}",
            row.get("entry"),
            row.get("tp"),
            row.get("sl"),
            row.get("margin_usd"),
            row.get("leverage"),
            row.get("qty"),
            row.get("uid")
        ])


def load_last_signal_times() -> Dict[str, datetime]:
    last = {}
    if not os.path.exists(SIGNAL_LOG_CSV):
        return last
    try:
        df = pd.read_csv(SIGNAL_LOG_CSV)
        if df.empty:
            return last
        # use (symbol,direction) -> last timestamp
        for _, r in df.iterrows():
            ts = pd.to_datetime(r['timestamp_utc'])
            key = f"{r['symbol']}|{r['direction']}"
            if key not in last or ts > last[key]:
                last[key] = ts.to_pydatetime()
    except Exception as e:
        logger.warning("Failed to load signal log: %s", e)
    return last


# ----------------- SIGNAL CONSTRUCTION -----------------
def construct_signal(symbol: str, checks: Dict, direction: str, score: float, passed_checks: Dict) -> Dict:
    entry_price = checks['last']['close']
    atr_pct = checks['atr_pct']
    tp_pct_from_atr = atr_pct * TP_ATR_MULT
    sl_pct_from_atr = atr_pct * SL_ATR_MULT
    tp_pct = max(tp_pct_from_atr, 2.0 * sl_pct_from_atr, TP_MIN_PCT)
    sl_pct = max(sl_pct_from_atr, SL_MIN_PCT)

    if direction == "SHORT":
        tp_price = entry_price * (1 - tp_pct)
        sl_price = entry_price * (1 + sl_pct)
    else:
        tp_price = entry_price * (1 + tp_pct)
        sl_price = entry_price * (1 - sl_pct)

    notional = MARGIN_USD * LEVERAGE
    qty = notional / entry_price if entry_price > 0 else 0.0

    uid = f"{symbol}-{direction}-{datetime.utcnow().strftime('%Y%m%dT%H%M')}"
    signal = {
        "symbol": symbol,
        "direction": direction,
        "entry": entry_price,
        "tp": tp_price,
        "sl": sl_price,
        "margin_usd": MARGIN_USD,
        "leverage": LEVERAGE,
        "qty": qty,
        "score": score,
        "passed_checks": passed_checks,
        "uid": uid
    }
    return signal


def format_signal_message(sig: Dict) -> str:
    pc = sig['passed_checks']
    passed_list = [k for k, v in pc.items() if v]
    failed_list = [k for k, v in pc.items() if not v]
    msg = (
        f"⚠️ Scalp SIGNAL — {sig['direction']} {sig['symbol']}\n"
        f"🔸 Price: {sig['entry']:.6f}\n"
        f"🔸 Margin: ${sig['margin_usd']:.2f} | Leverage: {sig['leverage']}x\n"
        f"🔸 Qty (approx): {sig['qty']:.6f}\n"
        f"🎯 TP: {sig['tp']:.6f}\n"
        f"🛑 SL: {sig['sl']:.6f}\n"
        f"📊 Score: {sig['score']*100:.1f}% (threshold {CONFIDENCE_THRESHOLD*100:.0f}%)\n"
        f"✅ Passed: {', '.join(passed_list) if passed_list else 'none'}\n"
        f"❌ Failed: {', '.join(failed_list) if failed_list else 'none'}\n"
        f"🕒 Cooldown: {COOLDOWN_MINUTES} min\n"
        f"🆔 UID: {sig['uid']}\n"
        f"⚠️ This is a signal only — execute manually. Risk is real with leverage."
    )
    return msg


# ----------------- MAIN ANALYZE / LOOP -----------------
def analyze_symbol(symbol: str, last_signal_times: Dict[str, datetime]) -> Optional[Dict]:
    try:
        df_entry = fetch_ohlcv_dataframe(symbol, ENTRY_TF)
        df_trend = fetch_ohlcv_dataframe(symbol, TREND_TF)
    except Exception as e:
        logger.warning("Fetch error %s: %s", symbol, e)
        return None

    if len(df_entry) < 50 or len(df_trend) < 50:
        logger.debug("Not enough data for %s", symbol)
        return None

    checks = compute_checks(df_entry, df_trend)

    # Evaluate both directions
    score_long, passed_long = score_for_direction(checks, "LONG")
    score_short, passed_short = score_for_direction(checks, "SHORT")

    # choose whichever direction exceeds threshold; if both, pick higher score
    candidates = []
    if score_long >= CONFIDENCE_THRESHOLD:
        candidates.append(("LONG", score_long, passed_long))
    if score_short >= CONFIDENCE_THRESHOLD:
        candidates.append(("SHORT", score_short, passed_short))
    if not candidates:
        return None

    # pick best
    candidates.sort(key=lambda x: x[1], reverse=True)
    direction, score, passed_checks = candidates[0]

    # dedupe/cooldown
    key = f"{symbol}|{direction}"
    last_time = last_signal_times.get(key)
    if last_time:
        cutoff = datetime.utcnow() - timedelta(minutes=COOLDOWN_MINUTES)
        if last_time > cutoff:
            logger.info("Skipping %s %s due cooldown (last at %s)", symbol, direction, last_time)
            return None

    # ensure fresh bar: last timestamp shouldn't be older than 120s (user wanted ~2min fresh)
    latest_bar_ts = df_entry.index[-1]
    # make both timestamps timezone-aware in UTC to avoid tz-naive vs tz-aware subtraction error
    latest_bar_ts = pd.Timestamp(latest_bar_ts)
    if latest_bar_ts.tz is None:
        latest_bar_ts_utc = latest_bar_ts.tz_localize('UTC')
    else:
        latest_bar_ts_utc = latest_bar_ts.tz_convert('UTC')
    now_utc = pd.Timestamp.utcnow().tz_localize('UTC')
    age_sec = (now_utc - latest_bar_ts_utc).total_seconds()
    if age_sec > 120:
        logger.info("Skipping %s due to stale data (age %.0fs)", symbol, age_sec)
        return None

    sig = construct_signal(symbol, checks, direction, score, passed_checks)
    return sig


def main_loop():
    ensure_log_exists()
    last_signal_times = load_last_signal_times()
    symbols = get_top_symbols_by_volume(TOP_N_BY_VOLUME)
    logger.info("Monitoring %d symbols. CHECK_INTERVAL=%ds", len(symbols), CHECK_INTERVAL)

    running = True

    def _handle_exit(signum, frame):
        nonlocal running
        logger.info("Received exit signal (%s). Shutting down...", signum)
        running = False

    signal.signal(signal.SIGINT, _handle_exit)
    signal.signal(signal.SIGTERM, _handle_exit)

    while running:
        try:
            # refresh symbols each cycle (to pick new top volumes) only if using TOP_N selection
            if not SYMBOLS_ENV:
                symbols = get_top_symbols_by_volume(TOP_N_BY_VOLUME)

            for sym in symbols:
                if not running:
                    break
                try:
                    sig = analyze_symbol(sym, last_signal_times)
                    if sig:
                        msg = format_signal_message(sig)
                        logger.info("Signal: %s %s score=%.3f", sig['symbol'], sig['direction'], sig['score'])
                        # send telegram
                        send_telegram(msg)
                        # append log and update last_signal_times
                        append_signal_log({
                            "timestamp_utc": datetime.utcnow().isoformat(),
                            **sig
                        })
                        key = f"{sig['symbol']}|{sig['direction']}"
                        last_signal_times[key] = datetime.utcnow()
                except Exception as e:
                    logger.exception("Error analyzing %s: %s", sym, e)
                # light delay between symbol fetches to reduce burst and rate-limit pressure
                time.sleep(0.2)
            if RUN_ONCE:
                logger.info("RUN_ONCE enabled. Exiting after single scan.")
                break
            # sleep loop with early exit support
            for _ in range(max(1, CHECK_INTERVAL)):
                time.sleep(1)
                if not running:
                    break
        except Exception as e:
            logger.exception("Main loop unexpected error: %s", e)
            time.sleep(5)
    logger.info("Bot stopped.")


if __name__ == "__main__":
    main_loop()
