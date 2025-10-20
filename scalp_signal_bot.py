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

    # Convert both timestamps to UTC-aware safely (handle tz-naive and tz-aware cases)
    latest_bar_ts = pd.Timestamp(latest_bar_ts)
    if latest_bar_ts.tz is None:
        latest_bar_ts_utc = latest_bar_ts.tz_localize('UTC')
    else:
        latest_bar_ts_utc = latest_bar_ts.tz_convert('UTC')

    now_ts = pd.Timestamp.utcnow()
    if now_ts.tz is None:
        now_utc = now_ts.tz_localize('UTC')
    else:
        now_utc = now_ts.tz_convert('UTC')

    age_sec = (now_utc - latest_bar_ts_utc).total_seconds()
    if age_sec > 120:
        logger.info("Skipping %s due to stale data (age %.0fs)", symbol, age_sec)
        return None

    sig = construct_signal(symbol, checks, direction, score, passed_checks)
    return sig
