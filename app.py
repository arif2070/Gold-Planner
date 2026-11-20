import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime

st.set_page_config(
    page_title="XAU/USD Technical & Fundamental Planner",
    page_icon="🥇",
    layout="wide"
)

if "journal" not in st.session_state:
    st.session_state.journal = pd.DataFrame(
        columns=[
            "Date & Time",
            "Symbol",
            "Capital ($)",
            "Setup Grade",
            "Score",
            "Bias / Side",
            "Entry",
            "SL",
            "TP1 (1:1.5)",
            "TP2 (1:2)",
            "TP3 (1:3)",
            "Lot Size",
            "Risk ($)",
            "Fundamental / Notes"
        ]
    )


def detect_liquidity_sweep(df: pd.DataFrame, lookback: int = 10) -> tuple[str, str]:
    """Detect Sell-Side (SSL) or Buy-Side (BSL) liquidity sweep over prior swing highs/lows."""
    if len(df) < lookback + 3:
        return "NONE", "Liquidity sweep check karne ke liye candles kam hain."

    prior = df.iloc[-(lookback + 2):-2]
    prev_c = df.iloc[-2]
    curr_c = df.iloc[-1]

    swing_low = float(prior["Low"].min())
    swing_high = float(prior["High"].max())

    ssl_swept = (
        (float(prev_c["Low"]) < swing_low and float(prev_c["Close"]) > swing_low)
        or (float(curr_c["Low"]) < swing_low and float(curr_c["Close"]) > swing_low)
    )
    bsl_swept = (
        (float(prev_c["High"]) > swing_high and float(prev_c["Close"]) < swing_high)
        or (float(curr_c["High"]) > swing_high and float(curr_c["Close"]) < swing_high)
    )

    if ssl_swept and not bsl_swept:
        return "SSL_SWEEP", f"SSL sweep ({swing_low:.2f} ke neeche sweep + wapsi) confirm."
    if bsl_swept and not ssl_swept:
        return "BSL_SWEEP", f"BSL sweep ({swing_high:.2f} ke upar sweep + wapsi) confirm."
    if ssl_swept and bsl_swept:
        return "BOTH_SWEEP", "Dono taraf (BSL & SSL) sweep hua hai — volatile zone."
    return "NONE", f"Koi fresh liquidity sweep nahi mila (High: {swing_high:.2f}, Low: {swing_low:.2f})."


def compute_indicators(
    df: pd.DataFrame,
    ema_fast: int,
    ema_mid: int,
    ema_slow: int,
    ema_trend: int,
    vwap_mode: str,
    vwap_window: int
) -> pd.DataFrame:
    """Calculate customizable EMAs and VWAP."""
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[0] for c in out.columns]

    out = out.dropna(subset=["Close", "High", "Low"]).copy()
    if "Volume" not in out.columns or out["Volume"].fillna(0).sum() == 0:
        out["Volume"] = 1.0

    out["EMA_FAST"] = out["Close"].ewm(span=ema_fast, adjust=False).mean()
    out["EMA_MID"] = out["Close"].ewm(span=ema_mid, adjust=False).mean()
    out["EMA_SLOW"] = out["Close"].ewm(span=ema_slow, adjust=False).mean()
    out["EMA_TREND"] = out["Close"].ewm(span=ema_trend, adjust=False).mean()

    typical_price = (out["High"] + out["Low"] + out["Close"]) / 3.0
    vol = out["Volume"].replace(0, pd.NA).ffill().fillna(1.0)

    if vwap_mode == "Rolling Window":
        pv_roll = (typical_price * vol).rolling(window=vwap_window, min_periods=1).sum()
        v_roll = vol.rolling(window=vwap_window, min_periods=1).sum()
        out["VWAP"] = pv_roll / v_roll
    else:
        out["VWAP"] = (typical_price * vol).cumsum() / vol.cumsum()

    return out


@st.cache_data(ttl=120)
def fetch_mtf_data(
    ticker: str,
    ema_fast: int,
    ema_mid: int,
    ema_slow: int,
    ema_trend: int,
    vwap_mode: str,
    vwap_window: int,
    vwap_strict: bool,
    liq_lookback: int
):
    """Fetch 1D, 4H, 1H, and 15m data and evaluate strict EMA + VWAP + Liquidity status."""
    configs = [
        ("1D", "1y", "1d"),
        ("1H", "60d", "1h"),
        ("15m", "10d", "15m"),
    ]
    raw_frames = {}
    for tf_label, period, interval in configs:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
        if df is not None and not df.empty:
            raw_frames[tf_label] = compute_indicators(
                df, ema_fast, ema_mid, ema_slow, ema_trend, vwap_mode, vwap_window
            )
        else:
            raw_frames[tf_label] = pd.DataFrame()

    df_4h = pd.DataFrame()
    if not raw_frames["1H"].empty:
        h1 = raw_frames["1H"].copy()
        if not isinstance(h1.index, pd.DatetimeIndex):
            h1.index = pd.to_datetime(h1.index)
        agg = h1.resample("4h").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        }).dropna(subset=["Close"])
        if not agg.empty:
            df_4h = compute_indicators(
                agg, ema_fast, ema_mid, ema_slow, ema_trend, vwap_mode, vwap_window
            )
    raw_frames["4H"] = df_4h

    rows = []
    for tf in ["1D", "4H", "1H", "15m"]:
        d = raw_frames.get(tf, pd.DataFrame())
        if d.empty or len(d) < max(ema_fast, ema_mid, ema_slow) + 2:
            rows.append({
                "Timeframe": tf,
                "Close": None,
                f"EMA {ema_fast}": None,
                f"EMA {ema_mid}": None,
                f"EMA {ema_slow}": None,
                "VWAP": None,
                "Liquidity": "NO DATA",
                "Side": "NO TRADE",
                "Reason": "Data available nahi hai ya candles kam hain."
            })
            continue

        last = d.iloc[-1]
        close_p = float(last["Close"])
        e_fast = float(last["EMA_FAST"])
        e_mid = float(last["EMA_MID"])
        e_slow = float(last["EMA_SLOW"])
        e_trend = float(last["EMA_TREND"]) if len(d) >= ema_trend else e_slow
        vwap_p = float(last["VWAP"])

        liq_code, liq_msg = detect_liquidity_sweep(d, lookback=liq_lookback)

        if vwap_strict:
            vwap_bull = (close_p > vwap_p) and (e_fast > vwap_p)
            vwap_bear = (close_p < vwap_p) and (e_fast < vwap_p)
        else:
            vwap_bull = close_p > vwap_p
            vwap_bear = close_p < vwap_p

        ema_stack_bull = (e_fast > e_mid > e_slow) and (close_p > e_fast) and (close_p >= e_trend)
        ema_stack_bear = (e_fast < e_mid < e_slow) and (close_p < e_fast) and (close_p <= e_trend)

        if vwap_bull and ema_stack_bull:
            side = "BUY SIDE"
            reason = f"Strict bullish: Price ({close_p:.2f}) > EMA {ema_fast} > EMA {ema_mid} > EMA {ema_slow} aur VWAP ({vwap_p:.2f})."
        elif vwap_bear and ema_stack_bear:
            side = "SELL SIDE"
            reason = f"Strict bearish: Price ({close_p:.2f}) < EMA {ema_fast} < EMA {ema_mid} < EMA {ema_slow} aur VWAP ({vwap_p:.2f})."
        else:
            side = "NO TRADE"
            reason = (
                f"Strict alignment incomplete — Price: {close_p:.2f}, "
                f"EMA({ema_fast}/{ema_mid}/{ema_slow}): {e_fast:.2f}/{e_mid:.2f}/{e_slow:.2f}, "
                f"VWAP: {vwap_p:.2f}."
            )

        rows.append({
            "Timeframe": tf,
            "Close": round(close_p, 2),
            f"EMA {ema_fast}": round(e_fast, 2),
            f"EMA {ema_mid}": round(e_mid, 2),
            f"EMA {ema_slow}": round(e_slow, 2),
            "VWAP": round(vwap_p, 2),
            "Liquidity": f"{liq_code} — {liq_msg}",
            "Side": side,
            "Reason": reason
        })

    return rows, raw_frames


@st.cache_data(ttl=180)
def fetch_fundamental_macro():
    """Fetch DXY (DX-Y.NYB) and US 10Y Treasury Yield (^TNX) bias for Gold fundamental context."""
    macro = {}
    symbols = {
        "DXY": "DX-Y.NYB",
        "US10Y": "^TNX"
    }
    for key, sym in symbols.items():
        df = yf.download(sym, period="1mo", interval="1d", progress=False, auto_adjust=False)
        if df is not None and not df.empty and len(df) >= 10:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            close_s = df["Close"].dropna()
            last_val = float(close_s.iloc[-1])
            ema10 = float(close_s.ewm(span=10, adjust=False).mean().iloc[-1])
            prev_val = float(close_s.iloc[-2])
            chg_pct = ((last_val - prev_val) / prev_val * 100.0) if prev_val else 0.0

            if last_val > ema10 and chg_pct >= 0:
                bias = "BULLISH"
            elif last_val < ema10 and chg_pct <= 0:
                bias = "BEARISH"
            else:
                bias = "NEUTRAL"

            macro[key] = {
                "value": round(last_val, 2),
                "ema10": round(ema10, 2),
                "chg_pct": round(chg_pct, 2),
                "bias": bias
            }
        else:
            macro[key] = None
    return macro


st.title("🥇 XAU/USD Technical + Fundamental A+ Planner")
st.caption("1D, 4H, 1H, 15m (EMA + VWAP + Liquidity) + Live DXY / US10Y / News Filter + SL/TP Distance Guard.")

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.header("💼 Manual Capital & Risk")
    cap_preset = st.radio("Capital Select Karein ($ / USDT)", ["$100", "$500", "Custom"], horizontal=True)
    if cap_preset == "$100":
        capital = 100.0
    elif cap_preset == "$500":
        capital = 500.0
    else:
        capital = st.number_input("Apna Capital ($ / USDT)", min_value=10.0, value=250.0, step=10.0)

    ticker_choice = st.selectbox("Gold Feed", ["GC=F (Gold Futures)", "XAUUSD=X (Spot Gold)"])
    ticker_sym = ticker_choice.split(" ")[0]

    risk_pct = st.slider("Risk Per Trade (%)", min_value=0.5, max_value=3.0, value=1.0, step=0.5)
    today_pnl = st.number_input("Aaj ka Current P&L ($)", value=0.0, step=1.0)

    min_target_usd = capital * 0.05
    max_target_usd = capital * 0.07
    max_dd_usd = capital * 0.03
    dd_locked = today_pnl <= -max_dd_usd

    st.metric("Daily Target (5%–7%)", f"${min_target_usd:.2f} – ${max_target_usd:.2f}")
    st.metric("Max Daily Drawdown (3%)", f"-${max_dd_usd:.2f}", delta_color="inverse")
    if dd_locked:
        st.error("🛑 3% Daily Drawdown hit! Aaj trading band rakhein.")
    elif today_pnl >= min_target_usd:
        st.success("🏆 5%–7% Daily Target achieve ho chuka hai!")

    st.divider()
    st.subheader("🌍 Fundamental & News Rules")
    news_status = st.selectbox(
        "High-Impact USD News Status",
        [
            "Safe — Koi bari USD news qareeb nahi",
            "Caution — News 30 mins mein aane wali hai",
            "Active Spike — News abhi aayi hai"
        ]
    )
    block_on_news = st.checkbox("News aane se 30 min pehle trade block karein", value=True)
    require_fund_align = st.checkbox("Fundamental (DXY/Yields) confirmation zaroori karein", value=False)

    st.divider()
    st.subheader("🛡️ SL / TP & A+ Score Filters")
    min_sl_usd = st.number_input("Minimum SL Distance ($)", min_value=0.5, max_value=50.0, value=2.0, step=0.5)
    max_sl_usd = st.number_input("Maximum SL Distance ($)", min_value=2.0, max_value=100.0, value=15.0, step=1.0)
    min_tp_usd = st.number_input("Minimum TP Distance ($)", min_value=1.0, max_value=150.0, value=4.0, step=0.5)
    min_rr_required = st.slider("Minimum Required R:R", min_value=1.5, max_value=4.0, value=2.0, step=0.5)
    min_aplus_score = st.slider("Minimum Score for Approval", min_value=70, max_value=100, value=85, step=5)

    st.divider()
    st.subheader("⚙️ Custom EMA & VWAP Rules")
    c_e1, c_e2 = st.columns(2)
    with c_e1:
        ema_fast = st.number_input("Fast EMA", min_value=2, max_value=100, value=9, step=1)
        ema_slow = st.number_input("Slow EMA", min_value=10, max_value=300, value=50, step=1)
    with c_e2:
        ema_mid = st.number_input("Mid EMA", min_value=5, max_value=200, value=21, step=1)
        ema_trend = st.number_input("Trend EMA", min_value=20, max_value=500, value=200, step=5)

    vwap_mode = st.selectbox("VWAP Mode", ["Cumulative", "Rolling Window"])
    vwap_window = st.number_input("Rolling VWAP Candles", min_value=5, max_value=200, value=24, step=1)
    vwap_strict = st.checkbox("Strict VWAP Rule", value=True)
    liq_lookback = st.slider("Liquidity Lookback Candles", min_value=5, max_value=30, value=10, step=1)
    require_liq = st.checkbox("Liquidity Sweep Compulsory Karein", value=True)
    require_all_htf = st.checkbox("1D aur 4H Donon Same Side Zaroori", value=True)
    contract_size = st.number_input("Contract Size (oz / Lot)", min_value=1, value=100, step=10)

    if st.button("🔄 Refresh Technical & Fundamental"):
        fetch_mtf_data.clear()
        fetch_fundamental_macro.clear()
        st.rerun()

# ==========================================
# TECHNICAL & FUNDAMENTAL DATA
# ==========================================
mtf_rows, frames = fetch_mtf_data(
    ticker_sym,
    int(ema_fast),
    int(ema_mid),
    int(ema_slow),
    int(ema_trend),
    vwap_mode,
    int(vwap_window),
    vwap_strict,
    int(liq_lookback)
)

macro_data = fetch_fundamental_macro()

st.subheader("📊 Technical Analysis: 1D, 4H, 1H & 15m (EMA + VWAP + Liquidity)")
st.dataframe(pd.DataFrame(mtf_rows), use_container_width=True, hide_index=True)

st.subheader("🌍 Fundamental Analysis: Dollar Index (DXY) & US 10Y Yields")
f1, f2, f3 = st.columns(3)

dxy_info = macro_data.get("DXY")
us10y_info = macro_data.get("US10Y")

if dxy_info:
    gold_dxy_implication = "Gold SELL ko support" if dxy_info["bias"] == "BULLISH" else ("Gold BUY ko support" if dxy_info["bias"] == "BEARISH" else "Neutral impact")
    f1.metric("DXY (Dollar Index)", f"{dxy_info['value']}", f"{dxy_info['chg_pct']}% | {dxy_info['bias']}")
    f1.caption(f"EMA10: {dxy_info['ema10']} -> {gold_dxy_implication}")
else:
    f1.metric("DXY (Dollar Index)", "N/A")
    f1.caption("DXY feed abhi available nahi.")

if us10y_info:
    gold_y_implication = "Gold SELL ko support" if us10y_info["bias"] == "BULLISH" else ("Gold BUY ko support" if us10y_info["bias"] == "BEARISH" else "Neutral impact")
    f2.metric("US 10Y Yield (^TNX)", f"{us10y_info['value']}%", f"{us10y_info['chg_pct']}% | {us10y_info['bias']}")
    f2.caption(f"EMA10: {us10y_info['ema10']}% -> {gold_y_implication}")
else:
    f2.metric("US 10Y Yield (^TNX)", "N/A")
    f2.caption("Yields feed abhi available nahi.")

f3.metric("USD News Filter", news_status.split(" — ")[0])
f3.caption("CPI / NFP / FOMC ke waqt spikes se bachein.")

# ==========================================
# ENTRY, SL/TP & POSITION SIZING
# ==========================================
st.divider()
st.subheader(f"📐 Entry, SL/TP & Position Sizing (Capital: ${capital:.2f} | Risk: {risk_pct}%)")

sides = {r["Timeframe"]: r["Side"] for r in mtf_rows}
liqs = {r["Timeframe"]: r["Liquidity"] for r in mtf_rows}

d1_buy, d1_sell = sides.get("1D") == "BUY SIDE", sides.get("1D") == "SELL SIDE"
h4_buy, h4_sell = sides.get("4H") == "BUY SIDE", sides.get("4H") == "SELL SIDE"
h1_buy, h1_sell = sides.get("1H") == "BUY SIDE", sides.get("1H") == "SELL SIDE"
m15_buy, m15_sell = sides.get("15m") == "BUY SIDE", sides.get("15m") == "SELL SIDE"

auto_bias_hint = "SELL" if (h1_sell or m15_sell) and not (h1_buy or m15_buy) else "BUY"

latest_15m = frames["15m"].iloc[-1] if not frames.get("15m", pd.DataFrame()).empty else None
def_entry = float(latest_15m["Close"]) if latest_15m is not None else 2650.00
def_sl = (def_entry - 3.00) if auto_bias_hint == "BUY" else (def_entry + 3.00)

p1, p2, p3 = st.columns(3)
with p1:
    trade_side = st.radio("Proposed Side", ["BUY", "SELL"], index=0 if auto_bias_hint == "BUY" else 1, horizontal=True)
    entry_price = st.number_input("Entry Price ($)", min_value=1.0, value=round(def_entry, 2), step=0.50, format="%.2f")
    sl_price = st.number_input("Stop Loss Price ($)", min_value=1.0, value=round(def_sl, 2), step=0.50, format="%.2f")
    custom_tp = st.number_input(
        "Planned Target TP ($)",
        min_value=1.0,
        value=round(def_entry + (6.0 if auto_bias_hint == "BUY" else -6.0), 2),
        step=0.50,
        format="%.2f"
    )

sl_dist = abs(entry_price - sl_price)
tp_dist = abs(custom_tp - entry_price)
actual_rr = (tp_dist / sl_dist) if sl_dist > 0 else 0.0

risk_usd = capital * (risk_pct / 100.0)
lot_size = round(max(0.01, risk_usd / (sl_dist * contract_size)), 2) if sl_dist > 0 else 0.0

direction_mult = 1.0 if trade_side == "BUY" else -1.0
tp1 = entry_price + direction_mult * sl_dist * 1.5
tp2 = entry_price + direction_mult * sl_dist * 2.0
tp3 = entry_price + direction_mult * sl_dist * 3.0

with p2:
    st.metric("Calculated Lot Size", f"{lot_size:.2f} Lots")
    st.metric("Risk Amount ($)", f"${risk_usd:.2f}")
    st.caption(f"SL Distance: ${sl_dist:.2f} ({sl_dist*10:.1f} pips) | Planned R:R = 1:{actual_rr:.2f}")

with p3:
    st.metric("TP1 (1:1.5)", f"${tp1:.2f}", f"+${risk_usd*1.5:.2f}")
    st.metric("TP2 (1:2.0)", f"${tp2:.2f}", f"+${risk_usd*2.0:.2f}")
    st.metric("TP3 (1:3.0)", f"${tp3:.2f}", f"+${risk_usd*3.0:.2f}")

# ==========================================
# COMBINED TECHNICAL + FUNDAMENTAL A+ SCORING
# ==========================================
score = 0
pass_reasons = []
fail_reasons = []
is_buy = trade_side == "BUY"

if dd_locked:
    fail_reasons.append(f"3% Daily Drawdown (-${max_dd_usd:.2f}) hit ho chuka hai.")

news_blocked = block_on_news and ("30 mins" in news_status)
if news_blocked:
    fail_reasons.append("High-impact USD news 30 minute ke andar aane wali hai — trade blocked.")

# 1. HTF Alignment (25 pts)
htf_aligned = (
    ((d1_buy and h4_buy) if require_all_htf else (d1_buy or h4_buy))
    if is_buy
    else ((d1_sell and h4_sell) if require_all_htf else (d1_sell or h4_sell))
)
if htf_aligned:
    score += 25
    pass_reasons.append(f"[+25 pts] Technical HTF (1D/4H) {trade_side} side ke saath aligned hai.")
else:
    fail_reasons.append(f"[0/25 pts] Technical HTF (1D/4H) par {trade_side} alignment poori nahi hai.")

# 2. 1H + 15m Alignment (30 pts: 15 each)
if (h1_buy if is_buy else h1_sell):
    score += 15
    pass_reasons.append(f"[+15 pts] 1H timeframe {trade_side} side confirm kar raha hai.")
else:
    fail_reasons.append(f"[0/15 pts] 1H timeframe par {trade_side} confirmation missing hai.")

m15_aligned = m15_buy if is_buy else m15_sell
if m15_aligned:
    score += 15
    pass_reasons.append(f"[+15 pts] 15m execution chart {trade_side} side par aligned hai.")
else:
    fail_reasons.append(f"[0/15 pts] 15m execution chart abhi {trade_side} trigger nahi de raha.")

# 3. Liquidity Sweep (15 pts)
liq_aligned = (
    any("SSL_SWEEP" in liqs.get(tf, "") for tf in ["1H", "15m"])
    if is_buy
    else any("BSL_SWEEP" in liqs.get(tf, "") for tf in ["1H", "15m"])
)
if liq_aligned:
    score += 15
    pass_reasons.append(f"[+15 pts] {'SSL' if is_buy else 'BSL'} Liquidity Sweep confirm hua hai.")
else:
    fail_reasons.append(f"[0/15 pts] {'SSL' if is_buy else 'BSL'} Liquidity Sweep abhi detect nahi hua.")

# 4. Fundamental DXY & US10Y Alignment (15 pts)
fund_bull_gold = (
    (dxy_info and dxy_info["bias"] == "BEARISH")
    or (us10y_info and us10y_info["bias"] == "BEARISH")
)
fund_bear_gold = (
    (dxy_info and dxy_info["bias"] == "BULLISH")
    or (us10y_info and us10y_info["bias"] == "BULLISH")
)
fund_aligned = fund_bull_gold if is_buy else fund_bear_gold

if fund_aligned:
    score += 15
    pass_reasons.append(f"[+15 pts] Fundamental macro (DXY / US10Y) {trade_side} Gold ko support kar raha hai.")
else:
    fail_reasons.append(f"[0/15 pts] Fundamental macro (DXY / US10Y) abhi {trade_side} Gold ko clear support nahi de raha.")

# 5. SL/TP Distance & R:R Filter (15 pts)
sl_side_valid = (sl_price < entry_price) if is_buy else (sl_price > entry_price)
tp_side_valid = (custom_tp > entry_price) if is_buy else (custom_tp < entry_price)
sl_tp_ok = (
    sl_side_valid
    and tp_side_valid
    and (min_sl_usd <= sl_dist <= max_sl_usd)
    and (tp_dist >= min_tp_usd)
    and (actual_rr >= min_rr_required)
)

if sl_tp_ok:
    score += 15
    pass_reasons.append(f"[+15 pts] SL (${sl_dist:.2f}), TP (${tp_dist:.2f}) aur R:R (1:{actual_rr:.2f}) rules pass hain.")
else:
    fail_reasons.append(f"[0/15 pts] SL/TP distance ya R:R filter fail hua (SL: ${sl_dist:.2f}, TP: ${tp_dist:.2f}, R:R: 1:{actual_rr:.2f}).")

if score >= 90:
    setup_grade = "A+ SETUP"
elif score >= 75:
    setup_grade = "A SETUP"
elif score >= 60:
    setup_grade = "B SETUP"
else:
    setup_grade = "C / NO-TRADE"

hard_block = (
    dd_locked
    or news_blocked
    or not sl_tp_ok
    or not htf_aligned
    or not m15_aligned
    or (require_liq and not liq_aligned)
    or (require_fund_align and not fund_aligned)
    or score < min_aplus_score
)

final_signal = f"{trade_side} SIDE" if not hard_block else "NO TRADE"

st.divider()
st.subheader("🏆 Technical + Fundamental A+ Score & Decision")

m1, m2, m3 = st.columns(3)
m1.metric("Combined Score", f"{score} / 100", f"Min Required: {min_aplus_score}")
m2.metric("Setup Grade", setup_grade)
m3.metric("Final Verdict", final_signal)

if final_signal != "NO TRADE":
    st.success(f"### ✅ {setup_grade} ({score}/100): {final_signal} TRADE APPROVED")
else:
    st.error(f"### ⛔ {setup_grade} ({score}/100): STRICT NO TRADE")

c_pass, c_fail = st.columns(2)
with c_pass:
    st.markdown("#### ✔️ Passed Confirmations:")
    if pass_reasons:
        for r in pass_reasons:
            st.markdown(f"- {r}")
    else:
        st.info("Abhi koi confirmation pass nahi hui.")

with c_fail:
    st.markdown("#### ❌ Missing / Blocking Factors:")
    if fail_reasons:
        for r in fail_reasons:
            st.markdown(f"- {r}")
    else:
        st.success("Koi rukawat nahi hai.")

# ==========================================
# TRADE JOURNAL & CSV EXPORT
# ==========================================
st.divider()
st.subheader("📒 Trade Journal & CSV Export")
dxy_txt = f"DXY:{dxy_info['bias']}({dxy_info['value']})" if dxy_info else "DXY:N/A"
y10_txt = f"10Y:{us10y_info['bias']}({us10y_info['value']}%)" if us10y_info else "10Y:N/A"
notes_val = st.text_input("Notes", value=f"{setup_grade} ({score}/100) | {dxy_txt} | {y10_txt} | {news_status}")

if st.button("➕ Save Setup to Journal", type="primary"):
    new_row = pd.DataFrame([{
        "Date & Time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Symbol": ticker_sym,
        "Capital ($)": capital,
        "Setup Grade": setup_grade,
        "Score": f"{score}/100",
        "Bias / Side": trade_side,
        "Entry": round(entry_price, 2),
        "SL": round(sl_price, 2),
        "TP1 (1:1.5)": round(tp1, 2),
        "TP2 (1:2)": round(tp2, 2),
        "TP3 (1:3)": round(tp3, 2),
        "Lot Size": lot_size,
        "Risk ($)": round(risk_usd, 2),
        "Fundamental / Notes": notes_val
    }])
    st.session_state.journal = pd.concat([st.session_state.journal, new_row], ignore_index=True)
    st.success("Journal mein save ho gaya!")

if not st.session_state.journal.empty:
    st.dataframe(st.session_state.journal, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download Journal CSV",
        data=st.session_state.journal.to_csv(index=False).encode("utf-8"),
        file_name=f"xauusd_tech_fund_journal_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
