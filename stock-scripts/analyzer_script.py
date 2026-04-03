#!/usr/bin/env python3
from __future__ import annotations

import csv
import smtplib
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Optional, List, Tuple

import pandas as pd
import yfinance as yf
import ta
import pandas_market_calendars as mcal


# =========================================================
# USER CONFIG
# =========================================================

# Fill these in with your SMTP credentials
SMTP_SERVER = "YOUR_SMTP_SERVER_HOST"
SMTP_PORT = 587
SMTP_USERNAME = "SMTP_USERNAME"
SMTP_PASSWORD = "SMTP_PASSWORD"
EMAIL_FROM = "FROM_EMAIL_ADDRESS"
EMAIL_TO = "TO_EMAIL_ADDRESS"

# Timezones
ET_TZ = "America/New_York"
LOCAL_TZ = "America/Chicago"  # Texas / CDT-CST

# File locations
BASE_DIR = Path("/srv/storage/custom-tools/stock-trading")
REPORT_DIR = BASE_DIR / "reports"
STATE_DIR = BASE_DIR / "state"
DAY_TRADE_LOG = STATE_DIR / "day_trades_c.csv"
LAST_PICKS_LOG = STATE_DIR / "last_picks_c.csv"

# Your watchlist
PRIMARY_WATCHLIST = [
    "GOOGL", "AAL", "AAPL", "AMD", "AMZN", "ARM", "BAC", "AVGO", "CVX", "COP",
    "COST", "CRWD", "DDOG", "DELL", "QBTS", "XOM", "F", "GME", "GE", "INTU",
    "META", "MU", "MSFT", "MRNA", "MDB", "NFLX", "NVDA", "ORCL", "PLTR", "PANW",
    "PFE", "RGTI", "RIVN", "HOOD", "CRM", "NOW", "SHOP", "SNOW", "SBUX", "SMCI",
    "TSM", "TGT", "TSLA", "UBER", "UAL", "VLO", "WMT", "ZS"
]

# A few liquid extras worth watching
EXTRA_WATCHLIST = ["SPY", "QQQ", "ANET", "AMAT", "COIN", "JPM", "LULU", "TSM", "AVGO"]

WATCHLIST = sorted(set(PRIMARY_WATCHLIST + EXTRA_WATCHLIST))

# Output count
MIN_PICKS = 5
MAX_PICKS = 6

# PDT / day-trade guardrail
MAX_DAY_TRADES_PER_5_TRADING_DAYS = 3

# Target
TARGET_PCT = 1.0

# Daily data filters
DAILY_PERIOD = "6mo"
INTRADAY_PERIOD = "5d"
INTRADAY_INTERVAL = "5m"

MIN_PRICE = 8
MAX_PRICE = 1600
MIN_AVG_VOLUME20 = 1_000_000
MIN_AVG_DOLLAR_VOL20 = 25_000_000
MIN_ATR_PCT = 1.0
MAX_ATR_PCT = 12.0

# Long setup preference
LONG_RSI_MIN = 48
LONG_RSI_MAX = 70

# Short setup preference
SHORT_RSI_MIN = 30
SHORT_RSI_MAX = 55

# =========================================================
# DATA MODEL
# =========================================================

@dataclass
class TradeIdea:
    ticker: str
    signal: str          # LONG or SHORT
    setup: str
    score: float
    close: float
    prev_close: float
    daily_change_pct: float
    premarket_gap_pct: float
    premarket_move_pct: float
    rsi14: float
    ema9: float
    ema20: float
    ema50: float
    atr14: float
    atr_pct: float
    avg_vol20: int
    avg_dollar_vol20: float
    prior_20_high: float
    prior_20_low: float
    premarket_high: float
    premarket_low: float
    premarket_last: float
    premarket_volume: int
    entry_hint: str
    stop_hint: str
    target_hint: str
    notes: str


# =========================================================
# FILES / STATE
# =========================================================

def ensure_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    if not DAY_TRADE_LOG.exists():
        with open(DAY_TRADE_LOG, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["trade_date", "ticker", "side", "notes"])

    if not LAST_PICKS_LOG.exists():
        with open(LAST_PICKS_LOG, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["run_date", "ticker"])


# =========================================================
# MARKET CALENDAR
# =========================================================

def nyse_calendar():
    return mcal.get_calendar("NYSE")


def is_trading_day() -> bool:
    cal = nyse_calendar()
    now = pd.Timestamp.now(tz=ET_TZ)
    sched = cal.schedule(
        start_date=now.strftime("%Y-%m-%d"),
        end_date=now.strftime("%Y-%m-%d")
    )
    return not sched.empty


def recent_trading_days(n: int) -> List[str]:
    cal = nyse_calendar()
    end_dt = pd.Timestamp.now(tz=ET_TZ)
    start_dt = end_dt - pd.Timedelta(days=20)
    sched = cal.schedule(
        start_date=start_dt.strftime("%Y-%m-%d"),
        end_date=end_dt.strftime("%Y-%m-%d")
    )
    return [idx.strftime("%Y-%m-%d") for idx in sched.index][-n:]


# =========================================================
# PDT / DAY-TRADE LOGGING
# =========================================================

def load_day_trade_log() -> pd.DataFrame:
    if not DAY_TRADE_LOG.exists():
        return pd.DataFrame(columns=["trade_date", "ticker", "side", "notes"])
    df = pd.read_csv(DAY_TRADE_LOG)
    if df.empty:
        return pd.DataFrame(columns=["trade_date", "ticker", "side", "notes"])
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")
    return df


def count_recent_day_trades() -> Tuple[int, List[str]]:
    df = load_day_trade_log()
    if df.empty:
        days = recent_trading_days(5)
        return 0, days

    last_5 = recent_trading_days(5)
    recent = df[df["trade_date"].isin(last_5)]
    return len(recent), last_5


def append_day_trade_log(trade_date: str, ticker: str, side: str, notes: str = "") -> None:
    with open(DAY_TRADE_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([trade_date, ticker, side.upper(), notes])


# =========================================================
# HELPERS
# =========================================================

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def to_et_index(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.to_datetime(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC").tz_convert(ET_TZ)
    else:
        idx = idx.tz_convert(ET_TZ)
    df = df.copy()
    df.index = idx
    return df


def pct_change(current: float, base: float) -> float:
    if base == 0:
        return 0.0
    return ((current - base) / base) * 100.0


# =========================================================
# DOWNLOADS
# =========================================================

def download_daily(ticker: str) -> Optional[pd.DataFrame]:
    try:
        df = yf.download(
            ticker,
            period=DAILY_PERIOD,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
            prepost=False,
        )
        if df is None or df.empty:
            return None
        df = flatten_columns(df).dropna().copy()
        required = {"Open", "High", "Low", "Close", "Volume"}
        if not required.issubset(df.columns):
            return None
        if len(df) < 60:
            return None
        return df
    except Exception:
        return None


def download_intraday(ticker: str) -> Optional[pd.DataFrame]:
    try:
        df = yf.download(
            ticker,
            period=INTRADAY_PERIOD,
            interval=INTRADAY_INTERVAL,
            auto_adjust=False,
            progress=False,
            threads=False,
            prepost=True,
        )
        if df is None or df.empty:
            return None
        df = flatten_columns(df).dropna().copy()
        required = {"Open", "High", "Low", "Close", "Volume"}
        if not required.issubset(df.columns):
            return None
        df = to_et_index(df)
        return df
    except Exception:
        return None


# =========================================================
# INDICATORS
# =========================================================

def add_daily_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()
    volume = df["Volume"].squeeze()

    df = df.copy()
    df["EMA9"] = ta.trend.EMAIndicator(close=close, window=9).ema_indicator()
    df["EMA20"] = ta.trend.EMAIndicator(close=close, window=20).ema_indicator()
    df["EMA50"] = ta.trend.EMAIndicator(close=close, window=50).ema_indicator()
    df["RSI14"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    df["ATR14"] = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
    df["ATR_PCT"] = (df["ATR14"] / df["Close"]) * 100.0
    df["AVG_VOL20"] = volume.rolling(20).mean()
    df["AVG_DOLLAR_VOL20"] = (close * volume).rolling(20).mean()
    df["PRIOR_20_HIGH"] = df["High"].rolling(20).max().shift(1)
    df["PRIOR_20_LOW"] = df["Low"].rolling(20).min().shift(1)
    return df


def get_today_premarket_slice(df_intraday: pd.DataFrame) -> pd.DataFrame:
    if df_intraday is None or df_intraday.empty:
        return pd.DataFrame()

    now_et = pd.Timestamp.now(tz=ET_TZ)
    today_str = now_et.strftime("%Y-%m-%d")

    today = df_intraday[df_intraday.index.strftime("%Y-%m-%d") == today_str].copy()
    if today.empty:
        return pd.DataFrame()

    # Robinhood early extended-hours start at 7:00 ET;
    # use up to 9:25 ET to avoid the open.
    pre = today.between_time("07:00", "09:25")
    return pre.copy()


# =========================================================
# MARKET REGIME
# =========================================================

def get_market_regime() -> str:
    """
    Uses SPY daily trend + premarket gap as a simple regime filter.
    Returns: bullish / bearish / neutral
    """
    daily = download_daily("SPY")
    intraday = download_intraday("SPY")
    if daily is None:
        return "neutral"

    daily = add_daily_indicators(daily).dropna()
    if daily.empty:
        return "neutral"

    last = daily.iloc[-1]
    close = float(last["Close"])
    ema20 = float(last["EMA20"])
    ema50 = float(last["EMA50"])

    pm_gap = 0.0
    if intraday is not None:
        pre = get_today_premarket_slice(intraday)
        if not pre.empty:
            pm_last = float(pre["Close"].iloc[-1])
            pm_gap = pct_change(pm_last, close)

    if close > ema20 > ema50 and pm_gap >= -0.40:
        return "bullish"
    if close < ema20 < ema50 and pm_gap <= 0.20:
        return "bearish"
    return "neutral"


# =========================================================
# SCORING
# =========================================================

def score_long(
    close: float,
    ema9: float,
    ema20: float,
    ema50: float,
    rsi14: float,
    atr_pct: float,
    prior_20_high: float,
    pm_last: float,
    pm_high: float,
    pm_gap_pct: float,
    pm_move_pct: float,
    pm_volume: int,
) -> Tuple[float, str]:
    score = 0.0
    notes = []

    if close > ema20:
        score += 10
        notes.append("close>ema20")
    if ema9 > ema20:
        score += 8
        notes.append("ema9>ema20")
    if ema20 > ema50:
        score += 10
        notes.append("ema20>ema50")

    if LONG_RSI_MIN <= rsi14 <= LONG_RSI_MAX:
        score += 10
        notes.append("healthy_rsi")

    if MIN_ATR_PCT <= atr_pct <= 6.0:
        score += 10
        notes.append("good_atr")
    elif 6.0 < atr_pct <= MAX_ATR_PCT:
        score += 6
        notes.append("high_atr")

    # Breakout / continuation
    if pm_high >= prior_20_high * 0.998:
        score += 12
        notes.append("near_breakout")
    if pm_last > close:
        score += 7
        notes.append("pm_above_prev_close")
    if pm_gap_pct > 0:
        score += min(6, pm_gap_pct * 2.0)
        notes.append("positive_gap")
    if pm_move_pct > 0:
        score += min(6, pm_move_pct * 1.5)
        notes.append("pm_momentum")
    if pm_volume > 0:
        score += 4
        notes.append("pm_volume")

    # Avoid very extended
    if pm_gap_pct > 4.0:
        score -= 8
        notes.append("too_extended")

    return round(score, 2), ",".join(notes)


def score_short(
    close: float,
    ema9: float,
    ema20: float,
    ema50: float,
    rsi14: float,
    atr_pct: float,
    prior_20_low: float,
    pm_last: float,
    pm_low: float,
    pm_gap_pct: float,
    pm_move_pct: float,
    pm_volume: int,
) -> Tuple[float, str]:
    score = 0.0
    notes = []

    if close < ema20:
        score += 10
        notes.append("close<ema20")
    if ema9 < ema20:
        score += 8
        notes.append("ema9<ema20")
    if ema20 < ema50:
        score += 10
        notes.append("ema20<ema50")

    if SHORT_RSI_MIN <= rsi14 <= SHORT_RSI_MAX:
        score += 10
        notes.append("bearish_rsi")

    if MIN_ATR_PCT <= atr_pct <= 6.0:
        score += 10
        notes.append("good_atr")
    elif 6.0 < atr_pct <= MAX_ATR_PCT:
        score += 6
        notes.append("high_atr")

    if pm_low <= prior_20_low * 1.002:
        score += 12
        notes.append("near_breakdown")
    if pm_last < close:
        score += 7
        notes.append("pm_below_prev_close")
    if pm_gap_pct < 0:
        score += min(6, abs(pm_gap_pct) * 2.0)
        notes.append("negative_gap")
    if pm_move_pct < 0:
        score += min(6, abs(pm_move_pct) * 1.5)
        notes.append("pm_down_momentum")
    if pm_volume > 0:
        score += 4
        notes.append("pm_volume")

    if pm_gap_pct < -5.0:
        score -= 8
        notes.append("too_extended")

    return round(score, 2), ",".join(notes)


# =========================================================
# IDEA GENERATION
# =========================================================

def build_trade_idea(ticker: str, market_regime: str) -> Optional[TradeIdea]:
    daily = download_daily(ticker)
    intraday = download_intraday(ticker)

    if daily is None:
        return None

    daily = add_daily_indicators(daily).dropna()
    if daily.empty:
        return None

    last = daily.iloc[-1]
    prev_close = float(last["Close"])
    ema9 = float(last["EMA9"])
    ema20 = float(last["EMA20"])
    ema50 = float(last["EMA50"])
    rsi14 = float(last["RSI14"])
    atr14 = float(last["ATR14"])
    atr_pct = float(last["ATR_PCT"])
    avg_vol20 = int(last["AVG_VOL20"])
    avg_dollar_vol20 = float(last["AVG_DOLLAR_VOL20"])
    prior_20_high = float(last["PRIOR_20_HIGH"])
    prior_20_low = float(last["PRIOR_20_LOW"])

    if not (MIN_PRICE <= prev_close <= MAX_PRICE):
        return None
    if avg_vol20 < MIN_AVG_VOLUME20:
        return None
    if avg_dollar_vol20 < MIN_AVG_DOLLAR_VOL20:
        return None
    if not (MIN_ATR_PCT <= atr_pct <= MAX_ATR_PCT):
        return None

    # Default to no premarket action if no fresh intraday data
    pm_last = prev_close
    pm_high = prev_close
    pm_low = prev_close
    pm_volume = 0
    pm_gap_pct = 0.0
    pm_move_pct = 0.0

    if intraday is not None:
        pre = get_today_premarket_slice(intraday)
        if not pre.empty:
            pm_last = float(pre["Close"].iloc[-1])
            pm_high = float(pre["High"].max())
            pm_low = float(pre["Low"].min())
            pm_volume = int(pre["Volume"].sum())
            pm_gap_pct = pct_change(pm_last, prev_close)
            first_pm = float(pre["Open"].iloc[0])
            pm_move_pct = pct_change(pm_last, first_pm)

    long_score, long_notes = score_long(
        close=prev_close,
        ema9=ema9,
        ema20=ema20,
        ema50=ema50,
        rsi14=rsi14,
        atr_pct=atr_pct,
        prior_20_high=prior_20_high,
        pm_last=pm_last,
        pm_high=pm_high,
        pm_gap_pct=pm_gap_pct,
        pm_move_pct=pm_move_pct,
        pm_volume=pm_volume,
    )

    short_score, short_notes = score_short(
        close=prev_close,
        ema9=ema9,
        ema20=ema20,
        ema50=ema50,
        rsi14=rsi14,
        atr_pct=atr_pct,
        prior_20_low=prior_20_low,
        pm_last=pm_last,
        pm_low=pm_low,
        pm_gap_pct=pm_gap_pct,
        pm_move_pct=pm_move_pct,
        pm_volume=pm_volume,
    )

    # Regime bias
    if market_regime == "bullish":
        long_score += 6
        short_score -= 4
    elif market_regime == "bearish":
        short_score += 6
        long_score -= 4

    signal = "LONG" if long_score >= short_score else "SHORT"
    score = long_score if signal == "LONG" else short_score
    notes = long_notes if signal == "LONG" else short_notes

    if score < 32:
        return None

    if signal == "LONG":
        setup = "breakout/continuation" if pm_high >= prior_20_high * 0.998 else "pullback-long"
        entry_hint = f"above {max(pm_high, prev_close):.2f}"
        stop_hint = f"below {min(ema9, ema20) - 0.50 * atr14:.2f}"
        target_hint = f"+{TARGET_PCT:.1f}% to +{min(2.0, max(1.0, atr_pct * 0.45)):.1f}%"
    else:
        setup = "breakdown/continuation" if pm_low <= prior_20_low * 1.002 else "fade-short"
        entry_hint = f"below {min(pm_low, prev_close):.2f}"
        stop_hint = f"above {max(ema9, ema20) + 0.50 * atr14:.2f}"
        target_hint = f"-{TARGET_PCT:.1f}% to -{min(2.0, max(1.0, atr_pct * 0.45)):.1f}%"

    return TradeIdea(
        ticker=ticker,
        signal=signal,
        setup=setup,
        score=round(score, 2),
        close=round(pm_last, 2),
        prev_close=round(prev_close, 2),
        daily_change_pct=round(pct_change(pm_last, prev_close), 2),
        premarket_gap_pct=round(pm_gap_pct, 2),
        premarket_move_pct=round(pm_move_pct, 2),
        rsi14=round(rsi14, 2),
        ema9=round(ema9, 2),
        ema20=round(ema20, 2),
        ema50=round(ema50, 2),
        atr14=round(atr14, 2),
        atr_pct=round(atr_pct, 2),
        avg_vol20=avg_vol20,
        avg_dollar_vol20=round(avg_dollar_vol20, 2),
        prior_20_high=round(prior_20_high, 2),
        prior_20_low=round(prior_20_low, 2),
        premarket_high=round(pm_high, 2),
        premarket_low=round(pm_low, 2),
        premarket_last=round(pm_last, 2),
        premarket_volume=pm_volume,
        entry_hint=entry_hint,
        stop_hint=stop_hint,
        target_hint=target_hint,
        notes=notes,
    )


# =========================================================
# PICK SELECTION
# =========================================================

def load_last_picks() -> pd.DataFrame:
    if not LAST_PICKS_LOG.exists():
        return pd.DataFrame(columns=["run_date", "ticker"])
    df = pd.read_csv(LAST_PICKS_LOG)
    if df.empty:
        return pd.DataFrame(columns=["run_date", "ticker"])
    return df


def save_last_picks(picks: List[TradeIdea]) -> None:
    today = pd.Timestamp.now(tz=ET_TZ).strftime("%Y-%m-%d")
    rows = [{"run_date": today, "ticker": p.ticker} for p in picks]
    pd.DataFrame(rows).to_csv(LAST_PICKS_LOG, index=False)


def select_picks(ideas: List[TradeIdea], min_count: int, max_count: int) -> List[TradeIdea]:
    last_df = load_last_picks()
    recent = set(last_df["ticker"].tolist()) if not last_df.empty else set()

    fresh = [x for x in ideas if x.ticker not in recent]
    repeat = [x for x in ideas if x.ticker in recent]

    ordered = sorted(fresh, key=lambda x: x.score, reverse=True) + \
              sorted(repeat, key=lambda x: x.score, reverse=True)

    picks = ordered[:max_count]
    if len(picks) < min_count:
        picks = sorted(ideas, key=lambda x: x.score, reverse=True)[:min_count]

    return picks


# =========================================================
# REPORTING / EMAIL
# =========================================================

def write_report(picks: List[TradeIdea]) -> Path:
    ts = pd.Timestamp.now(tz=ET_TZ).strftime("%Y%m%d_%H%M%S")
    out = REPORT_DIR / f"premarket_watchlist_{ts}_c.csv"
    pd.DataFrame([asdict(x) for x in picks]).to_csv(out, index=False)
    return out


def build_email_html(
    picks: List[TradeIdea],
    market_regime: str,
    day_trade_count: int,
    last_5_days: List[str],
    report_path: Path
) -> str:
    now_local = pd.Timestamp.now(tz=LOCAL_TZ).strftime("%Y-%m-%d %I:%M %p %Z")
    remaining = max(0, MAX_DAY_TRADES_PER_5_TRADING_DAYS - day_trade_count)

    rows = ""
    for i, p in enumerate(picks, start=1):
        color = "#0a7f2e" if p.signal == "LONG" else "#b42318"
        rows += f"""
        <tr>
          <td>{i}</td>
          <td><b>{p.ticker}</b></td>
          <td style="color:{color};"><b>{p.signal}</b></td>
          <td>{p.setup}</td>
          <td>{p.score}</td>
          <td>{p.close}</td>
          <td>{p.premarket_gap_pct}%</td>
          <td>{p.rsi14}</td>
          <td>{p.atr_pct}%</td>
          <td>{p.entry_hint}</td>
          <td>{p.stop_hint}</td>
          <td>{p.target_hint}</td>
        </tr>
        """

    return f"""
    <html>
      <body>
        <h2>Robinhood Premarket Watchlist</h2>
        <p><b>Generated:</b> {now_local}</p>
        <p><b>Market regime:</b> {market_regime.upper()}</p>
        <p><b>Local PDT tracker:</b> {day_trade_count} day trades logged in the last 5 trading days
        ({", ".join(last_5_days)}). <b>Remaining local slots:</b> {remaining}/{MAX_DAY_TRADES_PER_5_TRADING_DAYS}</p>
        <p><b>Goal:</b> ideas with roughly {TARGET_PCT:.1f}% move potential. These are watchlist candidates, not guaranteed trades.</p>

        <table border="1" cellpadding="6" cellspacing="0">
          <tr>
            <th>#</th>
            <th>Ticker</th>
            <th>Signal</th>
            <th>Setup</th>
            <th>Score</th>
            <th>Premarket Last</th>
            <th>PM Gap</th>
            <th>RSI</th>
            <th>ATR%</th>
            <th>Entry Hint</th>
            <th>Stop Hint</th>
            <th>Target Hint</th>
          </tr>
          {rows}
        </table>

        <p>CSV report saved at: {report_path}</p>
        <p>Reminder: if you execute same-day round trips, log them in the local day-trade CSV so the PDT counter stays accurate.</p>
      </body>
    </html>
    """


def send_email(subject: str, html_body: str) -> None:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    msg.set_content("This email contains an HTML watchlist.")
    msg.add_alternative(html_body, subtype="html")

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=30) as server:
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(msg)


# =========================================================
# CLI HELPER FOR MANUAL DAY-TRADE LOGGING
# =========================================================

def maybe_handle_cli_log() -> bool:
    """
    Example:
    python premarket_robinhood_scanner.py --log-daytrade 2026-03-16 NVDA LONG "same-day round trip"
    """
    if len(sys.argv) >= 5 and sys.argv[1] == "--log-daytrade":
        trade_date = sys.argv[2]
        ticker = sys.argv[3]
        side = sys.argv[4]
        notes = sys.argv[5] if len(sys.argv) >= 6 else ""
        append_day_trade_log(trade_date, ticker, side, notes)
        print(f"Logged day trade: {trade_date} {ticker} {side}")
        return True
    return False


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    ensure_dirs()

    if maybe_handle_cli_log():
        return

    if not is_trading_day():
        print("NYSE is closed today. Exiting.")
        return

    market_regime = get_market_regime()
    day_trade_count, last_5_days = count_recent_day_trades()

    ideas: List[TradeIdea] = []
    for ticker in WATCHLIST:
        idea = build_trade_idea(ticker, market_regime)
        if idea is not None:
            ideas.append(idea)

    if not ideas:
        print("No qualifying ideas found today.")
        return

    # Sort by score, then volatility, then premarket volume
    ideas = sorted(
        ideas,
        key=lambda x: (x.score, x.atr_pct, x.premarket_volume),
        reverse=True
    )

    picks = select_picks(ideas, MIN_PICKS, MAX_PICKS)
    if not picks:
        print("No picks selected.")
        return

    report_path = write_report(picks)
    save_last_picks(picks)

    subject = f"Premarket Watchlist - {pd.Timestamp.now(tz=LOCAL_TZ).strftime('%Y-%m-%d')}"
    html = build_email_html(
        picks=picks,
        market_regime=market_regime,
        day_trade_count=day_trade_count,
        last_5_days=last_5_days,
        report_path=report_path
    )
    send_email(subject, html)

    print(f"Sent {len(picks)} ideas to {EMAIL_TO}")
    print(f"Report: {report_path}")
    print(f"Market regime: {market_regime}")
    print(f"Local PDT tracker: {day_trade_count}/{MAX_DAY_TRADES_PER_5_TRADING_DAYS}")


if __name__ == "__main__":
    main()