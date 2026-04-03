import yfinance as yf
import pandas as pd
import ta

# === CONFIGURATION ===
STOCKS = ["AAPL", "AMD", "AMZN", "AVGO", "GOOGL", "META", "MSFT", "NFLX", "NVDA", "PLTR", "QBTS", "TSLA"]
PERIOD = "3mo"
INTERVAL = "1d"

RSI_LOW = 40
RSI_HIGH = 60

def analyze_stock(symbol):
    """Analyze a single stock and return if it’s a good swing trade candidate."""
    try:
        df = yf.download(symbol, period=PERIOD, interval=INTERVAL, progress=False, auto_adjust=False)
        if df.empty:
            print(f"⚠️ No data for {symbol}")
            return None

        # Ensure data is 1D
        close = df["Close"].squeeze()
        high = df["High"].squeeze()
        low = df["Low"].squeeze()

        # Technical Indicators
        df["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()
        df["EMA9"] = ta.trend.EMAIndicator(close, window=9).ema_indicator()
        df["EMA21"] = ta.trend.EMAIndicator(close, window=21).ema_indicator()
        df["ATR"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()

        # Latest values
        latest = df.iloc[-1]
        rsi_value = float(latest["RSI"].iloc[0] if hasattr(latest["RSI"], "iloc") else latest["RSI"])
        ema9 = float(latest["EMA9"].iloc[0] if hasattr(latest["EMA9"], "iloc") else latest["EMA9"])
        ema21 = float(latest["EMA21"].iloc[0] if hasattr(latest["EMA21"], "iloc") else latest["EMA21"])

        # Conditions
        momentum_ok = RSI_LOW <= rsi_value <= RSI_HIGH
        ema_ok = ema9 > ema21

        if momentum_ok and ema_ok:
            print(f"✅ {symbol} looks good → RSI: {rsi_value:.1f}, EMA9>EMA21 - {ema9:.2f}/{ema21:.2f}")
            return symbol
        else:
            print(f"❌ {symbol} skipped → RSI: {rsi_value:.1f}, EMA9/EMA21: {ema9:.2f}/{ema21:.2f}")
            return None

    except Exception as e:
        print(f"Error analyzing {symbol}: {e}")
        return None


if __name__ == "__main__":
    print("🔍 Running swing trading screener...\n")
    candidates = []

    for stock in STOCKS:
        res = analyze_stock(stock)
        if res:
            candidates.append(res)

    print("\n📊 === Summary ===")
    if candidates:
        print(f"🎯 Potential swing trade candidates: {', '.join(candidates)}")
    else:
        print("No strong signals found today.")
