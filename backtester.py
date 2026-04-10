import os
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

try:
    from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
    from alpaca.data.timeframe import TimeFrameUnit, TimeFrame
    from alpaca.data.enums import Adjustment
    ALPACA_AVAILABLE = True
except Exception:
    ALPACA_AVAILABLE = False


def get_data(ticker, years=5, interval="1Day"):
    """Fetch historical data from Alpaca or raise if the package/config is unavailable."""
    if not ALPACA_AVAILABLE:
        raise RuntimeError(
            "Alpaca SDK is not installed or unavailable in this environment.")

    API_KEY = os.environ.get('APCA_API_KEY_ID', '')
    SECRET_KEY = os.environ.get('APCA_API_SECRET_KEY', '')

    if not API_KEY or not SECRET_KEY:
        raise RuntimeError(
            "Alpaca API keys are not configured in environment variables.")

    tf_map = {
        "1Min": TimeFrame(1, TimeFrameUnit.Minute),
        "15Min": TimeFrame(15, TimeFrameUnit.Minute),
        "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
        "4Hour": TimeFrame(4, TimeFrameUnit.Hour),
        "1Day": TimeFrame(1, TimeFrameUnit.Day)
    }

    start_date = datetime(datetime.now().year - years, 1, 1)
    timeframe = tf_map.get(interval, TimeFrame(1, TimeFrameUnit.Day))
    is_crypto = '/' in ticker

    print(
        f"Fetching {'crypto' if is_crypto else 'stock'} data for {ticker} from {start_date.date()}...")

    if is_crypto:
        client = CryptoHistoricalDataClient(API_KEY, SECRET_KEY)
        request_params = CryptoBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=timeframe,
            start=start_date
        )
        bars = client.get_crypto_bars(request_params)
        df = bars.df
        if df.empty:
            raise ValueError(f"No crypto data found for {ticker}")
        df = df.xs(ticker)
    else:
        client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
        request_params = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=timeframe,
            start=start_date,
            adjustment=Adjustment.ALL
        )
        bars = client.get_stock_bars(request_params)
        df = bars.df
        if df.empty:
            raise ValueError(f"No stock data found for {ticker}")
        df = df.xs(ticker)

    df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)
    df = df.sort_index()

    print(f"Successfully fetched {len(df)} bars for {ticker}")
    return df


def generate_sample_data(periods=200, start_price=100.0):
    dates = pd.date_range(end=pd.Timestamp.today(), periods=periods, freq='B')
    returns = np.random.normal(loc=0.0005, scale=0.02, size=periods)
    close = start_price * np.exp(np.cumsum(returns))
    open_ = close * (1 + np.random.normal(0, 0.005, size=periods))
    high = np.maximum(close, open_) * \
        (1 + np.abs(np.random.normal(0, 0.01, size=periods)))
    low = np.minimum(close, open_) * \
        (1 - np.abs(np.random.normal(0, 0.01, size=periods)))
    volume = np.random.randint(1000, 10000, size=periods)

    df = pd.DataFrame({
        'Open': open_,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    }, index=dates)
    return df


def sma_crossover_strategy(df):
    df['SMA9'] = df['Close'].rolling(9).mean()
    df['SMA21'] = df['Close'].rolling(21).mean()
    df['Signal'] = 0
    df.loc[df['SMA9'] > df['SMA21'], 'Signal'] = 1
    df.loc[df['SMA9'] < df['SMA21'], 'Signal'] = -1
    return df


def rsi_strategy(df, overbought=70, oversold=30):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['Signal'] = 0
    df.loc[df['RSI'] < oversold, 'Signal'] = 1
    df.loc[df['RSI'] > overbought, 'Signal'] = -1
    return df


def bollinger_strategy(df, window=20, std_dev=2):
    df['BB_Mid'] = df['Close'].rolling(window=window).mean()
    df['BB_Std'] = df['Close'].rolling(window=window).std()
    df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * std_dev)
    df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * std_dev)
    df['Signal'] = 0
    df.loc[df['Close'] < df['BB_Lower'], 'Signal'] = 1
    df.loc[df['Close'] > df['BB_Upper'], 'Signal'] = -1
    return df


def macd_strategy(df, fast=12, slow=26, signal=9):
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    df['MACD_Line'] = ema_fast - ema_slow
    df['MACD_Signal'] = df['MACD_Line'].ewm(span=signal, adjust=False).mean()
    df['Signal'] = 0
    df.loc[df['MACD_Line'] > df['MACD_Signal'], 'Signal'] = 1
    df.loc[df['MACD_Line'] < df['MACD_Signal'], 'Signal'] = -1
    return df


def pin_bar_strategy(df):
    df['Body'] = abs(df['Close'] - df['Open'])
    df['Range'] = df['High'] - df['Low']
    df['Upper_Wick'] = df['High'] - df[['Close', 'Open']].max(axis=1)
    df['Lower_Wick'] = df[['Close', 'Open']].min(axis=1) - df['Low']
    df['Signal'] = 0
    bullish_pin = (df['Lower_Wick'] > 2 * df['Body']
                   ) & (df['Upper_Wick'] < df['Body'])
    df.loc[bullish_pin, 'Signal'] = 1
    bearish_pin = (df['Upper_Wick'] > 2 * df['Body']
                   ) & (df['Lower_Wick'] < df['Body'])
    df.loc[bearish_pin, 'Signal'] = -1
    return df


def engulfing_strategy(df):
    df['Prev_Open'] = df['Open'].shift(1)
    df['Prev_Close'] = df['Close'].shift(1)
    df['Signal'] = 0
    bull_engulf = (
        (df['Close'] > df['Open']) &
        (df['Prev_Close'] < df['Prev_Open']) &
        (df['Open'] < df['Prev_Close']) &
        (df['Close'] > df['Prev_Open'])
    )
    df.loc[bull_engulf, 'Signal'] = 1
    bear_engulf = (
        (df['Open'] > df['Close']) &
        (df['Prev_Close'] > df['Prev_Open']) &
        (df['Open'] > df['Prev_Close']) &
        (df['Close'] < df['Prev_Open'])
    )
    df.loc[bear_engulf, 'Signal'] = -1
    return df


def confluence_strategy(df, threshold=2):
    df = sma_crossover_strategy(df.copy())
    df['SMA_Vote'] = df['Signal']
    df = rsi_strategy(df.copy())
    df['RSI_Vote'] = df['Signal']
    df = pin_bar_strategy(df.copy())
    df['Pin_Vote'] = df['Signal']
    df = engulfing_strategy(df.copy())
    df['Engulf_Vote'] = df['Signal']
    df = bollinger_strategy(df.copy())
    df['BB_Vote'] = df['Signal']
    df = macd_strategy(df.copy())
    df['MACD_Vote'] = df['Signal']
    vote_cols = ['SMA_Vote', 'RSI_Vote', 'Pin_Vote',
                 'Engulf_Vote', 'BB_Vote', 'MACD_Vote']
    df['Total_Score'] = df[vote_cols].sum(axis=1)
    df['Signal'] = 0
    df.loc[df['Total_Score'] >= threshold, 'Signal'] = 1
    df.loc[df['Total_Score'] <= -threshold, 'Signal'] = -1
    return df


def scalper_strategy(df, threshold=2):
    df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['Trend_Vote'] = np.where(df['EMA9'] > df['EMA21'], 1, -1)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    df['RSI_Vote'] = np.where(
        df['RSI'] < 30, 1, np.where(df['RSI'] > 70, -1, 0))
    df['Total_Score'] = df['Trend_Vote'] + df['RSI_Vote']
    df['Signal'] = 0
    df.loc[df['Total_Score'] >= threshold, 'Signal'] = 1
    df.loc[df['Total_Score'] <= -threshold, 'Signal'] = -1
    return df


def run_backtest(df, ticker='strategy', initial_balance=10000, comm=0.001, tp_pct=0.05, trail_pct=0.02):
    cash = float(initial_balance)
    shares = 0
    entry_price = 0.0
    peak_price = 0.0
    current_side = 0
    wealth_history = []
    trades = []
    trade_dates = []
    daily_returns = []

    for i, row in enumerate(df.itertuples()):
        price = float(row.Close)
        high = float(row.High)
        low = float(row.Low)
        signal = row.Signal

        if pd.isna(signal):
            if current_side == 1:
                val = cash + shares * price
            elif current_side == -1:
                val = cash - (shares * price)
            else:
                val = cash
            wealth_history.append(val)
            continue

        if current_side == 0:
            if signal == 1:
                current_side = 1
                shares = (cash * (1 - comm)) / price
                cash = 0.0
                entry_price = price
                peak_price = price
            elif signal == -1:
                current_side = -1
                shares = (cash * (1 - comm)) / price
                cash = cash + shares * price * (1 - comm)
                entry_price = price
                peak_price = price

        elif current_side == 1:
            peak_price = max(peak_price, high)
            stop_p = peak_price * (1 - trail_pct)
            tp_p = entry_price * (1 + tp_pct)

            if low <= stop_p or high >= tp_p or signal == -1:
                exit_p = stop_p if low <= stop_p else (
                    tp_p if high >= tp_p else price)
                cash = shares * exit_p * (1 - comm)
                trade_return = (exit_p - entry_price) / entry_price
                trades.append(trade_return)
                trade_dates.append(df.index[i])
                shares = 0
                current_side = 0

        elif current_side == -1:
            peak_price = min(peak_price, low)
            stop_p = peak_price * (1 + trail_pct)
            tp_p = entry_price * (1 - tp_pct)

            if high >= stop_p or low <= tp_p or signal == 1:
                exit_p = stop_p if high >= stop_p else (
                    tp_p if low <= tp_p else price)
                buyback_cost = shares * exit_p * (1 + comm)
                cash = cash - buyback_cost
                trade_return = (entry_price - exit_p) / entry_price
                trades.append(trade_return)
                trade_dates.append(df.index[i])
                shares = 0
                current_side = 0

        if current_side == 1:
            val = cash + shares * price
        elif current_side == -1:
            val = cash - (shares * price)
        else:
            val = cash
        wealth_history.append(val)

        if i > 0 and wealth_history[-2] != 0:
            daily_returns.append(
                (wealth_history[-1] - wealth_history[-2]) / wealth_history[-2])

    wealth_series = pd.Series(wealth_history, index=df.index)
    total_return = (
        (wealth_series.iloc[-1] - initial_balance) / initial_balance) * 100
    first_price = float(df['Close'].iloc[0])
    last_price = float(df['Close'].iloc[-1])
    buy_hold_return = ((last_price - first_price) / first_price) * 100
    buy_hold_wealth = (initial_balance / first_price) * df['Close']
    total_trades = len(trades)
    if total_trades > 0:
        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t <= 0]
        win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
        avg_win = np.mean(wins) * 100 if wins else 0
        avg_loss = np.mean(losses) * 100 if losses else 0
        profit_factor = abs(sum(wins) / sum(losses)
                            ) if sum(losses) != 0 else float('inf')
    else:
        win_rate = avg_win = avg_loss = profit_factor = 0
    rolling_max = wealth_series.cummax()
    max_drawdown = ((wealth_series - rolling_max) / rolling_max).min() * 100
    sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * \
        np.sqrt(252) if daily_returns and np.std(daily_returns) > 0 else 0
    calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0

    print(f"{' ENHANCED STRATEGY vs BENCHMARK ':=^50}")
    print(f"Strategy Return:          {total_return:>10.2f}%")
    print(f"Buy & Hold:               {buy_hold_return:>10.2f}%")
    print(
        f"Alpha (Edge):             {total_return - buy_hold_return:>10.2f}%")
    print(f"{'-'*50}")
    print(f"Total Trades:             {total_trades:>10}")
    print(f"Win Rate:                 {win_rate:>10.2f}%")
    print(f"Avg Win / Avg Loss:       {avg_win:>10.2f}% / {avg_loss:>10.2f}%")
    print(f"Profit Factor:            {profit_factor:>10.2f}")
    print(f"Max Drawdown:             {max_drawdown:>10.2f}%")
    print(f"Sharpe Ratio:             {sharpe_ratio:>10.2f}")
    print(f"Calmar Ratio:             {calmar_ratio:>10.2f}")
    print(f"{'='*50}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    ax1.plot(df.index, wealth_series, label='Algo Strategy',
             color='#2ecc71', linewidth=2)
    ax1.plot(df.index, buy_hold_wealth, label='Buy & Hold',
             color='#95a5a6', ls='--', linewidth=2)
    ax1.legend(fontsize=12)
    ax1.set_title(
        f'Backtest Results - {ticker.upper()}', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    drawdown = ((wealth_series - rolling_max) / rolling_max) * 100
    ax2.fill_between(df.index, drawdown, 0, color='red', alpha=0.3)
    ax2.plot(df.index, drawdown, color='red', linewidth=1)
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return wealth_history, trades, trade_dates


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Run the backtester from notebook code.')
    parser.add_argument('--ticker', default='BTC/USD',
                        help='Ticker to fetch or simulate')
    parser.add_argument('--years', type=int, default=5,
                        help='Years of historical data')
    parser.add_argument('--interval', default='1Day', help='Data interval')
    parser.add_argument('--initial-balance', type=float,
                        default=10000.0, help='Starting cash balance')
    parser.add_argument('--use-sample', action='store_true',
                        help='Use synthetic sample data instead of Alpaca')
    args = parser.parse_args()

    if args.use_sample:
        print('Using synthetic sample data instead of Alpaca.')
        raw_data = generate_sample_data(periods=250, start_price=100.0)
    else:
        try:
            raw_data = get_data(
                args.ticker, years=args.years, interval=args.interval)
        except Exception as exc:
            print(f"Warning: data fetch failed: {exc}")
            print('Falling back to synthetic sample data.')
            raw_data = generate_sample_data(periods=250, start_price=100.0)

    processed_df = confluence_strategy(raw_data)
    print('\nProcessed DataFrame head:')
    print(processed_df[['Open', 'High', 'Low', 'Close', 'Signal']].head(10))
    run_backtest(processed_df, ticker=args.ticker,
                 initial_balance=args.initial_balance)


if __name__ == '__main__':
    main()
