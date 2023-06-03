import streamlit as st
from binance.client import Client
import pandas as pd
import numpy as np
import time
from ta.trend import EMAIndicator, SMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator, StochRSIIndicator
from ta.volume import VolumePriceTrendIndicator, OnBalanceVolumeIndicator
from ta.volatility import AverageTrueRange
from datetime import datetime
import requests
import time
import pytz
import config


def generate_signals(data):
    data['close'] = pd.to_numeric(data['close'], errors='coerce').fillna(0)
    data['volume'] = pd.to_numeric(data['volume'], errors='coerce').fillna(0)
    data['high'] = pd.to_numeric(data['high'], errors='coerce').fillna(0)
    data['low'] = pd.to_numeric(data['low'], errors='coerce').fillna(0)
    # Calculate MACD
    macd = MACD(data['close'], window_slow=26, window_fast=12, window_sign=9)
    data['macd'] = macd.macd()
    data['signal_line'] = macd.macd_signal()
    data['macd_histogram'] = macd.macd_diff()

    # Calculate Stochastic Oscillator
    stochastic = StochasticOscillator(
        data['high'], data['low'], data['close'], window=14)
    stochastic_oscillator = stochastic.stoch()

    # Calculate Stochastic RSI (StochRSI)
    stoch_rsi = StochRSIIndicator(
        data['close'], window=14, smooth1=5, smooth2=3)
    data['stoch_rsi'] = stoch_rsi.stochrsi()

    # Calculate Exponential Moving Average (EMA)
    ema_7 = EMAIndicator(data['close'], window=7, fillna=True)
    ema_25 = EMAIndicator(data['close'], window=25, fillna=True)
    ema_smooth_7 = ema_7.ema_indicator().ewm(span=7, adjust=False).mean()
    ema_smooth_25 = ema_25.ema_indicator().ewm(span=25, adjust=False).mean()
    data['ema_7'] = ema_smooth_7
    data['ema_25'] = ema_smooth_25

    # Calculate Moving Average (MA)
    ma_7 = SMAIndicator(data['close'], window=7, fillna=True)
    ma_25 = SMAIndicator(data['close'], window=25, fillna=True)
    ma_smooth_7 = ma_7.sma_indicator()
    ma_smooth_25 = ma_25.sma_indicator()
    data['ma_7'] = ma_smooth_7
    data['ma_25'] = ma_smooth_25

    # Calculate Relative Strength Index (RSI)
    rsi = RSIIndicator(data['close'], window=14)
    data['rsi'] = rsi.rsi()

    # Calculate Average True Range (ATR)
    atr = AverageTrueRange(data['high'], data['low'], data['close'], window=14)
    data['atr'] = atr.average_true_range()

    # Calculate Volume-Price Relation
    volume_price_trend = VolumePriceTrendIndicator(
        data['close'], data['volume'])
    data['volume_price_relation'] = volume_price_trend.volume_price_trend()

    # Calculate the On-Balance Volume (OBV)
    obv_indicator = OnBalanceVolumeIndicator(data['close'], data['volume'])
    obv = obv_indicator.on_balance_volume()
    data['obv'] = obv

    # Determine signals and entry prices
    data['signal'] = 'Hold'
    data['entry_price'] = 0.0

    for i in range(1, len(data)):
        # Check if all indicators confirm a strong buy signal
        if (
            data['rsi'].iloc[i] > 50 and
            data['stoch_rsi'].iloc[i] > 0.8 and
            data['obv'].iloc[i] > data['obv'].iloc[i-1] and
            data['volume_price_relation'].iloc[i] > 0 and
            data['macd'].iloc[i] > data['signal_line'].iloc[i]
        ):
            data.at[i, 'signal'] = 'Strong Buy'
            data.at[i, 'entry_price'] = data['close'].iloc[i]

        # Check if all indicators confirm a strong sell signal
        elif (
            data['rsi'].iloc[i] < 50 and
            data['stoch_rsi'].iloc[i] < 0.2 and
            data['obv'].iloc[i] < data['obv'].iloc[i-1] and
            data['volume_price_relation'].iloc[i] < 0 and
            data['macd'].iloc[i] < data['signal_line'].iloc[i]
        ):
            data.at[i, 'signal'] = 'Strong Sell'
            data.at[i, 'entry_price'] = data['close'].iloc[i]

        # Check if the EMA 7 crosses above the EMA 25
        if (
            data['ma_7'].iloc[i] > data['ma_25'].iloc[i] and
            data['ma_7'].iloc[i-1] < data['ma_25'].iloc[i-1]
        ):
            data.at[i, 'signal'] = 'Buy'
            data.at[i, 'entry_price'] = data['close'].iloc[i]

        # Check if the EMA 7 crosses below the EMA 25
        if (
            data['ma_7'].iloc[i] < data['ma_25'].iloc[i] and
            data['ma_7'].iloc[i-1] > data['ma_25'].iloc[i-1]
        ):
            data.at[i, 'signal'] = 'Sell'
            data.at[i, 'entry_price'] = data['close'].iloc[i]

    data = data.sort_values(
        'timestamp', ascending=False).reset_index(drop=True)

    return data


def get_nearest_price(data, index, price_type):
    current_price = data[price_type].iloc[index]
    previous_prices = data[price_type].iloc[:index]
    nearest_price_index = np.abs(
        previous_prices - current_price).idxmin()
    nearest_price = data[price_type].iloc[nearest_price_index]
    return nearest_price


def calculate_order_book_pressure(symbol):
    url = f'https://api.binance.com/api/v3/depth?symbol={symbol}&limit=1000'
    response = requests.get(url)
    data = response.json()

    bids = data['bids']  # Buy orders
    asks = data['asks']  # Sell orders

    # Cumulative volume of buy orders
    bid_pressure = sum([float(bid[1]) for bid in bids])
    # Cumulative volume of sell orders
    ask_pressure = sum([float(ask[1]) for ask in asks])

    return bid_pressure, ask_pressure


def main():
    # Set up Binance API credentials

    client = Client(config.api_key, config.api_secret)

    # Get a list of available cryptocurrencies
    exchange_info = client.futures_exchange_info()
    symbols = [symbol['symbol'] for symbol in exchange_info['symbols']
               if symbol['symbol'].endswith('USDT')]

    # Display a selectbox to choose a cryptocurrency
    selected_symbol = st.select_slider(
        'Select a cryptocurrency', symbols)
    # Calculate order book pressure for the selected symbol
    buy_pressure, sell_pressure = calculate_order_book_pressure(
        selected_symbol)

    # Display the buy and sell pressure
    st.write(f"Buy Pressure : {buy_pressure}")
    st.write(f"Sell Pressure: {sell_pressure}")

    # Fetch and display live data
    while True:
        # Get cryptocurrency data from Binance API
        data = client.futures_klines(
            symbol=selected_symbol, interval=Client.KLINE_INTERVAL_1HOUR)
        data = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                           'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                                           'taker_buy_quote_asset_volume', 'ignore'])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        data['timestamp'] = data['timestamp'].dt.tz_localize(
            'UTC').dt.tz_convert(pytz.timezone('Asia/Riyadh'))

        # Generate signals
        data_with_signals = generate_signals(data)

        # Display the data with signals
        pd.set_option('display.max_rows', None)  # Show all rows
        pd.set_option('display.max_columns', None)  # Show all columns
        st.dataframe(data_with_signals.head(100))

        # Add a delay to update the data at a specific interval
        time.sleep(3600)  # Wait for 32 minutes


if __name__ == '__main__':
    main()
