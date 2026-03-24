"""
SOLBOT - Core Trading Engine
Handles: Kraken data, TA indicators, pattern detection, paper/live trading
"""

import os
import time
import json
import logging
import requests
import pandas as pd
import ta
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════
PAPER_TRADING   = os.getenv('PAPER_TRADING', 'true').lower() == 'true'
TRADE_AMOUNT    = float(os.getenv('TRADE_AMOUNT', 1000))
LEVERAGE        = float(os.getenv('LEVERAGE', 1))          # 1x = no leverage (safe default)
SYMBOL          = os.getenv('SYMBOL', 'SOLUSDT')
KRAKEN_SYMBOL   = 'XETHZUSD'  # Kraken uses SOLXXX format — mapped below
TIMEFRAME       = int(os.getenv('TIMEFRAME', 60))       # minutes
STOP_LOSS_PCT   = float(os.getenv('STOP_LOSS_PCT', 3))
TAKE_PROFIT_PCT = float(os.getenv('TAKE_PROFIT_PCT', 6))

# Kraken OHLC interval map
KRAKEN_INTERVALS = {1: 1, 5: 5, 15: 15, 30: 30, 60: 60, 240: 240, 1440: 1440}

# ══════════════════════════════════════════════════
# PAPER TRADING STATE
# ══════════════════════════════════════════════════
paper_state = {
    'balance': TRADE_AMOUNT,
    'position': None,        # 'long' | 'short' | None
    'entry_price': None,
    'entry_time': None,
    'stop_loss': None,
    'take_profit': None,
    'trades': [],
    'wins': 0,
    'losses': 0,
    'total_pnl': 0.0,
    'leverage': LEVERAGE,
}

# ══════════════════════════════════════════════════
# KRAKEN DATA FEED
# ══════════════════════════════════════════════════
def get_sol_price():
    """Fetch current SOL/USD price from multiple exchanges."""
    
    # Try Kraken first
    try:
        url = 'https://api.kraken.com/0/public/Ticker?pair=SOLUSD'
        r = requests.get(url, timeout=10)
        data = r.json()
        if not data.get('error'):
            result = data['result']
            pair_key = list(result.keys())[0]
            price = float(result[pair_key]['c'][0])
            log.info(f"Kraken SOL/USD: ${price:.4f}")
            return price
    except Exception as e:
        log.error(f"Kraken error: {e}")
    
    # Try Binance (more reliable)
    try:
        url = 'https://api.binance.com/api/v3/ticker/price?symbol=SOLUSDT'
        r = requests.get(url, timeout=10)
        data = r.json()
        price = float(data['price'])
        log.info(f"Binance SOL/USDT: ${price:.4f}")
        return price
    except Exception as e:
        log.error(f"Binance error: {e}")
    
    # Try CoinGecko
    try:
        url = 'https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=usd'
        r = requests.get(url, timeout=10)
        data = r.json()
        price = data['solana']['usd']
        log.info(f"CoinGecko SOL/USD: ${price:.4f}")
        return price
    except Exception as e:
        log.error(f"CoinGecko error: {e}")
    
    # Fallback to mock
    log.warning("All price APIs failed, using fallback")
    return 90.19


def get_ohlcv(interval=60, candles=100):
    """Fetch OHLCV candles from Kraken for SOL/USD."""
    try:
        url = f'https://api.kraken.com/0/public/OHLC?pair=SOLUSD&interval={interval}'
        r = requests.get(url, timeout=15)
        data = r.json()
        if data.get('error'):
            log.error(f"Kraken OHLC error: {data['error']}")
            return None
        result = data['result']
        pair_key = [k for k in result.keys() if k != 'last'][0]
        raw = result[pair_key][-candles:]
        df = pd.DataFrame(raw, columns=['time','open','high','low','close','vwap','volume','count'])
        for col in ['open','high','low','close','volume']:
            df[col] = df[col].astype(float)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        log.info(f"Fetched {len(df)} candles for SOL/USD {interval}m")
        return df
    except Exception as e:
        log.error(f"OHLCV fetch error: {e}")
        return None


# ══════════════════════════════════════════════════
# TECHNICAL ANALYSIS ENGINE
# ══════════════════════════════════════════════════
def calculate_indicators(df):
    try:
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()
        bb = ta.volatility.BollingerBands(df['close'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_mid'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['ma20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
        df['ma50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
        df['ema9'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        return df
    except Exception as e:
        log.error(f"Indicator error: {e}")
        return df


def get_indicator_snapshot(df):
    """Return latest indicator values as a clean dict."""
    last = df.iloc[-1]
    prev = df.iloc[-2]

    rsi   = float(last['rsi']) if not pd.isna(last['rsi']) else 50.0
    macd  = float(last['macd']) if not pd.isna(last['macd']) else 0.0
    macd_sig = float(last['macd_signal']) if not pd.isna(last['macd_signal']) else 0.0
    close = float(last['close'])
    bb_upper = float(last['bb_upper']) if not pd.isna(last['bb_upper']) else close
    bb_lower = float(last['bb_lower']) if not pd.isna(last['bb_lower']) else close
    bb_mid   = float(last['bb_mid'])   if not pd.isna(last['bb_mid'])   else close
    ma20  = float(last['ma20']) if not pd.isna(last['ma20']) else close
    ma50  = float(last['ma50']) if not pd.isna(last['ma50']) else close

    # BB position: 0 = at lower, 100 = at upper
    bb_range = bb_upper - bb_lower
    bb_pos = ((close - bb_lower) / bb_range * 100) if bb_range > 0 else 50

    # MACD crossover
    macd_cross_bull = float(prev['macd']) < float(prev['macd_signal']) and macd > macd_sig
    macd_cross_bear = float(prev['macd']) > float(prev['macd_signal']) and macd < macd_sig

    return {
        'price':       round(close, 4),
        'rsi':         round(rsi, 2),
        'macd':        round(macd, 4),
        'macd_signal': round(macd_sig, 4),
        'macd_bull':   macd > macd_sig,
        'macd_cross_bull': macd_cross_bull,
        'macd_cross_bear': macd_cross_bear,
        'bb_upper':    round(bb_upper, 4),
        'bb_lower':    round(bb_lower, 4),
        'bb_mid':      round(bb_mid, 4),
        'bb_pos':      round(bb_pos, 1),
        'ma20':        round(ma20, 4),
        'ma50':        round(ma50, 4),
        'above_ma20':  close > ma20,
        'above_ma50':  close > ma50,
        'atr':         round(float(last['atr']), 4) if not pd.isna(last['atr']) else 0,
    }


# ══════════════════════════════════════════════════
# PATTERN DETECTION ENGINE
# ══════════════════════════════════════════════════
def detect_patterns(df):
    """Detect candlestick and chart patterns. Returns list of detected patterns."""
    detected = []
    c = df.iloc[-1]   # current candle
    p = df.iloc[-2]   # previous candle
    p2 = df.iloc[-3]  # two back

    body_c = abs(float(c['close']) - float(c['open']))
    range_c = float(c['high']) - float(c['low'])
    body_p = abs(float(p['close']) - float(p['open']))

    bull_c = float(c['close']) > float(c['open'])
    bull_p = float(p['close']) > float(p['open'])

    upper_wick_c = float(c['high']) - max(float(c['close']), float(c['open']))
    lower_wick_c = min(float(c['close']), float(c['open'])) - float(c['low'])

    # ── SINGLE CANDLESTICK ──

    # Hammer
    if (not bull_c and lower_wick_c >= 2 * body_c and upper_wick_c <= body_c * 0.3 and range_c > 0):
        detected.append({'id': 'hammer', 'name': 'Hammer', 'signal': 'BULLISH', 'reliability': 72})

    # Shooting Star
    if (bull_p and upper_wick_c >= 2 * body_c and lower_wick_c <= body_c * 0.3 and range_c > 0):
        detected.append({'id': 'shooting_star', 'name': 'Shooting Star', 'signal': 'BEARISH', 'reliability': 74})

    # Doji
    if body_c <= range_c * 0.1 and range_c > 0:
        if lower_wick_c >= range_c * 0.6:
            detected.append({'id': 'dragonfly_doji', 'name': 'Dragonfly Doji', 'signal': 'BULLISH', 'reliability': 68})
        elif upper_wick_c >= range_c * 0.6:
            detected.append({'id': 'gravestone_doji', 'name': 'Gravestone Doji', 'signal': 'BEARISH', 'reliability': 70})

    # Marubozu
    if upper_wick_c <= body_c * 0.05 and lower_wick_c <= body_c * 0.05 and body_c > 0:
        if bull_c:
            detected.append({'id': 'bullish_marubozu', 'name': 'Bullish Marubozu', 'signal': 'BULLISH', 'reliability': 78})
        else:
            detected.append({'id': 'bearish_marubozu', 'name': 'Bearish Marubozu', 'signal': 'BEARISH', 'reliability': 77})

    # ── MULTI-CANDLE ──

    # Bullish Engulfing
    if (not bull_p and bull_c and
        float(c['open']) < float(p['close']) and
        float(c['close']) > float(p['open']) and
        body_c > body_p):
        detected.append({'id': 'bullish_engulfing', 'name': 'Bullish Engulfing', 'signal': 'BULLISH', 'reliability': 82})

    # Bearish Engulfing
    if (bull_p and not bull_c and
        float(c['open']) > float(p['close']) and
        float(c['close']) < float(p['open']) and
        body_c > body_p):
        detected.append({'id': 'bearish_engulfing', 'name': 'Bearish Engulfing', 'signal': 'BEARISH', 'reliability': 81})

    # Tweezer Top
    if (bull_p and not bull_c and
        abs(float(p['high']) - float(c['high'])) <= float(c['high']) * 0.001):
        detected.append({'id': 'tweezer_top', 'name': 'Tweezer Top', 'signal': 'BEARISH', 'reliability': 70})

    # Tweezer Bottom
    if (not bull_p and bull_c and
        abs(float(p['low']) - float(c['low'])) <= float(c['low']) * 0.001):
        detected.append({'id': 'tweezer_bottom', 'name': 'Tweezer Bottom', 'signal': 'BULLISH', 'reliability': 70})

    # Morning Star
    body_p2 = abs(float(p2['close']) - float(p2['open']))
    bull_p2 = float(p2['close']) > float(p2['open'])
    if (not bull_p2 and body_p <= body_p2 * 0.3 and bull_c and
        float(c['close']) > (float(p2['open']) + float(p2['close'])) / 2):
        detected.append({'id': 'morning_star', 'name': 'Morning Star', 'signal': 'BULLISH', 'reliability': 84})

    # Evening Star
    if (bull_p2 and body_p <= body_p2 * 0.3 and not bull_c and
        float(c['close']) < (float(p2['open']) + float(p2['close'])) / 2):
        detected.append({'id': 'evening_star', 'name': 'Evening Star', 'signal': 'BEARISH', 'reliability': 83})

    # Dark Cloud Cover
    if (bull_p and not bull_c and
        float(c['open']) > float(p['high']) and
        float(c['close']) < (float(p['open']) + float(p['close'])) / 2):
        detected.append({'id': 'dark_cloud_cover', 'name': 'Dark Cloud Cover', 'signal': 'BEARISH', 'reliability': 73})

    # Piercing Line
    if (not bull_p and bull_c and
        float(c['open']) < float(p['low']) and
        float(c['close']) > (float(p['open']) + float(p['close'])) / 2):
        detected.append({'id': 'piercing_line', 'name': 'Piercing Line', 'signal': 'BULLISH', 'reliability': 74})

    # ── CHART PATTERNS (multi-candle window) ──
    window = df.iloc[-20:]  # last 20 candles

    # Bull Flag detection
    recent_high = window['close'].max()
    recent_low  = window['close'].min()
    pole_size   = recent_high - recent_low
    last5_range = window.iloc[-5:]['close'].max() - window.iloc[-5:]['close'].min()
    if pole_size > 0 and last5_range < pole_size * 0.3 and float(c['close']) > float(window.iloc[-5]['close']):
        detected.append({'id': 'bull_flag', 'name': 'Bull Flag', 'signal': 'BULLISH', 'reliability': 85})

    # Double Bottom (simplified)
    lows = window['low'].nsmallest(2).values
    if len(lows) == 2 and abs(lows[0] - lows[1]) / lows[0] < 0.01:
        detected.append({'id': 'double_bottom', 'name': 'Double Bottom', 'signal': 'BULLISH', 'reliability': 83})

    # Double Top (simplified)
    highs = window['high'].nlargest(2).values
    if len(highs) == 2 and abs(highs[0] - highs[1]) / highs[0] < 0.01:
        detected.append({'id': 'double_top', 'name': 'Double Top', 'signal': 'BEARISH', 'reliability': 83})

    log.info(f"Detected {len(detected)} patterns: {[p['name'] for p in detected]}")
    return detected


# ══════════════════════════════════════════════════
# SIGNAL FUSION ENGINE
# ══════════════════════════════════════════════════
def generate_signal(indicators, patterns, sentiment_score):
    """
    Combine TA indicators + detected patterns + news sentiment
    into a final trade signal: LONG | SHORT | HOLD
    """
    bull_score = 0
    bear_score = 0
    reasons    = []

    # ── RSI ──
    rsi = indicators['rsi']
    if rsi < 35:
        bull_score += 2
        reasons.append(f'RSI oversold ({rsi:.1f})')
    elif rsi > 65:
        bear_score += 2
        reasons.append(f'RSI overbought ({rsi:.1f})')
    elif rsi < 50:
        bear_score += 0.5
    else:
        bull_score += 0.5

    # ── MACD ──
    if indicators['macd_cross_bull']:
        bull_score += 3
        reasons.append('MACD bullish crossover')
    elif indicators['macd_cross_bear']:
        bear_score += 3
        reasons.append('MACD bearish crossover')
    elif indicators['macd_bull']:
        bull_score += 1
        reasons.append('MACD above signal')
    else:
        bear_score += 1
        reasons.append('MACD below signal')

    # ── Bollinger Bands ──
    bb_pos = indicators['bb_pos']
    if bb_pos < 20:
        bull_score += 2
        reasons.append(f'Price near BB lower ({bb_pos:.0f}%)')
    elif bb_pos > 80:
        bear_score += 2
        reasons.append(f'Price near BB upper ({bb_pos:.0f}%)')

    # ── Moving Averages ──
    if indicators['above_ma20'] and indicators['above_ma50']:
        bull_score += 1.5
        reasons.append('Price above MA20 & MA50')
    elif not indicators['above_ma20'] and not indicators['above_ma50']:
        bear_score += 1.5
        reasons.append('Price below MA20 & MA50')

    # ── Pattern Scores ──
    for pat in patterns:
        weight = pat['reliability'] / 100
        if pat['signal'] == 'BULLISH':
            bull_score += 2 * weight
            reasons.append(f"Pattern: {pat['name']} (bull, {pat['reliability']}%)")
        elif pat['signal'] == 'BEARISH':
            bear_score += 2 * weight
            reasons.append(f"Pattern: {pat['name']} (bear, {pat['reliability']}%)")

    # ── Sentiment Score ──
    if sentiment_score > 0.3:
        bull_score += 1.5
        reasons.append(f'News sentiment bullish ({sentiment_score:+.2f})')
    elif sentiment_score < -0.3:
        bear_score += 1.5
        reasons.append(f'News sentiment bearish ({sentiment_score:+.2f})')
    else:
        reasons.append(f'News sentiment neutral ({sentiment_score:+.2f})')

    # ── Final Decision ──
    total = bull_score + bear_score
    confidence = abs(bull_score - bear_score) / total * 100 if total > 0 else 0

    if bull_score > bear_score and confidence >= 30:
        signal = 'LONG'
    elif bear_score > bull_score and confidence >= 30:
        signal = 'SHORT'
    else:
        signal = 'HOLD'

    log.info(f"Signal: {signal} | Bull: {bull_score:.1f} Bear: {bear_score:.1f} Conf: {confidence:.0f}%")

    return {
        'signal':     signal,
        'bull_score': round(bull_score, 2),
        'bear_score': round(bear_score, 2),
        'confidence': round(confidence, 1),
        'reasons':    reasons,
    }


# ══════════════════════════════════════════════════
# PAPER TRADING ENGINE
# ══════════════════════════════════════════════════
def paper_trade(signal_data, price):
    """Execute paper trade based on signal. Returns trade result dict."""
    global paper_state
    result = {'action': 'none', 'message': '', 'price': price}

    # Check existing position stop/target
    if paper_state['position']:
        entry = paper_state['entry_price']
        sl    = paper_state['stop_loss']
        tp    = paper_state['take_profit']
        pos   = paper_state['position']

        hit_sl = (pos == 'long'  and price <= sl) or (pos == 'short' and price >= sl)
        hit_tp = (pos == 'long'  and price >= tp) or (pos == 'short' and price <= tp)

        if hit_sl or hit_tp:
            if pos == 'long':
                pnl = (price - entry) / entry * TRADE_AMOUNT * LEVERAGE
            else:
                pnl = (entry - price) / entry * TRADE_AMOUNT * LEVERAGE

            outcome = 'WIN' if pnl > 0 else 'LOSS'
            paper_state['balance'] += pnl
            paper_state['total_pnl'] += pnl
            if pnl > 0: paper_state['wins'] += 1
            else: paper_state['losses'] += 1

            trade_record = {
                'type':     pos.upper(),
                'entry':    entry,
                'exit':     price,
                'pnl':      round(pnl, 2),
                'outcome':  outcome,
                'reason':   'STOP LOSS' if hit_sl else 'TAKE PROFIT',
                'time':     datetime.now().isoformat(),
                'patterns': [],
            }
            paper_state['trades'].append(trade_record)
            paper_state['position']    = None
            paper_state['entry_price'] = None
            paper_state['stop_loss']   = None
            paper_state['take_profit'] = None

            result = {'action': 'close', 'outcome': outcome, 'pnl': round(pnl, 2),
                      'message': f"Position closed: {outcome} | P&L: ${pnl:+.2f}", 'price': price}
            log.info(result['message'])
            return result

    # Open new position
    signal = signal_data['signal']
    if signal in ('LONG', 'SHORT') and paper_state['position'] is None:
        if signal == 'LONG':
            sl = round(price * (1 - STOP_LOSS_PCT / 100), 4)
            tp = round(price * (1 + TAKE_PROFIT_PCT / 100), 4)
        else:
            sl = round(price * (1 + STOP_LOSS_PCT / 100), 4)
            tp = round(price * (1 - TAKE_PROFIT_PCT / 100), 4)

        paper_state['position']    = signal.lower()
        paper_state['entry_price'] = price
        paper_state['entry_time']  = datetime.now().isoformat()
        paper_state['stop_loss']   = sl
        paper_state['take_profit'] = tp

        result = {
            'action':  'open',
            'type':    signal,
            'entry':   price,
            'sl':      sl,
            'tp':      tp,
            'amount':  TRADE_AMOUNT,
            'message': f"[PAPER] {signal} opened @ ${price} | SL: ${sl} | TP: ${tp}",
            'price':   price,
        }
        log.info(result['message'])

    return result


def get_paper_state():
    """Return clean paper trading state for the dashboard."""
    ps = paper_state
    total_trades = ps['wins'] + ps['losses']
    win_rate = (ps['wins'] / total_trades * 100) if total_trades > 0 else 0

    # Unrealized P&L
    unrealized = 0
    current_price = get_sol_price()
    if ps['position'] and ps['entry_price'] and current_price:
        if ps['position'] == 'long':
            unrealized = (current_price - ps['entry_price']) / ps['entry_price'] * TRADE_AMOUNT * LEVERAGE
        else:
            unrealized = (ps['entry_price'] - current_price) / ps['entry_price'] * TRADE_AMOUNT * LEVERAGE

    return {
        'balance':       round(ps['balance'], 2),
        'position':      ps['position'].upper() if ps['position'] else 'NONE',
        'entry_price':   ps['entry_price'],
        'stop_loss':     ps['stop_loss'],
        'take_profit':   ps['take_profit'],
        'unrealized_pnl': round(unrealized, 2),
        'total_pnl':     round(ps['total_pnl'], 2),
        'total_trades':  total_trades,
        'wins':          ps['wins'],
        'losses':        ps['losses'],
        'win_rate':      round(win_rate, 1),
        'recent_trades': ps['trades'][-10:],
        'paper_mode':    PAPER_TRADING,
        'leverage':      LEVERAGE,
        'starting_balance': TRADE_AMOUNT,
    }
