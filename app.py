"""
SOLBOT - Flask Web Server
Serves the dashboard and provides real-time API endpoints
"""

import os
import json
import logging
import threading
import time
from datetime import datetime
from flask import Flask, jsonify, render_template_string, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv

from engine import (
    get_sol_price, get_ohlcv, calculate_indicators,
    get_indicator_snapshot, detect_patterns, generate_signal,
    paper_trade, get_paper_state, PAPER_TRADING
)
from scraper import NewsScraper
from ai_brain import analyze_trade_with_claude, get_brain_summary, get_pattern_weight

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'solbot-secret-2024')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='eventlet')

# ── Global state ──
bot_running   = False
bot_paused    = False
scraper       = NewsScraper(account=os.getenv('SCRAPER_ACCOUNT', 'solidintel_x'))
scheduler     = BackgroundScheduler()
last_scan     = None
last_signal   = None
last_indicators = None
last_patterns   = None
last_sentiment  = None
system_log    = []
scan_count    = 0


def add_log(msg, level='info'):
    """Add entry to system log and broadcast to dashboard."""
    entry = {
        'time':    datetime.now().strftime('%H:%M:%S'),
        'message': msg,
        'level':   level,
    }
    system_log.append(entry)
    if len(system_log) > 200:
        system_log.pop(0)
    socketio.emit('log', entry)
    log.info(f"[{level.upper()}] {msg}")


# ══════════════════════════════════════════════════
# MAIN BOT SCAN CYCLE
# ══════════════════════════════════════════════════
def run_scan():
    """Core hourly scan: fetch data → analyze → signal → trade."""
    global last_scan, last_signal, last_indicators, last_patterns, last_sentiment, scan_count

    if not bot_running or bot_paused:
        return

    scan_count += 1
    add_log(f'Scan #{scan_count} initiated — fetching SOL/USD 1H data from Kraken...', 'info')

    try:
        # 1. Fetch OHLCV
        df = get_ohlcv(interval=60, candles=100)
        if df is None:
            add_log('Failed to fetch candle data — retrying next cycle', 'error')
            return

        add_log(f'{len(df)} candles loaded. Running TA calculations...', 'info')

        # 2. Calculate indicators
        df = calculate_indicators(df)
        indicators = get_indicator_snapshot(df)
        last_indicators = indicators

        add_log(
            f"RSI: {indicators['rsi']:.1f} | "
            f"MACD: {'BULL' if indicators['macd_bull'] else 'BEAR'} | "
            f"BB: {indicators['bb_pos']:.0f}% | "
            f"MA: {'ABOVE' if indicators['above_ma20'] else 'BELOW'}",
            'sol'
        )

        # 3. Detect patterns
        patterns = detect_patterns(df)
        last_patterns = patterns
        if patterns:
            names = ', '.join([p['name'] for p in patterns])
            add_log(f"Patterns detected: {names}", 'info')
        else:
            add_log('No clear patterns detected this candle', 'info')

        # 4. News sentiment
        add_log(f'Scraping @{scraper.account} for news...', 'info')
        sentiment_data = scraper.analyze()
        last_sentiment = sentiment_data
        add_log(
            f"Sentiment: {sentiment_data['signal']} ({sentiment_data['score']:+.2f}) "
            f"from {sentiment_data['tweet_count']} posts",
            'success' if sentiment_data['score'] > 0 else 'warning'
        )

        # 5. Generate signal
        signal_data = generate_signal(indicators, patterns, sentiment_data['score'])
        last_signal = signal_data
        add_log(
            f"Signal: {signal_data['signal']} | "
            f"Bull: {signal_data['bull_score']:.1f} Bear: {signal_data['bear_score']:.1f} | "
            f"Confidence: {signal_data['confidence']:.0f}%",
            'success' if signal_data['signal'] != 'HOLD' else 'warning'
        )

        # 6. Execute paper trade
        price = indicators['price']
        prev_position = get_paper_state()['position']
        trade_result  = paper_trade(signal_data, price)

        if trade_result['action'] == 'open':
            add_log(
                f"[PAPER] {trade_result['type']} opened @ ${price:.4f} | "
                f"SL: ${trade_result['sl']:.4f} | TP: ${trade_result['tp']:.4f}",
                'success'
            )
        elif trade_result['action'] == 'close':
            add_log(trade_result['message'],
                    'success' if trade_result['pnl'] > 0 else 'error')

            # 7. AI analysis on closed trade
            add_log('Sending closed trade to Claude AI for analysis...', 'info')
            trade_record = get_paper_state()['recent_trades']
            if trade_record:
                last_trade = trade_record[-1]
                memory = analyze_trade_with_claude(
                    last_trade, patterns, indicators, sentiment_data['score']
                )
                if memory:
                    add_log(f"AI verdict: {memory['verdict']} | {memory['key_lesson']}", 'sol')
                    if memory['weight_changes']:
                        for pat, change in memory['weight_changes'].items():
                            add_log(f"Weight update: {pat} → {change['from']}% → {change['to']}%", 'info')

        last_scan = datetime.now().isoformat()

        # 8. Broadcast state update to all dashboard clients
        socketio.emit('state_update', {
            'indicators': indicators,
            'patterns':   patterns,
            'signal':     signal_data,
            'sentiment':  sentiment_data,
            'paper':      get_paper_state(),
            'last_scan':  last_scan,
        })

    except Exception as e:
        add_log(f'Scan error: {str(e)}', 'error')
        log.exception("Scan failed")


def price_ticker():
    """Broadcast live price every 5 seconds."""
    while True:
        if bot_running:
            price = get_sol_price()
            if price:
                socketio.emit('price', {'price': price, 'time': datetime.now().isoformat()})
        time.sleep(5)


# ══════════════════════════════════════════════════
# REST API ENDPOINTS
# ══════════════════════════════════════════════════
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/brain')
def brain():
    return send_from_directory('static', 'brain.html')

@app.route('/api/status')
def api_status():
    return jsonify({
        'bot_running':  bot_running,
        'bot_paused':   bot_paused,
        'paper_mode':   PAPER_TRADING,
        'last_scan':    last_scan,
        'scan_count':   scan_count,
        'uptime':       'active',
    })

@app.route('/api/state')
def api_state():
    return jsonify({
        'indicators': last_indicators,
        'patterns':   last_patterns,
        'signal':     last_signal,
        'sentiment':  last_sentiment,
        'paper':      get_paper_state(),
        'last_scan':  last_scan,
    })

@app.route('/api/price')
def api_price():
    price = get_sol_price()
    return jsonify({'price': price, 'symbol': 'SOL/USD'})

@app.route('/api/brain')
def api_brain():
    return jsonify(get_brain_summary())

@app.route('/api/logs')
def api_logs():
    return jsonify({'logs': system_log[-50:]})

@app.route('/api/bot/start', methods=['POST'])
def bot_start():
    global bot_running, bot_paused
    bot_running = True
    bot_paused  = False
    add_log('Bot started — connecting to Kraken & initializing systems...', 'success')
    add_log(f'Mode: {"PAPER TRADING (safe)" if PAPER_TRADING else "LIVE TRADING"}', 'warning')
    # Trigger immediate first scan
    threading.Thread(target=run_scan, daemon=True).start()
    return jsonify({'status': 'started'})

@app.route('/api/bot/pause', methods=['POST'])
def bot_pause():
    global bot_paused
    bot_paused = not bot_paused
    state = 'paused' if bot_paused else 'resumed'
    add_log(f'Bot {state}. Open positions maintained.', 'warning')
    return jsonify({'status': state})

@app.route('/api/bot/stop', methods=['POST'])
def bot_stop():
    global bot_running, bot_paused
    bot_running = False
    bot_paused  = False
    add_log('Bot stopped. All monitoring halted.', 'error')
    return jsonify({'status': 'stopped'})

@app.route('/api/scan/now', methods=['POST'])
def scan_now():
    """Trigger an immediate scan (useful for testing)."""
    if not bot_running:
        return jsonify({'error': 'Bot not running'}), 400
    threading.Thread(target=run_scan, daemon=True).start()
    return jsonify({'status': 'scan triggered'})


# ══════════════════════════════════════════════════
# WEBSOCKET EVENTS
# ══════════════════════════════════════════════════
@socketio.on('connect')
def on_connect():
    log.info("Dashboard client connected")
    emit('state_update', {
        'indicators': last_indicators,
        'patterns':   last_patterns,
        'signal':     last_signal,
        'sentiment':  last_sentiment,
        'paper':      get_paper_state(),
        'last_scan':  last_scan,
    })
    emit('logs', {'logs': system_log[-30:]})

@socketio.on('disconnect')
def on_disconnect():
    log.info("Dashboard client disconnected")


# ══════════════════════════════════════════════════
# STARTUP
# ══════════════════════════════════════════════════
def start_background_tasks():
    # Hourly scan scheduler
    scheduler.add_job(run_scan, 'interval', minutes=60, id='hourly_scan')
    scheduler.start()

    # Price ticker thread
    ticker_thread = threading.Thread(target=price_ticker, daemon=True)
    ticker_thread.start()

    add_log('SOLBOT systems initialized. Press START to begin trading.', 'sol')
    add_log(f'Mode: {"PAPER TRADING — no real money at risk" if PAPER_TRADING else "LIVE TRADING"}', 'warning')


if __name__ == '__main__':
    start_background_tasks()
    port = int(os.getenv('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
