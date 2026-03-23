"""
SOLBOT - AI Learning Engine (Claude API)
Analyzes closed trades, updates pattern confidence weights, builds bot memory
"""

import os
import json
import logging
from datetime import datetime
from anthropic import Anthropic

log = logging.getLogger(__name__)
client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY', ''))

# ── Pattern confidence weights (starts at base, learns over time) ──
# Loaded from memory file if exists, otherwise initialized from defaults
WEIGHT_FILE = 'pattern_weights.json'

DEFAULT_WEIGHTS = {
    'hammer': 72, 'inv_hammer': 65, 'dragonfly_doji': 68,
    'bullish_marubozu': 78, 'shooting_star': 74, 'hanging_man': 65,
    'gravestone_doji': 70, 'bearish_marubozu': 77,
    'bullish_engulfing': 82, 'morning_star': 84, 'three_white_soldiers': 83,
    'tweezer_bottom': 70, 'bullish_harami': 67, 'piercing_line': 74,
    'bearish_engulfing': 81, 'evening_star': 83, 'three_black_crows': 82,
    'tweezer_top': 70, 'bearish_harami': 66, 'dark_cloud_cover': 73,
    'head_shoulders': 88, 'inv_head_shoulders': 87,
    'double_top': 83, 'double_bottom': 83,
    'bull_flag': 85, 'bear_flag': 84, 'ascending_triangle': 80,
    'descending_triangle': 80, 'symmetrical_triangle': 74,
    'cup_handle': 86, 'rising_wedge': 78, 'falling_wedge': 79,
}

# In-memory brain log
brain_memory = []


def load_weights():
    """Load pattern weights from file (persists across restarts)."""
    try:
        if os.path.exists(WEIGHT_FILE):
            with open(WEIGHT_FILE, 'r') as f:
                weights = json.load(f)
            log.info(f"Loaded {len(weights)} pattern weights from {WEIGHT_FILE}")
            return weights
    except Exception as e:
        log.warning(f"Could not load weights: {e}")
    return dict(DEFAULT_WEIGHTS)


def save_weights(weights):
    """Persist updated weights to file."""
    try:
        with open(WEIGHT_FILE, 'w') as f:
            json.dump(weights, f, indent=2)
        log.info("Pattern weights saved")
    except Exception as e:
        log.error(f"Could not save weights: {e}")


# Load on startup
pattern_weights = load_weights()


def analyze_trade_with_claude(trade, patterns_used, indicators, sentiment):
    """
    After a trade closes, send it to Claude for analysis.
    Claude reviews whether the outcome was pattern-driven or externally caused,
    then recommends a weight adjustment.
    """
    global pattern_weights, brain_memory

    if not os.getenv('ANTHROPIC_API_KEY'):
        log.warning("No Anthropic API key — skipping AI analysis")
        return None

    pattern_names = [p['name'] for p in patterns_used] if patterns_used else ['No pattern']
    outcome = trade.get('outcome', 'UNKNOWN')
    pnl     = trade.get('pnl', 0)
    reason  = trade.get('reason', 'UNKNOWN')

    prompt = f"""You are the AI brain of SOLBOT, an autonomous SOL/USD crypto trading bot running on 1-hour candles.

A trade just closed. Analyze it and give a brief, technical post-mortem.

TRADE DETAILS:
- Type: {trade.get('type', 'UNKNOWN')}
- Entry: ${trade.get('entry', 0):.4f}
- Exit: ${trade.get('exit', 0):.4f}  
- P&L: ${pnl:+.2f}
- Outcome: {outcome}
- Closed by: {reason}

SIGNALS USED:
- Patterns detected: {', '.join(pattern_names)}
- RSI at entry: {indicators.get('rsi', 'N/A')}
- MACD signal: {'Bullish' if indicators.get('macd_bull') else 'Bearish'}
- BB position: {indicators.get('bb_pos', 'N/A')}%
- News sentiment: {sentiment:.2f} ({'+bullish' if sentiment > 0.2 else '-bearish' if sentiment < -0.2 else 'neutral'})

CURRENT PATTERN WEIGHTS:
{json.dumps({p: pattern_weights.get(p, 'N/A') for p in [pat['id'] for pat in patterns_used]}, indent=2) if patterns_used else 'No patterns used'}

Respond in exactly this JSON format (no markdown, no extra text):
{{
  "verdict": "pattern_valid | pattern_failed | external_factor",
  "explanation": "2-3 sentence technical explanation of why the trade won or lost",
  "weight_changes": {{"pattern_id": delta_integer}},
  "key_lesson": "one actionable insight for future SOL trades",
  "confidence": 0-100
}}

Rules:
- If a flash crash/pump (>4% single candle) caused the loss, use "external_factor" and set weight_changes to {{}}
- Weight changes: max +5 for wins, max -4 for losses
- Be conservative — only adjust weights you're confident about
- Keep explanation under 60 words"""

    try:
        response = client.messages.create(
            model='claude-sonnet-4-20250514',
            max_tokens=500,
            messages=[{'role': 'user', 'content': prompt}]
        )
        raw = response.content[0].text.strip()
        analysis = json.loads(raw)

        # Apply weight changes
        changes_applied = {}
        if analysis.get('verdict') != 'external_factor':
            for pat_id, delta in analysis.get('weight_changes', {}).items():
                if pat_id in pattern_weights:
                    old = pattern_weights[pat_id]
                    # Clamp weights between 40-95
                    new_weight = max(40, min(95, old + delta))
                    pattern_weights[pat_id] = new_weight
                    changes_applied[pat_id] = {'from': old, 'to': new_weight, 'delta': delta}
            save_weights(pattern_weights)

        # Store in brain memory
        memory_entry = {
            'timestamp':      datetime.now().isoformat(),
            'trade':          trade,
            'patterns':       pattern_names,
            'verdict':        analysis.get('verdict'),
            'explanation':    analysis.get('explanation'),
            'key_lesson':     analysis.get('key_lesson'),
            'weight_changes': changes_applied,
            'confidence':     analysis.get('confidence', 0),
            'sentiment':      sentiment,
        }
        brain_memory.append(memory_entry)
        if len(brain_memory) > 100:
            brain_memory.pop(0)

        log.info(f"AI analysis complete: {analysis.get('verdict')} | Changes: {changes_applied}")
        return memory_entry

    except json.JSONDecodeError as e:
        log.error(f"Claude returned invalid JSON: {e}")
        return None
    except Exception as e:
        log.error(f"AI analysis error: {e}")
        return None


def get_brain_summary():
    """Return brain stats for the dashboard."""
    total = len(brain_memory)
    verdicts = [m['verdict'] for m in brain_memory]
    patterns_adjusted = set()
    for m in brain_memory:
        patterns_adjusted.update(m.get('weight_changes', {}).keys())

    return {
        'total_analyses':     total,
        'pattern_valid':      verdicts.count('pattern_valid'),
        'pattern_failed':     verdicts.count('pattern_failed'),
        'external_factor':    verdicts.count('external_factor'),
        'patterns_adjusted':  len(patterns_adjusted),
        'current_weights':    dict(pattern_weights),
        'recent_memory':      brain_memory[-10:][::-1],
        'top_patterns':       sorted(pattern_weights.items(), key=lambda x: x[1], reverse=True)[:5],
    }


def get_pattern_weight(pattern_id):
    """Get current confidence weight for a pattern."""
    return pattern_weights.get(pattern_id, DEFAULT_WEIGHTS.get(pattern_id, 70))
