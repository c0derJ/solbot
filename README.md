# SOLBOT — Autonomous SOL/USD Trading Bot

AI-powered trading bot for Solana (SOL/USD) on Kraken.
Features: TA engine, pattern detection, news sentiment, Claude AI learning.

## Quick Deploy to Railway

### Step 1 — Push to GitHub
1. Create a new repository on github.com (name it `solbot`)
2. Upload all files from this folder to that repository

### Step 2 — Deploy on Railway
1. Go to railway.app
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your `solbot` repository
4. Railway will auto-detect and build

### Step 3 — Set Environment Variables
In Railway dashboard → your project → Variables, add:

```
ANTHROPIC_API_KEY=your_key_here
PAPER_TRADING=true
TRADE_AMOUNT=500
STOP_LOSS_PCT=3
TAKE_PROFIT_PCT=6
SCRAPER_ACCOUNT=solidintel_x
```

Leave KRAKEN_API_KEY and KRAKEN_API_SECRET blank for paper trading.

### Step 4 — Get your URL
Railway gives you a public URL like `solbot-production.up.railway.app`
Open it in any browser — your bot dashboard is live.

## Files
- `app.py` — Flask web server + WebSocket
- `engine.py` — Kraken data, TA indicators, pattern detection, paper trading
- `scraper.py` — @solidintel_x news scraper + sentiment analysis
- `ai_brain.py` — Claude AI trade analysis + pattern weight learning
- `static/index.html` — Main dashboard UI
- `static/brain.html` — AI Brain pattern library UI

## Paper Trading
Paper trading mode is ON by default. The bot:
- Fetches real SOL/USD prices from Kraken
- Runs real technical analysis
- Makes real trading decisions
- But executes with fake money ($500 starting balance)

Switch to live trading by setting `PAPER_TRADING=false` and adding Kraken API keys.
